# Physics-Informed Neural Networks (PINNs) for the Nonlinear Schrödinger (NLS) Equation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, self-contained implementation of **Physics-Informed Neural Networks (PINNs)** using **PyTorch** to solve the 1D **Nonlinear Schrödinger (NLS) Equation**, simulating quantum soliton dynamics and complex wave propagation.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement & Formulation](#-problem-statement--formulation)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Getting Started & Running in Colab](#-getting-started--running-in-colab)
- [Results & Visualizations](#-results--visualizations)
- [References](#-references)

---

## 🚀 Project Overview
Physics-Informed Neural Networks (PINNs) embed physical laws—expressed as Partial Differential Equations (PDEs)—directly into the neural network's loss function as a regularization term. This repository models quantum bright soliton dynamics governed by the NLS equation without requiring dense empirical training datasets, using an analytic self-contained generation method.

---

## 🧮 Problem Statement & Formulation

### **1. The Problem Solved**
The project models complex wave propagation and quantum soliton behavior in nonlinear media. 

### **2. Governing Equation**
The dimensionless 1D Nonlinear Schrödinger equation for the complex wave field $h(x,t) = u(x,t) + iv(x,t)$ is given by[cite: 5]:

$$iu_t + \frac{1}{2}u_{xx} + \vert{}u\vert{}^2u = 0$$

Splitting into real ($u$) and imaginary ($v$) components yields the system of PDEs enforced inside the residual loss ($f_u, f_v$):
- **Real Residual:** $f_u = u_t + \frac{1}{2}v_{xx} + (u^2 + v^2)v = 0$
- **Imaginary Residual:** $f_v = v_t - \frac{1}{2}u_{xx} - (u^2 + v^2)u = 0$

### **3. Analytic Benchmark Solution**
To bypass external data dependencies, the exact bright soliton reference solution is generated analytically using the standard formula:
$$h(x,t) = 2 \, \text{sech}(x) \, e^{i \, 0.5 \, t}$$

---

## ✨ Key Features
* **Modern Framework:** Built entirely in **PyTorch** leveraging automatic differentiation (`torch.autograd`) for exact spatial and temporal derivative computations.
* **Self-Contained Data Generation:** Generates benchmark bright soliton solutions via hyperbolic secant (`sech`) and complex exponential formulations internally.
* **Hybrid Two-Step Optimization:**
  1. **Adam Optimizer:** For fast global convergence and escaping local minima.
  2. **L-BFGS Optimizer:** For high-precision scientific fine-tuning.
* **Comprehensive Visualization:** Automatically renders side-by-side comparison heatmaps and multi-time slice 1D profile graphs using `matplotlib`.

---

## 📂 Repository Structure
```text
├── nls_pinn.py          # Main PyTorch implementation script
├── README.md            # Project documentation
└── assets/              # Output heatmaps and line plots (optional)
