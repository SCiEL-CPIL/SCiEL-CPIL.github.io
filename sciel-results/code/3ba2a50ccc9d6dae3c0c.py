"""
Strategy: Global Minimax Optimization via Differential Evolution.

The current approach uses local gradient-based optimization (L-BFGS-B/Powell) 
starting from Gauss-Legendre nodes. This can get trapped in local minima 
specific to the training grid. 

This new strategy uses Differential Evolution (DE), a global stochastic 
optimizer, to directly minimize the worst-case relative error (minimax) 
over a broad, stratified set of cosmologies and redshifts. 

Key differences:
1. Global search: DE explores the entire parameter space, avoiding local minima.
2. Minimax objective: Directly targets the worst-case error, ensuring robustness.
3. Stratified sampling: Uses a diverse set of cosmologies to prevent overfitting.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution
import sys

# Deadline guard
START_TIME = time.time()
DEADLINE = 85  # seconds, leave 5s margin

def E_z(z, w0, wa):
    """Calculate E(z) = H(z)/H0 for flat universe with CPL w(z) = w0 + wa*(1 - 1/(1+z))"""
    if z < 0:
        return 1.0
    # E(z)^2 = (1+z)^(-3*w0) * (1 - wa * z / (1+z))^(1.5*wa)
    term1 = (1 + z) ** (-3 * w0)
    base = 1 - wa * z / (1 + z)
    if base < 0:
        base = 1e-12
    term2 = base ** (1.5 * wa)
    return np.sqrt(term1 * term2)

def true_integral(z, w0, wa):
    """Calculate the true integral part: int_0^1 dx / E(z*x) using high-precision quadrature"""
    if z == 0:
        return 1.0
    # Use Gauss-Legendre with high order for high precision reference
    # 50 nodes is more than enough for smooth integrands
    x_gl, w_gl = np.polynomial.legendre.leggauss(50)
    x_map = 0.5 * (x_gl + 1)
    w_map = w_gl / 2.0
    z_vals = z * x_map
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(w_map / E_vals)

def quadrature_approx(nodes, weights, z, w0, wa):
    """Calculate the quadrature approximation: sum w_i / E(z * x_i)"""
    if z == 0:
        return 1.0
    z_vals = z * nodes
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(weights / E_vals)

# Stratified set of cosmologies for robustness
# Covers a wide range of w0 and wa
COSMOLOGIES = [
    (-1.0, 0.0),  # LambdaCDM
    (-1.0, 0.5),  # CPL with positive wa
    (-1.0, -0.5), # CPL with negative wa
    (-0.5, 0.0),  # Matter-dominated-ish
    (-1.5, 0.0),  # Phantom
    (-0.9, 0.3),  # WMAP-like
    (-1.1, 0.2),  # Slightly phantom
    (-0.8, -0.2), # Quintessence-like
    (-1.2, 0.5),  # Strong phantom + wa
    (-0.7, 0.4),  # Mild quintessence + wa
    (-1.0, 1.0),  # Extreme wa
    (-1.0, -1.0), # Extreme negative wa
]

# Stratified redshifts
Z_VALS = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])

def minimax_objective(params):
    """
    Objective: Minimize the worst-case relative error over the stratified set.
    params: 16-vector [x1..x8, w1..w8]
    """
    if time.time() > DEADLINE - 2:
        return 1e9

    x = params[:8]
    w = params[8:]
    
    # Constraints: x in [0,1] increasing, w > 0, sum(w)=1
    if np.any(x < 0) or np.any(x > 1):
        return 1e9
    if np.any(np.diff(x) <= 0):
        return 1e9
    if np.any(w <= 0):
        return 1e9
        
    w_sum = np.sum(w)
    if w_sum < 1e-6:
        return 1e9
    w_norm = w / w_sum
    
    max_rel_err = 0.0
    for z in Z_VALS:
        for w0, wa in COSMOLOGIES:
            true_val = true_integral(z, w0, wa)
            if true_val < 1e-10:
                continue
            approx_val = quadrature_approx(x, w_norm, z, w0, wa)
            rel_err = abs(approx_val - true_val) / true_val
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                
    return max_rel_err

def construct():
    # Initial guess: Gauss-Legendre nodes and weights
    x_init, w_init = np.polynomial.legendre.leggauss(8)
    x_init = 0.5 * (x_init + 1)
    w_init = w_init / 2.0
    
    # Bounds for Differential Evolution
    # Nodes: strictly increasing in [0,1]
    # We use loose bounds and rely on the penalty in the objective
    # To ensure strict increase, we could parameterize, but DE handles penalties well
    bounds = [(1e-6, 1 - 1e-6)] * 8 + [(1e-6, 1.0)] * 8
    
    best_params = np.concatenate([x_init, w_init])
    best_score = minimax_objective(best_params)
    
    try:
        if time.time() < DEADLINE - 5:
            # Differential Evolution settings
            # popsize=10 means 10*16=160 individuals per generation
            # maxiter=50 means 50 generations
            # Total function evals ~ 8000, which is manageable if true_integral is fast
            # true_integral uses 50-point Gauss-Legendre, which is very fast
            result = differential_evolution(
                minimax_objective, 
                bounds, 
                maxiter=50, 
                popsize=10,
                tol=1e-8,
                seed=42,
                polish=True,  # Use L-BFGS-B to polish the result
                polish_options={'maxiter': 100}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
    except Exception as e:
        pass

    # Extract and sanitize
    x = best_params[:8]
    w = best_params[8:]
    
    # Ensure strict increase and bounds
    x = np.sort(x)
    x = np.clip(x, 1e-8, 1 - 1e-8)
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-8
    x = np.clip(x, 1e-8, 1 - 1e-8)
    
    # Normalize weights
    w = np.abs(w)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        w = np.full(8, 1.0/8.0)
    else:
        w = w / w_sum
        
    return {"nodes": x, "weights": w}
