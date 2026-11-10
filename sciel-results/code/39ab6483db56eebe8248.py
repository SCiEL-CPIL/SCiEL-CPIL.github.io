"""
Strategy: Chebyshev Spectral Projection with Universal Shape Function.

Instead of optimizing nodes/weights to minimize error against a fixed set of 
cosmologies (which is prone to overfitting and local minima), we assume that 
the "shape" of the integrand $f(x) = 1/E(z \cdot x)$ is similar across 
cosmologies when normalized by a reference shape.

We define a reference integrand $S(x)$ from a representative cosmology (e.g., 
w0=-1, wa=0). We then seek a quadrature rule that is exact for the space of 
functions spanned by $S(x) \cdot T_k(x)$ for $k=0 \dots 7$, where $T_k$ are 
Chebyshev polynomials of the first kind on [0,1].

This ensures that the rule perfectly captures the "universal" part of the 
integrand's structure. The residual error will come from the "deviation" of 
other cosmologies from this reference shape, which is typically smaller and 
smoother than the full integrand.

We optimize the nodes and weights to satisfy these moment conditions (least 
squares) and then perform a final local minimax refinement to handle the 
residuals.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
import sys

# Deadline guard
START_TIME = time.time()
DEADLINE = 85  # seconds, leave 5s margin

def E_z(z, w0, wa):
    """Calculate E(z) = H(z)/H0 for flat universe with CPL w(z) = w0 + wa*(1 - 1/(1+z))"""
    if z < 0:
        return 1.0
    # E(z)^2 = (1+z)^(-3*w0) * (1 - wa * z / (1+z))^(1.5*wa)
    # Using vectorized operations where possible, but keeping scalar for clarity in loops if needed
    term1 = (1 + z) ** (-3 * w0)
    base = 1 - wa * z / (1 + z)
    if base < 0:
        base = 1e-12
    term2 = base ** (1.5 * wa)
    return np.sqrt(term1 * term2)

def true_integral(z, w0, wa):
    """Calculate the true integral part: int_0^1 dx / E(z*x)"""
    if z == 0:
        return 1.0
    def integrand(x):
        return 1.0 / E_z(z * x, w0, wa)
    val, _ = quad(integrand, 0, 1, limit=100)
    return val

def quadrature_approx(nodes, weights, z, w0, wa):
    """Calculate the quadrature approximation: sum w_i / E(z * x_i)"""
    if z == 0:
        return 1.0
    z_vals = z * nodes
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(weights / E_vals)

# Reference cosmology for the "shape function"
# LambdaCDM is a good anchor as it's the standard model
W0_REF = -1.0
WA_REF = 0.0

def shape_function(x, z_ref=1.0):
    """
    Universal shape function S(x) derived from the reference cosmology.
    We use z=1.0 as a representative redshift to define the shape on [0,1].
    """
    return 1.0 / E_z(z_ref * x, W0_REF, WA_REF)

def chebyshev_T(k, x):
    """Chebyshev polynomial of the first kind T_k mapped to [0,1]."""
    # T_k(t) on [-1,1]. Map x in [0,1] to t in [-1,1]: t = 2x - 1
    t = 2 * x - 1
    # Use numpy polynomial or recurrence
    if k == 0:
        return np.ones_like(x)
    if k == 1:
        return t
    # Recurrence: T_{k+1} = 2t T_k - T_{k-1}
    T_k_minus_1 = np.ones_like(x)
    T_k_curr = t
    for i in range(2, k + 1):
        T_k_next = 2 * t * T_k_curr - T_k_minus_1
        T_k_minus_1 = T_k_curr
        T_k_curr = T_k_next
    return T_k_curr

def spectral_moment_objective(params):
    """
    Objective: Minimize the error in satisfying the moment conditions 
    int S(x) T_k(x) dx = sum w_i S(x_i) T_k(x_i) for k=0..7.
    Also include a small penalty for node spacing to avoid clustering.
    """
    if time.time() > DEADLINE - 5:
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
    
    # Compute S(x) at nodes
    S_nodes = shape_function(x)
    
    # Compute moments for k=0..7
    error = 0.0
    for k in range(8):
        # True moment: int_0^1 S(x) T_k(x) dx
        def integrand_moment(x_val):
            return shape_function(x_val) * chebyshev_T(k, x_val)
        
        # Use quad for high precision reference moment
        true_moment, _ = quad(integrand_moment, 0, 1, limit=100)
        
        # Approximate moment
        approx_moment = np.sum(w_norm * S_nodes * chebyshev_T(k, x))
        
        error += (true_moment - approx_moment) ** 2
        
    # Penalty for non-uniform spacing (encourages stability)
    diffs = np.diff(x)
    spacing_var = np.var(diffs)
    
    return error + 1e-4 * spacing_var

def minimax_refinement_objective(params):
    """
    Final refinement: Minimize worst-case relative error over a diverse set 
    of cosmologies and redshifts.
    """
    if time.time() > DEADLINE - 2:
        return 1e9

    x = params[:8]
    w = params[8:]
    
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
    
    # Representative cosmologies
    COSMOLOGIES = [
        (-1.0, 0.0), (-1.0, 0.5), (-0.9, 0.3), (-1.1, 0.0),
        (-0.5, 0.0), (-1.0, -0.5), (-0.8, 0.2), (-1.2, 0.1),
        (-0.95, 0.1), (-1.0, 0.8), (-0.7, 0.4)
    ]
    Z_VALS = np.linspace(0.1, 3.0, 20)
    
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
    
    params_init = np.concatenate([x_init, w_init])
    
    bounds = [(0.0, 1.0)] * 8 + [(1e-6, 1.0)] * 8
    
    best_params = params_init
    best_score = 1e9
    
    # Phase 1: Spectral Moment Matching
    # This projects the rule onto the subspace defined by the reference shape.
    try:
        if time.time() < DEADLINE - 30:
            # Use L-BFGS-B for smooth convergence to the moment conditions
            result1 = minimize(
                spectral_moment_objective, 
                params_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result1.fun < best_score:
                best_score = result1.fun
                best_params = result1.x
    except Exception as e:
        pass
        
    # Phase 2: Minimax Refinement
    # Adjust nodes/weights slightly to minimize the actual worst-case error 
    # over the broader class of cosmologies, starting from the spectral solution.
    try:
        if time.time() < DEADLINE - 5:
            # Powell is good for non-smooth minimax objectives
            result2 = minimize(
                minimax_refinement_objective, 
                best_params, 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 100, 'maxfev': 2000}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
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
