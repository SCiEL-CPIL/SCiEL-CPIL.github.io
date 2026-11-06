"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

_start_time = time.time()
_TIME_LIMIT = 80.0  # Safety margin before 90s kill


def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for CPL model in flat universe."""
    # E^2 = Om_m * a^-3 + Om_r * a^-4 + (1-Om_m-Om_r) * a^(-3(1+w0+wa))
    # For simplicity, use Om_m=0.3, Om_r=0.0 (negligible), w0, wa
    Om_m = 0.3
    Om_r = 0.0
    Om_de = 1.0 - Om_m - Om_r
    if Om_de <= 0:
        # Fallback
        return np.sqrt(Om_m * a**(-3) + Om_r * a**(-4))
    return np.sqrt(Om_m * a**(-3) + Om_r * a**(-4) + Om_de * a**(-3*(1+w0+wa)))


def _true_integral(z, w0, wa):
    """Compute dL/(c/H0) via adaptive quadrature."""
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        return zp / _E_of_a(a, w0, wa)
    
    val, _ = quad(integrand, 0, z, limit=200)
    return (1 + z) * val


def _quad_integral(z, nodes, weights, w0, wa):
    """Compute dL/(c/H0) via the 8-node quadrature rule."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a_vals, w0, wa)
    return (1 + z) * z * np.sum(weights / E_vals)


def _relative_error(z, nodes, weights, w0, wa):
    """Relative error in percent for a given z and model."""
    true_val = _true_integral(z, w0, wa)
    quad_val = _quad_integral(z, nodes, weights, w0, wa)
    if abs(true_val) < 1e-15:
        return 0.0
    return abs(quad_val - true_val) / abs(true_val) * 100.0


def _worst_case_error(params, models, z_values):
    """Compute worst-case relative error across all models and redshifts."""
    n = 8
    # Unpack: first 8 are nodes (transformed), next 8 are log-weights
    node_params = params[:n]
    log_weights = params[n:2*n]
    
    # Transform nodes to (0,1) via sigmoid-like mapping
    nodes = 1.0 / (1.0 + np.exp(-node_params))
    # Ensure strictly increasing by sorting
    nodes = np.sort(nodes)
    # Add small epsilon to ensure strict inequality and stay in (0,1)
    eps = 1e-10
    nodes = np.clip(nodes, eps, 1-eps)
    # Force strictly increasing
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + eps
    
    # Weights from log-space, normalized
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    max_err = 0.0
    for w0, wa in models:
        for z in z_values:
            err = _relative_error(z, nodes, weights, w0, wa)
            if err > max_err:
                max_err = err
    
    return max_err


def _objective(params):
    """Objective function for optimization."""
    if time.time() - _start_time > _TIME_LIMIT:
        return 1e10  # Penalty for timeout
    
    models = [
        (0.0, 0.0),    # LambdaCDM
        (-0.5, 0.5),   # CPL
        (-1.0, 0.0),   # w=-1
        (-0.9, 0.3),   # Another CPL
        (-1.2, 0.5),   # Phantom
        (-0.8, -0.2),  # Another
    ]
    z_values = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    return _worst_case_error(params, models, z_values)


def construct():
    global _start_time
    _start_time = time.time()
    
    n = 8
    
    # Initial guess: midpoint rule
    nodes0 = (np.arange(n) + 0.5) / n
    # Inverse transform for initial params
    node_params0 = np.log(nodes0 / (1 - nodes0))
    log_weights0 = np.full(n, np.log(1.0/n))
    params0 = np.concatenate([node_params0, log_weights0])
    
    best_params = params0.copy()
    best_score = _objective(params0)
    
    # Try multiple optimization methods
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        try:
            result = minimize(
                _objective,
                params0,
                method=method,
                options={
                    'maxiter': 200,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
                params0 = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() - _start_time < _TIME_LIMIT - 10:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract final nodes and weights
    node_params = best_params[:n]
    log_weights = best_params[n:2*n]
    
    nodes = 1.0 / (1.0 + np.exp(-node_params))
    nodes = np.sort(nodes)
    eps = 1e-10
    nodes = np.clip(nodes, eps, 1-eps)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + eps
    
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    # Ensure weights are strictly positive
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
