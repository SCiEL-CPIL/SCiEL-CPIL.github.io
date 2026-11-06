import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

def _E_of_a(a, w0, wa):
    """CPL dark energy model: E(a) = (a/1)^(-3(1+w0+wa)/2) * (1 + w0 + wa*(1-a/1))^(3wa/2)
    For flat universe with matter and CPL dark energy:
    E^2(a) = a^(-3) * Om_m + a^(-3(1+w0+wa)) * (1 + w0 + wa*(1-a))^(3*wa) * Om_de
    with Om_m + Om_de = 1
    """
    Om_m = 0.3
    Om_de = 0.7
    a_safe = np.maximum(a, 1e-10)
    term_matter = a_safe**(-3) * Om_m
    # CPL: rho_de(a) = rho_de0 * (1 + w0 + wa*(1-a))^(3*(1+w0+wa))
    # H^2/H0^2 = Om_m/a^3 + Om_de * (1+w0+wa*(1-a))^(3*(1+w0+wa))
    cpl_factor = (1.0 + w0 + wa * (1.0 - a_safe)) ** (3.0 * (1.0 + w0 + wa))
    term_de = Om_de * cpl_factor
    E_sq = term_matter + term_de
    return np.sqrt(E_sq)

def _integrand(t, z, w0, wa):
    """Integrand for the quadrature: 1/E(a) where a = 1/(1+z*t)"""
    a = 1.0 / (1.0 + z * t)
    return 1.0 / _E_of_a(a, w0, wa)

def _true_integral(z, w0, wa):
    """Compute the true integral using adaptive quadrature"""
    result, _ = quad(lambda t: _integrand(t, z, w0, wa), 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return result

def _quadrature_value(nodes, weights, z, w0, wa):
    """Compute quadrature approximation"""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a_vals, w0, wa)
    return np.sum(weights / E_vals)

def _relative_error(nodes, weights, z, w0, wa):
    """Compute relative error for a given cosmology and redshift"""
    true_val = _true_integral(z, w0, wa)
    approx_val = _quadrature_value(nodes, weights, z, w0, wa)
    if abs(true_val) < 1e-15:
        return 0.0
    return abs((approx_val - true_val) / true_val) * 100.0  # in percent

def _objective(params, test_cases):
    """Objective: maximize the worst-case relative error"""
    n = len(test_cases)
    nodes = params[:8]
    weights = params[8:16]
    
    # Constraint penalties
    # Nodes must be strictly increasing in [0, 1]
    if nodes[0] < 0 or nodes[7] > 1:
        return 1e6
    for i in range(7):
        if nodes[i+1] - nodes[i] < 1e-10:
            return 1e6
    # Weights must be positive
    if np.any(weights <= 0):
        return 1e6
    # Weights sum to 1
    if abs(np.sum(weights) - 1.0) > 1e-6:
        return 1e6
    
    worst_err = 0.0
    for z, w0, wa in test_cases:
        err = _relative_error(nodes, weights, z, w0, wa)
        worst_err = max(worst_err, err)
    return worst_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85 second deadline
    
    # Define test cosmologies: (z, w0, wa)
    # LambdaCDM: w0=-1, wa=0
    # Various CPL models
    test_cases = [
        (0.1, -1.0, 0.0),
        (0.5, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (2.0, -1.0, 0.0),
        (0.5, -0.5, -0.5),
        (1.0, -0.5, -0.5),
        (0.5, -1.0, 0.5),
        (1.0, -1.0, 0.5),
        (0.5, -0.9, -0.1),
        (1.0, -0.9, -0.1),
        (0.5, -1.1, 0.1),
        (1.0, -1.1, 0.1),
        (0.3, -0.8, -0.2),
        (1.5, -0.8, -0.2),
        (0.7, -1.2, 0.3),
        (2.0, -1.2, 0.3),
    ]
    
    # Precompute true integrals
    true_vals = {}
    for z, w0, wa in test_cases:
        true_vals[(z, w0, wa)] = _true_integral(z, w0, wa)
    
    def _objective_fast(params):
        nodes = params[:8]
        weights = params[8:16]
        
        if nodes[0] < 0 or nodes[7] > 1:
            return 1e6
        for i in range(7):
            if nodes[i+1] - nodes[i] < 1e-10:
                return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e6
        
        worst_err = 0.0
        for idx, (z, w0, wa) in enumerate(test_cases):
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_of_a(a_vals, w0, wa)
            approx_val = np.sum(weights / E_vals)
            true_val = true_vals[(z, w0, wa)]
            err = abs((approx_val - true_val) / true_val) * 100.0
            worst_err = max(worst_err, err)
        return worst_err
    
    # Start from Gauss-Legendre
    x, w = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (x + 1.0)
    weights0 = 0.5 * w
    params0 = np.concatenate([nodes0, weights0])
    
    best_params = params0.copy()
    best_score = _objective_fast(best_params)
    
    # Try multiple restarts with random perturbations
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            params = params0.copy()
        else:
            params = params0 + 0.05 * np.random.randn(16)
            # Project back to feasible region roughly
            params[:8] = np.clip(params[:8], 0.01, 0.99)
            params[:8] = np.sort(params[:8])
            params[8:] = np.abs(params[8:])
            params[8:] = params[8:] / np.sum(params[8:])
        
        result = minimize(_objective_fast, params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    nodes = best_params[:8]
    weights = best_params[8:16]
    
    # Final projection: ensure strict ordering and positive weights summing to 1
    nodes = np.sort(nodes)
    # Ensure strictly increasing with minimum spacing
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-8:
            nodes[i] = nodes[i-1] + 1e-8
    # Clip to [0,1]
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strictly increasing after clipping
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    # If nodes went out of [0,1], fall back to Gauss-Legendre
    if nodes[0] < 0 or nodes[7] > 1:
        nodes = nodes0.copy()
        weights = weights0.copy()
    else:
        weights = np.abs(weights)
        if np.sum(weights) < 1e-15:
            weights = weights0.copy()
        else:
            weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
