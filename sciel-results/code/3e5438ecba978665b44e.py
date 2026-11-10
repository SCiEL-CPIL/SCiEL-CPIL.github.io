"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 85.0  # seconds, leave 5s margin

def _time_exceeded():
    return (time.time() - _START_TIME) > TIME_LIMIT

def _E(z, Om, w0, w1):
    """E(z) = H(z)/H0 for flat CPL universe."""
    z = np.asarray(z)
    a = 1.0 / (1.0 + z)
    # Dark energy density ratio: (rho_de/rho_de0) = (1+z)^(3(1+w0+w1)) * exp(-3*w1*ln(1+z))
    ln1pz = np.log(1.0 + z)
    exponent = 3.0 * (1.0 + w0 + w1)
    exp_term = np.exp(-3.0 * w1 * ln1pz)
    de_term = np.power(1.0 + z, exponent) * exp_term
    matter_term = Om * np.power(1.0 + z, 3.0)
    return np.sqrt(matter_term + (1.0 - Om) * de_term)

def _true_integral(z, Om, w0, w1):
    """Compute ∫₀^z dz'/E(z') via adaptive quadrature."""
    if z <= 0:
        return 0.0
    integrand = lambda zp: 1.0 / _E(zp, Om, w0, w1)
    result, _ = quad(integrand, 0.0, z, limit=200, epsabs=1e-14, epsrel=1e-14)
    return result

def _quadrature_integral(z, nodes, weights, Om, w0, w1):
    """Compute the quadrature approximation of ∫₀^z dz'/E(z')."""
    z_samples = z * nodes
    E_vals = _E(z_samples, Om, w0, w1)
    return np.sum(weights / E_vals)

def _compute_errors(nodes, weights, cosmologies, z_values):
    """Compute relative errors for all test cases."""
    errors = []
    for Om, w0, w1 in cosmologies:
        for z in z_values:
            true_val = _true_integral(z, Om, w0, w1)
            quad_val = _quadrature_integral(z, nodes, weights, Om, w0, w1)
            if abs(true_val) > 1e-15:
                rel_err = abs(quad_val - true_val) / abs(true_val)
                errors.append(rel_err)
    return np.array(errors)

def _objective(x):
    """Objective: smooth minimax of relative errors."""
    if _time_exceeded():
        return 1e10
    n = 8
    # Decode parameters
    # First 8: log-gaps for nodes (softmax)
    # Next 8: log-weights (softmax)
    node_logits = x[:8]
    weight_logits = x[8:16]
    
    # Softmax for gaps -> nodes
    node_gaps = np.exp(node_logits - np.max(node_logits))
    node_gaps = node_gaps / node_gaps.sum()
    nodes = np.cumsum(node_gaps)
    
    # Softmax for weights
    w = np.exp(weight_logits - np.max(weight_logits))
    weights = w / w.sum()
    
    # Check constraints
    if not np.all(np.diff(nodes) > 0):
        return 1e10
    if not np.all(weights > 0):
        return 1e10
    if abs(weights.sum() - 1.0) > 1e-12:
        return 1e10
    
    # Compute errors
    errors = _compute_errors(nodes, weights, _COSMOLOGIES, _Z_VALUES)
    if len(errors) == 0:
        return 1e10
    
    # Smooth minimax: log-sum-exp of squared errors
    max_err = np.max(errors)
    if max_err < 1e-16:
        return 0.0
    # Use a large power to approximate L-infinity
    p = 20
    smooth_max = (np.sum(np.power(errors, p)) / len(errors)) ** (1.0 / p)
    return smooth_max

def construct():
    global _COSMOLOGIES, _Z_VALUES
    _START_TIME = time.time()
    
    # Define test cosmologies: (Om, w0, w1)
    _COSMOLOGIES = [
        (0.27, -1.0, 0.0),    # Lambda CDM
        (0.30, -1.0, 0.0),    # Lambda CDM
        (0.25, -1.0, 0.0),    # Lambda CDM
        (0.30, -1.2, 0.5),    # CPL
        (0.27, -0.8, -0.5),   # CPL
        (0.30, -1.5, 1.0),    # CPL
        (0.25, -0.5, 0.0),    # w0w1
        (0.35, -1.0, -1.0),   # CPL
        (0.20, -1.0, 0.5),    # CPL
        (0.30, -1.0, -0.5),   # CPL
        (0.27, -0.9, 0.3),    # CPL
        (0.30, -1.1, -0.3),   # CPL
        (0.25, -1.3, 0.7),    # CPL
        (0.35, -0.7, -0.7),   # CPL
        (0.20, -1.0, 0.0),    # Lambda CDM
        (0.30, -1.0, 0.3),    # CPL
    ]
    
    # Test redshifts: log-spaced from 0.01 to 5
    _Z_VALUES = np.logspace(np.log10(0.01), np.log10(5.0), 20)
    
    # Initial guess: midpoint rule
    n = 8
    x0 = np.zeros(16)
    # Set initial nodes to midpoint: gaps are all 1/8
    x0[:8] = np.log(np.full(8, 1.0/8))
    # Set initial weights to uniform
    x0[8:16] = np.log(np.full(8, 1.0/8))
    
    best_x = x0.copy()
    best_score = _objective(x0)
    
    # Try multiple optimization methods
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        if _time_exceeded():
            break
        try:
            result = minimize(_objective, best_x, method=method, 
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Final decode
    node_logits = best_x[:8]
    weight_logits = best_x[8:16]
    
    node_gaps = np.exp(node_logits - np.max(node_logits))
    node_gaps = node_gaps / node_gaps.sum()
    nodes = np.cumsum(node_gaps)
    
    w = np.exp(weight_logits - np.max(weight_logits))
    weights = w / w.sum()
    
    # Ensure strict positivity and ordering
    if not np.all(np.diff(nodes) > 1e-15):
        # Fallback to midpoint
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0/n)
    if not np.all(weights > 0):
        weights = np.full(n, 1.0/n)
    
    # Normalize weights to sum exactly to 1
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
