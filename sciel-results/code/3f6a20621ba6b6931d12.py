"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max(h, beta=50.0):
    """Soft-max approximation of max of autoconvolution entries."""
    conv = np.convolve(h, h, mode='full')
    # Use soft-max to approximate max
    m = np.max(conv)
    if m > 0:
        return m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    else:
        return np.max(conv)

def _objective(x, beta=50.0):
    """Objective to minimize: 2*N*max(autoconv) / (sum(h))^2"""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv_max = _autoconv_max(h, beta)
    return 2.0 * N * conv_max / (s * s)

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Random initialization with mixed signs
        h0 = rng.standard_normal(N)
        # Ensure sum is not zero
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        
        # Try to optimize using L-BFGS-B
        try:
            res = minimize(_objective, h0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        except Exception:
            continue
        
        h_opt = res.x
        val = _objective_hard(h_opt)
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
        
        # Also try with the soft-max result
        val_soft = _objective(h_opt)
        if val_soft < best_val:
            # Re-evaluate with hard
            val_hard = _objective_hard(h_opt)
            if val_hard < best_val:
                best_val = val_hard
                best_h = h_opt.copy()
    
    # Final polish: try local refinement on best found
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_objective, best_h, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            val = _objective_hard(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
    
    # Safety margin check
    if time.time() > deadline - 5.0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
