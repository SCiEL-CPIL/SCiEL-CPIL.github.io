"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
w = 1.0 / 1200.0  # interval width

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on 600 intervals of width w."""
    # Pad h to length 1200 for full convolution
    g = np.zeros(2 * N)
    g[:N] = h
    # Discrete autoconvolution
    c = np.convolve(g, g, mode='full')
    # The continuous autoconvolution values are c * w^2
    # But we need to be careful about the scaling
    return c * (w ** 2)

def objective(h):
    """Objective: normalized maximum of autoconvolution."""
    # Normalize so that integral = 1: sum(h) * w = 1
    s = np.sum(h)
    if s <= 0:
        return 1e10
    h_norm = h / (s * w)  # now sum(h_norm) * w = 1
    ac = compute_autoconv(h_norm)
    return np.max(ac)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # Try to optimize
    best_h = h0.copy()
    best_val = objective(h0)
    
    # Use SLSQP with bounds
    bounds = [(0, None)] * N
    try:
        result = minimize(
            objective,
            h0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_h = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best found so far
        h_out = best_h
        s = np.sum(h_out)
        if s > 0:
            h_out = h_out / (s * w)
        return list(h_out)
    
    # Final normalization
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / (s * w)
    
    return list(best_h)
# EVOLVE-BLOCK-END
