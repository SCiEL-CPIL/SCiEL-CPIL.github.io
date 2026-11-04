"""Optimized program for the suite task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _evaluate(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    auto = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(auto) / (s * s))

def construct():
    deadline = time.time() + 120.0  # safety margin
    
    # Start with a smooth positive function
    # Use a raised cosine: h_i = 1 + a*cos(2*pi*i/N)
    # Try several initial parameters and optimize
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Parameterized family: h_i = 1 + a*cos(2*pi*i/N) + b*cos(4*pi*i/N)
    # This keeps h positive for small a,b and is smooth
    t = np.arange(N)
    cos1 = np.cos(2 * np.pi * t / N)
    cos2 = np.cos(4 * np.pi * t / N)
    cos3 = np.cos(6 * np.pi * t / N)
    
    def objective(params):
        a, b, c = params
        h = 1.0 + a * cos1 + b * cos2 + c * cos3
        s = np.sum(h)
        if s == 0:
            return 1e10
        auto = np.convolve(h, h, mode='full')
        return abs(2.0 * N * np.max(auto) / (s * s))
    
    # Try a few initial points
    initial_guesses = [
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],
        [0.2, -0.1, 0.0],
        [0.0, 0.0, 0.1],
        [0.1, 0.0, 0.1],
        [0.15, 0.05, 0.05],
    ]
    
    for x0 in initial_guesses:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                a, b, c = res.x
                best_h = 1.0 + a * cos1 + b * cos2 + c * cos3
        except Exception:
            pass
    
    # Also try a pure optimization on the full vector with a smoothness penalty
    if time.time() < deadline - 40:
        # Start from best_h and do a few steps of projected gradient
        h = best_h.copy()
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 20:
                break
            # Compute gradient numerically (approximate)
            # Use finite differences on a few coordinates
            grad = np.zeros(N)
            eps = 1e-5
            for i in range(0, N, 10):  # sample every 10th
                h_plus = h.copy()
                h_plus[i] += eps
                h_minus = h.copy()
                h_minus[i] -= eps
                grad[i] = (_evaluate(h_plus) - _evaluate(h_minus)) / (2 * eps)
            # Scale gradient for the sampled coordinates
            h -= lr * grad
            # Ensure sum is not zero
            s = np.sum(h)
            if abs(s) < 1e-10:
                h += 1e-10
            # Renormalize to keep sum positive
            if np.sum(h) < 0:
                h = -h
            val = _evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Final check
    final_val = _evaluate(best_h)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure finite and valid
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
