"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400


def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def soft_max(x, beta):
    """Smooth approximation of max(x) using log-sum-exp."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta


def make_obj(beta):
    """Create objective function with given soft-max beta."""
    def obj(params):
        h = params
        s = np.sum(h)
        if s == 0 or not np.all(np.isfinite(h)):
            return 1e10
        conv = np.convolve(h, h)
        # Normalize by sum^2 to make scale-invariant
        peak = soft_max(conv, beta)
        val = 2.0 * N * abs(peak) / (s ** 2)
        return val
    return obj


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Initial guess: raised cosine (low-frequency, flatter autocorrelation)
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    h0 = 1.0 + 0.5 * np.cos(x)  # non-negative, smooth
    h0 = h0 / np.sum(h0) * N  # normalize so sum = N
    
    best_h = h0.copy()
    best_val = compute_value(best_h)
    
    # Try several soft-max beta values, from smooth to sharp
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > DEADLINE - 5.0:
            break
        
        obj = make_obj(beta)
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                obj,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_val:
                h_new = result.x
                val_new = compute_value(h_new)
                if val_new < best_val:
                    best_h = h_new.copy()
                    best_val = val_new
                    h0 = h_new.copy()  # warm start for next beta
        except Exception:
            pass
    
    # Final refinement: direct optimization on the true max using a 
    # penalty-based approach (since max is not differentiable)
    if time.time() < DEADLINE - 10.0:
        # Use a high beta soft-max as a proxy for the true max
        obj_final = make_obj(1000.0)
        try:
            result = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_val:
                h_new = result.x
                val_new = compute_value(h_new)
                if val_new < best_val:
                    best_h = h_new.copy()
                    best_val = val_new
        except Exception:
            pass
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
