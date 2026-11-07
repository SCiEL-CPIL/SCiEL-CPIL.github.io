"""Improved program for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max."""
    c = np.correlate(h, 1.0 - h, mode="full")
    # Soft max: (1/beta) * log(sum(exp(beta * c)))
    c_max = np.max(c)
    # Shift for numerical stability
    shifted = beta * (c - c_max)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
    return float(log_sum / N * 2.0)

def project(h):
    """Project to [0,1]^N with sum = N/2 using bisection on the offset."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final fine-tuning
    for _ in range(20):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        diff = target - s
        free = (h > 1e-15) & (h < 1.0 - 1e-15)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
        else:
            break
    return h

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy 1: Optimize smooth surrogate from multiple cosine initializations
    for k in range(1, 12):
        if remaining() < 10:
            break
        
        for A_init in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]:
            if remaining() < 3:
                break
            
            h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N)
            h_init = project(h_init)
            
            def obj_smooth(x):
                return objective_smooth(project(x), beta=50.0)
            
            try:
                result = minimize(
                    obj_smooth,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 2: Polish the best solution with exact objective
    if remaining() > 15:
        def obj_exact(x):
            return objective(project(x))
        
        try:
            result = minimize(
                obj_exact,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_candidate = project(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    # Strategy 3: Try random perturbations and polish
    if remaining() > 10:
        np.random.seed(42)
        for _ in range(15):
            if remaining() < 3:
                break
            
            h_pert = best_h + 0.03 * np.random.randn(N)
            h_pert = project(h_pert)
            
            def obj_smooth2(x):
                return objective_smooth(project(x), beta=30.0)
            
            try:
                result = minimize(
                    obj_smooth2,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Final polish with very tight tolerance
    if remaining() > 5:
        def obj_exact_tight(x):
            return objective(project(x))
        
        try:
            result = minimize(
                obj_exact_tight,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_candidate = project(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    best_h = project(best_h)
    return best_h
