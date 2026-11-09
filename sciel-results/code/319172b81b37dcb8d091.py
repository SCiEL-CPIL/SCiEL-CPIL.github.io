"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_profile(h):
    """Compute overlap profile: for each shift k, sum h[i] * (1 - h[i+k])"""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        # Circular shift
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1 - shifted))
    return overlaps

def _score(h):
    """Score is the maximum overlap (lower is better)"""
    return np.max(_overlap_profile(h))

def _fix_sum(h, target_sum=None):
    """Ensure sum equals N/2 and values in [0,1]"""
    if target_sum is None:
        target_sum = N / 2
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-10:
        return h
    # Distribute correction evenly, respecting bounds
    n_adjust = len(h)
    per_elem = diff / n_adjust
    h_new = h + per_elem
    h_new = np.clip(h_new, 0, 1)
    # If still off, do iterative adjustment
    for _ in range(100):
        diff = target_sum - np.sum(h_new)
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            # Need to increase some elements
            can_increase = h_new < 1 - 1e-12
            if not np.any(can_increase):
                break
            h_new[can_increase] += diff / np.sum(can_increase)
        else:
            can_decrease = h_new > 1e-12
            if not np.any(can_decrease):
                break
            h_new[can_decrease] += diff / np.sum(can_decrease)
        h_new = np.clip(h_new, 0, 1)
    return h_new

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with constant 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _score(best_h)
    
    # Try several random initializations and local optimization
    rng = np.random.default_rng(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization with sum constraint
        h_init = rng.uniform(0, 1, N)
        h_init = _fix_sum(h_init)
        
        # Objective: minimize max overlap (use smooth approximation)
        def objective(x):
            x = _fix_sum(x)
            overlaps = _overlap_profile(x)
            # Smooth max using log-sum-exp
            temperature = 0.01
            return temperature * np.log(np.sum(np.exp(overlaps / temperature)))
        
        def jac(x):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            f0 = objective(x)
            for i in range(N):
                x_plus = x.copy()
                x_plus[i] += eps
                x_plus = _fix_sum(x_plus)
                x_minus = x.copy()
                x_minus[i] -= eps
                x_minus = _fix_sum(x_minus)
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad
        
        try:
            result = minimize(
                objective, 
                h_init, 
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = _fix_sum(result.x)
            score_opt = _score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # Final high-precision polish if time remains
    if time.time() < deadline - 20:
        def objective_precise(x):
            x = _fix_sum(x)
            overlaps = _overlap_profile(x)
            temperature = 0.001
            return temperature * np.log(np.sum(np.exp(overlaps / temperature)))
        
        try:
            result = minimize(
                objective_precise,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_final = _fix_sum(result.x)
            score_final = _score(h_final)
            if score_final < best_score:
                best_h = h_final
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
