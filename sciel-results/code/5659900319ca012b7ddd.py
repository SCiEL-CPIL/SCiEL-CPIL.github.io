import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Basis functions - use more harmonics for richer representation
    i = np.arange(N)
    n_harm = 15
    basis = [np.ones(N)]
    for k in range(1, n_harm + 1):
        basis.append(np.cos(2 * np.pi * k * i / N))
        basis.append(np.sin(2 * np.pi * k * i / N))
    
    def compute_h(coeffs):
        h = np.zeros(N)
        for c, b in zip(coeffs, basis):
            h += c * b
        return h
    
    def objective(coeffs):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        h = compute_h(coeffs)
        S = np.sum(h)
        if abs(S) < 1e-6:
            return 1e6
        conv = np.convolve(h, h)
        max_val = np.max(conv)
        return abs(2 * N * max_val / (S ** 2))
    
    # Smooth surrogate using log-sum-exp for soft-max
    def smooth_objective(coeffs):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        h = compute_h(coeffs)
        S = np.sum(h)
        if abs(S) < 1e-6:
            return 1e6
        conv = np.convolve(h, h)
        # Soft-max approximation: log-sum-exp with temperature
        temp = 0.01
        conv_scaled = conv / temp
        # Numerically stable log-sum-exp
        max_conv = np.max(conv_scaled)
        log_sum_exp = max_conv + np.log(np.sum(np.exp(conv_scaled - max_conv)))
        soft_max_val = temp * log_sum_exp
        return abs(2 * N * soft_max_val / (S ** 2))
    
    # Bounds for coefficients
    # a0 (DC) should be large to keep S large and positive
    # AC terms can be negative
    bounds = [(-20, 20)] + [(-2, 2)] * (2 * n_harm)
    
    # Run differential evolution with time guard
    best_coeffs = None
    best_score = 1e6
    
    try:
        def de_objective_with_time(coeffs):
            score = smooth_objective(coeffs)
            if score < best_score:
                nonlocal best_score
                best_score = score
                nonlocal best_coeffs
                best_coeffs = coeffs.copy()
            return score
        
        res = differential_evolution(
            de_objective_with_time, 
            bounds, 
            maxiter=50, 
            popsize=8, 
            seed=42, 
            tol=1e-3,
            polish=False  # We'll do our own polish
        )
        if res.x is not None and res.fun < best_score:
            best_coeffs = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Final polish with L-BFGS-B on the smooth surrogate
    if best_coeffs is not None and time.time() - start_time < DEADLINE - 10:
        try:
            polish_res = minimize(
                smooth_objective,
                best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if polish_res.fun < best_score:
                best_coeffs = polish_res.x
                best_score = polish_res.fun
        except Exception:
            pass
    
    # If we still have time, try a second polish with tighter tolerance
    if best_coeffs is not None and time.time() - start_time < DEADLINE - 5:
        try:
            polish_res2 = minimize(
                smooth_objective,
                best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if polish_res2.fun < best_score:
                best_coeffs = polish_res2.x
                best_score = polish_res2.fun
        except Exception:
            pass
    
    # Compute final h and verify with exact objective
    h = compute_h(best_coeffs)
    S = np.sum(h)
    if abs(S) < 1e-6:
        # Fallback to seed
        return np.ones(N)
    
    # Check if the result is valid and finite
    if not np.all(np.isfinite(h)):
        return np.ones(N)
    
    return h
