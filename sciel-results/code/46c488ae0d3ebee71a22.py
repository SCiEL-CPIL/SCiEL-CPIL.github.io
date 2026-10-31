"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_overlaps(h):
    """Compute overlap of h with shifted (1-h) for all circular shifts."""
    n = len(h)
    ones_minus_h = 1.0 - h
    overlaps = np.zeros(n)
    for s in range(n):
        # h shifted right by s: h[(i-s) % n]
        shifted = np.roll(h, s)
        overlaps[s] = np.sum(shifted * ones_minus_h)
    return overlaps


def _max_overlap(h):
    """Maximum overlap of h with shifted (1-h)."""
    return np.max(_compute_overlaps(h))


def _smooth_max(x, temperature=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x = np.asarray(x)
    m = np.max(x)
    if temperature <= 0:
        return m
    return m + temperature * np.log(np.sum(np.exp((x - m) / temperature)))


def _objective(params):
    """Objective: smooth max of overlaps."""
    h = params
    n = len(h)
    ones_minus_h = 1.0 - h
    overlaps = np.zeros(n)
    for s in range(n):
        shifted = np.roll(h, s)
        overlaps[s] = np.sum(shifted * ones_minus_h)
    return _smooth_max(overlaps, temperature=0.5)


def _objective_grad(params):
    """Gradient of smooth max of overlaps."""
    h = params
    n = len(h)
    ones_minus_h = 1.0 - h
    overlaps = np.zeros(n)
    for s in range(n):
        shifted = np.roll(h, s)
        overlaps[s] = np.sum(shifted * ones_minus_h)
    
    # Gradient of log-sum-exp
    temp = 0.5
    m = np.max(overlaps)
    exp_vals = np.exp((overlaps - m) / temp)
    weights = exp_vals / np.sum(exp_vals)
    
    # d(overlap_s)/d(h_i)
    grad = np.zeros(n)
    for s in range(n):
        # overlap_s = sum_i h[(i-s)%n] * (1 - h[i])
        # d/d(h_i) = (1 - h[i-s]) + h[i-s] * (-1) = 1 - 2*h[i-s] ... wait
        # overlap_s = sum_j h[j] * (1 - h[(j+s)%n])
        # d/d(h_i): term where j=i: (1 - h[(i+s)%n])
        #           term where (j+s)%n = i, i.e., j = (i-s)%n: h[(i-s)%n] * (-1)
        # So d(overlap_s)/d(h_i) = (1 - h[(i+s)%n]) - h[(i-s)%n]
        for i in range(n):
            grad[i] += weights[s] * ((1 - h[(i + s) % n]) - h[(i - s) % n])
    
    return temp * grad


def construct():
    """
    Returns a sequence of step heights h of length N.
    h_i in [0, 1].
    Sum of h_i is exactly N/2.
    Minimizes the maximum overlap of h with shifted 1-h.
    Uses scipy SLSQP with smooth max objective.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = N
    # Initial: uniform
    h0 = np.full(n, 0.5)
    
    best_h = h0.copy()
    best_score = _max_overlap(best_h)
    
    # Try SLSQP optimization
    try:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - n / 2.0,
             'jac': lambda x: np.ones(n)}
        ]
        bounds = [(0.0, 1.0)] * n
        
        result = minimize(
            _objective,
            h0,
            jac=_objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if result.success or (time.time() < deadline):
            h_opt = np.clip(result.x, 0.0, 1.0)
            # Project to feasible: adjust sum
            diff = n / 2.0 - np.sum(h_opt)
            h_opt += diff / n
            h_opt = np.clip(h_opt, 0.0, 1.0)
            # Re-project more carefully
            for _ in range(100):
                diff = n / 2.0 - np.sum(h_opt)
                if abs(diff) < 1e-12:
                    break
                h_opt += diff / n
                h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score_opt = _max_overlap(h_opt)
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
    except Exception:
        pass
    
    # If we have time, try gradient descent from different initializations
    if time.time() < deadline - 10.0:
        # Try sinusoidal initialization
        for freq in [1, 2, 3]:
            if time.time() > deadline - 10.0:
                break
            h_init = 0.5 + 0.3 * np.sin(2 * np.pi * freq * np.arange(n) / n)
            h_init = np.clip(h_init, 0.0, 1.0)
            # Project
            for _ in range(100):
                diff = n / 2.0 - np.sum(h_init)
                if abs(diff) < 1e-12:
                    break
                h_init += diff / n
                h_init = np.clip(h_init, 0.0, 1.0)
            
            try:
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - n / 2.0,
                     'jac': lambda x: np.ones(n)}
                ]
                bounds = [(0.0, 1.0)] * n
                
                result = minimize(
                    _objective,
                    h_init,
                    jac=_objective_grad,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if result.success or (time.time() < deadline):
                    h_opt = np.clip(result.x, 0.0, 1.0)
                    for _ in range(100):
                        diff = n / 2.0 - np.sum(h_opt)
                        if abs(diff) < 1e-12:
                            break
                        h_opt += diff / n
                        h_opt = np.clip(h_opt, 0.0, 1.0)
                    
                    score_opt = _max_overlap(h_opt)
                    if score_opt < best_score:
                        best_h = h_opt
                        best_score = score_opt
            except Exception:
                pass
    
    # Final projection to ensure exact sum
    for _ in range(200):
        diff = n / 2.0 - np.sum(best_h)
        if abs(diff) < 1e-12:
            break
        best_h += diff / n
        best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final correction for exact sum
    diff = n / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Distribute remaining diff to elements not at boundary
        mask = (best_h > 1e-9) & (best_h < 1.0 - 1e-9)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
        else:
            best_h += diff / n
    
    return best_h
# EVOLVE-BLOCK-END
