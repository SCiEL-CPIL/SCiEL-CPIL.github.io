"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_vector(h):
    """Compute overlap at all shifts: overlap[s] = sum_i h[i] * (1 - h[(i+s)%N])"""
    n = len(h)
    overlaps = np.zeros(n)
    for s in range(n):
        overlaps[s] = np.dot(h, 1.0 - np.roll(h, s))
    return overlaps

def _smooth_max(overlaps, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _objective(params, beta):
    h = params
    overlaps = _overlap_vector(h)
    return _smooth_max(overlaps, beta)

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # Initial guess: trivial solution
    h0 = np.full(N, 0.5)
    
    # Constraints: sum h_i = N/2, 0 <= h_i <= 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    bounds = [(0.0, 1.0)] * N
    
    best_h = h0.copy()
    best_score = _overlap_vector(best_h).max()
    
    # Try SLSQP with smooth objective
    beta = 50.0
    try:
        res = minimize(
            _objective, h0, args=(beta,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if time.time() - start_time > deadline:
            return best_h
        if res.success and res.fun < best_score:
            h_candidate = np.clip(res.x, 0, 1)
            # Re-normalize to satisfy sum constraint
            h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
            h_candidate = np.clip(h_candidate, 0, 1)
            score = _overlap_vector(h_candidate).max()
            if score < best_score:
                best_h = h_candidate
                best_score = score
    except Exception:
        pass
    
    if time.time() - start_time > deadline:
        return best_h
    
    # Try with higher beta for sharper max
    for beta_try in [100.0, 200.0]:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(
                _objective, best_h, args=(beta_try,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if time.time() - start_time > deadline:
                break
            if res.success:
                h_candidate = np.clip(res.x, 0, 1)
                h_candidate = h_candidate / np.sum(h_candidate) * (N/2.0)
                h_candidate = np.clip(h_candidate, 0, 1)
                score = _overlap_vector(h_candidate).max()
                if score < best_score:
                    best_h = h_candidate
                    best_score = score
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
