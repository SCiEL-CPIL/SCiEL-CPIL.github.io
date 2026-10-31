"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    N = len(h)
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        overlap = np.sum(h * (1.0 - shifted))
        if overlap > max_ov:
            max_ov = overlap
    return max_ov

def _objective(params):
    h = params
    return _max_overlap(h)

def _constraints_sum(params):
    return np.sum(params) - N / 2.0

def _constraints_bounds(params):
    # Return violations for bounds [0, 1]
    lower_viol = np.maximum(0, -params)
    upper_viol = np.maximum(0, params - 1.0)
    return np.concatenate([lower_viol, upper_viol])

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with the seed solution
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try SLSQP optimization
    try:
        x0 = np.full(N, 0.5)
        constraints = [
            {'type': 'eq', 'fun': _constraints_sum},
        ]
        bounds = [(0.0, 1.0)] * N
        
        result = minimize(
            _objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if time.time() > deadline:
            return best_h
        
        if result.success or result.fun < best_score:
            h_candidate = np.clip(result.x, 0.0, 1.0)
            # Normalize to ensure sum = N/2
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N / 2.0) / s
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            score_candidate = _max_overlap(h_candidate)
            if score_candidate < best_score:
                best_h = h_candidate
                best_score = score_candidate
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_h
    
    # Try a second round with different initial conditions if time permits
    try:
        # Try a triangular or sinusoidal initial guess
        x0_sin = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)
        x0_sin = np.clip(x0_sin, 0.0, 1.0)
        # Normalize
        s = np.sum(x0_sin)
        if s > 0:
            x0_sin = x0_sin * (N / 2.0) / s
            x0_sin = np.clip(x0_sin, 0.0, 1.0)
        
        constraints = [
            {'type': 'eq', 'fun': _constraints_sum},
        ]
        bounds = [(0.0, 1.0)] * N
        
        result = minimize(
            _objective,
            x0_sin,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if time.time() > deadline:
            return best_h
        
        if result.success or result.fun < best_score:
            h_candidate = np.clip(result.x, 0.0, 1.0)
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N / 2.0) / s
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            score_candidate = _max_overlap(h_candidate)
            if score_candidate < best_score:
                best_h = h_candidate
                best_score = score_candidate
    except Exception:
        pass
    
    return best_h
# EVOLVE-BLOCK-END
