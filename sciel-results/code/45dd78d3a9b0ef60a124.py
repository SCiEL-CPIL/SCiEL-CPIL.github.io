"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_overlap(h, k):
    """Compute overlap between h and shifted (1-h) by k positions."""
    shifted_1mh = 1.0 - np.roll(h, -k)
    return np.sum(np.minimum(h, shifted_1mh))


def _max_overlap(h):
    """Compute maximum overlap over all shifts."""
    n = len(h)
    overlaps = np.array([_compute_overlap(h, k) for k in range(n)])
    return np.max(overlaps)


def _objective(h):
    """Smooth objective using log-sum-exp of overlaps."""
    n = len(h)
    overlaps = np.array([_compute_overlap(h, k) for k in range(n)])
    beta = 40.0
    m = np.max(overlaps)
    if m > 1e10:
        return m
    log_sum = m + np.log(np.sum(np.exp(overlaps - m)))
    return log_sum / beta


def _project(h):
    """Project h onto constraint set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    target_sum = N / 2.0
    for _ in range(100):
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        if abs(diff) < 1e-12:
            break
        h = np.clip(h + diff / N, 0.0, 1.0)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with alternating pattern
    h0 = np.zeros(N)
    h0[::2] = 1.0
    h0[1::2] = 0.0
    h0 = _project(h0)
    
    best_h = h0.copy()
    best_score = _max_overlap(h0)
    
    # Try multiple random restarts
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = h0
        else:
            x0 = np.random.uniform(0, 1, N)
            x0 = _project(x0)
        
        # Use SLSQP with constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 'jac': lambda x: np.ones(N)}
        ]
        bounds = [(0.0, 1.0)] * N
        
        try:
            result = minimize(
                _objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
        except Exception:
            continue
        
        if result.success or result.fun < _objective(x0):
            h_candidate = _project(result.x)
            true_score = _max_overlap(h_candidate)
            if true_score < best_score:
                best_score = true_score
                best_h = h_candidate
    
    return best_h
# EVOLVE-BLOCK-END

