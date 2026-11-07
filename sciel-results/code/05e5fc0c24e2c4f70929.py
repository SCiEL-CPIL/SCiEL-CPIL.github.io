"""Improved program for the Erdos minimum overlap task."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _objective(h):
    """
    Compute the objective value: max(correlate(h, 1-h, mode='full')) / N * 2
    Lower is better.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _sum_constraint(h):
    """
    Return the sum constraint violation: sum(h) - N/2
    """
    return np.sum(h) - N / 2

def _bounds():
    """
    Return bounds for each variable: [0, 1]
    """
    return [(0.0, 1.0)] * N

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with the seed: all 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _objective(h0)
    
    # Try multiple optimization strategies
    # Strategy 1: SLSQP with sum constraint
    try:
        res = minimize(
            _objective,
            h0,
            method='SLSQP',
            bounds=_bounds(),
            constraints=[{'type': 'eq', 'fun': _sum_constraint}],
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res.success and np.isfinite(res.fun):
            h_candidate = res.x
            # Project back to feasible set
            h_candidate = np.clip(h_candidate, 0, 1)
            # Adjust to satisfy sum constraint
            diff = N / 2 - np.sum(h_candidate)
            if abs(diff) > 1e-15:
                # Distribute the difference evenly, but respect bounds
                for _ in range(100):
                    h_candidate += diff / N
                    h_candidate = np.clip(h_candidate, 0, 1)
                    new_diff = N / 2 - np.sum(h_candidate)
                    if abs(new_diff) < 1e-12:
                        break
                    diff = new_diff
            val = _objective(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return best_h
    
    # Strategy 2: Try different initializations
    rng = np.random.default_rng(42)
    for init_seed in range(5):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random initialization with sum constraint
        h_rand = rng.uniform(0, 1, N)
        # Scale to satisfy sum constraint
        h_rand = h_rand / np.sum(h_rand) * (N / 2)
        h_rand = np.clip(h_rand, 0, 1)
        # Fine-tune sum
        diff = N / 2 - np.sum(h_rand)
        for _ in range(100):
            h_rand += diff / N
            h_rand = np.clip(h_rand, 0, 1)
            new_diff = N / 2 - np.sum(h_rand)
            if abs(new_diff) < 1e-12:
                break
            diff = new_diff
        
        try:
            res = minimize(
                _objective,
                h_rand,
                method='SLSQP',
                bounds=_bounds(),
                constraints=[{'type': 'eq', 'fun': _sum_constraint}],
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if res.success and np.isfinite(res.fun):
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, 1)
                diff = N / 2 - np.sum(h_candidate)
                if abs(diff) > 1e-15:
                    for _ in range(100):
                        h_candidate += diff / N
                        h_candidate = np.clip(h_candidate, 0, 1)
                        new_diff = N / 2 - np.sum(h_candidate)
                        if abs(new_diff) < 1e-12:
                            break
                        diff = new_diff
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polish: use the best solution as starting point for a high-precision run
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=_bounds(),
                constraints=[{'type': 'eq', 'fun': _sum_constraint}],
                options={'maxiter': 5000, 'ftol': 1e-15}
            )
            if res.success and np.isfinite(res.fun):
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, 1)
                diff = N / 2 - np.sum(h_candidate)
                if abs(diff) > 1e-15:
                    for _ in range(100):
                        h_candidate += diff / N
                        h_candidate = np.clip(h_candidate, 0, 1)
                        new_diff = N / 2 - np.sum(h_candidate)
                        if abs(new_diff) < 1e-12:
                            break
                        diff = new_diff
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h
