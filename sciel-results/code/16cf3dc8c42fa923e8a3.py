"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, dual_annealing
import time

N = 100

# EVOLVE-BLOCK-START

def _objective(h):
    """Calculate the max correlation value."""
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]
    
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        # Enforce sum constraint strictly
        diff = N / 2.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate = h_candidate + diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
        val = _objective(h_candidate)
        if val < best_val:
            best_h = h_candidate.copy()
            best_val = val

    # 1. Global Search using Dual Annealing
    # dual_annealing supports constraints and bounds
    # We run it a few times or for a long time if possible
    
    def run_dual_annealing(seed):
        if time.time() > deadline:
            return
        try:
            res = dual_annealing(
                _objective,
                bounds=bounds,
                constraints=constraints,
                seed=seed,
                maxiter=50, # Limit iterations to ensure it finishes
                restarts=10
            )
            if res.success:
                try_update(res.x)
        except Exception:
            pass

    # Run global search
    for s in range(5):
        if time.time() > deadline - 20: # Leave time for polishing
            break
        run_dual_annealing(s)

    # 2. Local Polishing with SLSQP
    # SLSQP is good for refining the solution found by DA
    
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            try_update(res.x)
        except Exception:
            pass

    # 3. Additional Refinement
    # Sometimes SLSQP gets stuck. Try perturbing and re-optimizing
    if time.time() < deadline - 2:
        np.random.seed(42)
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            h_perturbed = best_h + np.random.normal(0, 0.05, N)
            h_perturbed = np.clip(h_perturbed, 0, 1)
            try:
                res = minimize(
                    _objective,
                    h_perturbed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                try_update(res.x)
            except Exception:
                pass

    # Final check and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        
    return best_h

# EVOLVE-BLOCK-END
