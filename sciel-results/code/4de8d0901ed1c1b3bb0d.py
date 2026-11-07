"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(x):
    """x is a vector of 100 heights in [0,1]. Minimize max correlation / N * 2."""
    h = x
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _constraint_sum(x):
    return np.sum(x) - N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial guess: 0.5 everywhere
    x0 = np.full(N, 0.5)
    
    # Bounds: each h[i] in [0, 1]
    bounds = [(0.0, 1.0)] * N
    
    # Constraints: sum must equal N/2
    constraints = [{'type': 'eq', 'fun': _constraint_sum}]
    
    best_h = x0.copy()
    best_val = _objective(best_h)
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if r == 0:
            x_init = x0
        else:
            # Random perturbation around 0.5, then project to satisfy sum constraint
            x_init = np.random.uniform(0.2, 0.8, size=N)
            # Adjust to satisfy sum constraint
            diff = N / 2.0 - np.sum(x_init)
            x_init += diff / N
            # Clip to [0,1] and re-adjust
            x_init = np.clip(x_init, 0.0, 1.0)
            # Iteratively fix sum
            for _ in range(10):
                diff = N / 2.0 - np.sum(x_init)
                if abs(diff) < 1e-9:
                    break
                # Add to elements that have room
                mask = (x_init < 1.0) if diff > 0 else (x_init > 0.0)
                if not np.any(mask):
                    break
                x_init[mask] += diff / np.sum(mask)
                x_init = np.clip(x_init, 0.0, 1.0)
        
        try:
            result = minimize(
                _objective, x_init, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if result.success or result.fun < best_val:
                h_candidate = result.x
                # Ensure constraints are satisfied
                if np.all(h_candidate >= -1e-9) and np.all(h_candidate <= 1.0 + 1e-9):
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                    # Fix sum precisely
                    diff = N / 2.0 - np.sum(h_candidate)
                    if abs(diff) > 1e-9:
                        # Small adjustment
                        h_candidate += diff / N
                        h_candidate = np.clip(h_candidate, 0.0, 1.0)
                        # Re-fix
                        for _ in range(5):
                            diff = N / 2.0 - np.sum(h_candidate)
                            if abs(diff) < 1e-9:
                                break
                            mask = (h_candidate < 1.0) if diff > 0 else (h_candidate > 0.0)
                            if not np.any(mask):
                                break
                            h_candidate[mask] += diff / np.sum(mask)
                            h_candidate = np.clip(h_candidate, 0.0, 1.0)
                    
                    val = _objective(h_candidate)
                    if val < best_val:
                        best_val = val
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-9:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
