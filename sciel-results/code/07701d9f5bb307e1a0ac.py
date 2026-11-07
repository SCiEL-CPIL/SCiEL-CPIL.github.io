"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(h):
    """Compute the value to minimize."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _constraints_check(h):
    """Check if h is valid."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    # Try smooth sinusoidal patterns
    for freq in range(1, 20):
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            h = 0.5 + 0.3 * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            h = np.clip(h, 0, 1)
            # Adjust to satisfy sum constraint
            diff = N/2 - np.sum(h)
            h = h + diff / N
            h = np.clip(h, 0, 1)
            if _constraints_check(h):
                val = _objective(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
    
    # Try local optimization from best found
    h0 = best_h.copy()
    
    def obj_func(x):
        h = np.clip(x, 0, 1)
        # Enforce sum constraint via penalty
        sum_penalty = (np.sum(h) - N/2) ** 2 * 1e6
        return _objective(h) + sum_penalty
    
    # Use L-BFGS-B with bounds
    bounds = [(0, 1)] * N
    result = minimize(obj_func, h0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 500, 'ftol': 1e-12})
    
    if _constraints_check(result.x):
        val = _objective(result.x)
        if val < best_val:
            best_val = val
            best_h = result.x.copy()
    
    # Try more local optimizations from different starting points
    if time.time() < deadline:
        rng = np.random.RandomState(42)
        for _ in range(20):
            if time.time() > deadline:
                break
            h0 = 0.5 + 0.3 * np.sin(2*np.pi*rng.randint(1,20)*np.arange(N)/N + rng.uniform(0, 2*np.pi))
            h0 = np.clip(h0, 0, 1)
            diff = N/2 - np.sum(h0)
            h0 = h0 + diff / N
            h0 = np.clip(h0, 0, 1)
            if not _constraints_check(h0):
                continue
            
            result = minimize(obj_func, h0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-12})
            if _constraints_check(result.x):
                val = _objective(result.x)
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
    
    # Final projection to ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Distribute the difference
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    # Verify
    if not _constraints_check(best_h):
        return np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
