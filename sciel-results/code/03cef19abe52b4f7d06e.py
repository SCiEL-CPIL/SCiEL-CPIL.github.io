"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_overlaps(h):
    """Compute all circular overlap values."""
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1 - shifted)) * (2.0 / n)
    return overlaps

def _objective(h):
    """Max overlap as objective."""
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project(h):
    """Project to simplex-like constraint: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0, 1)
    # Adjust to make sum exactly N/2
    diff = N / 2.0 - np.sum(h)
    if diff > 0:
        # Need to increase sum; add to elements with room
        mask = h < 1.0
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    elif diff < 0:
        # Need to decrease sum
        mask = h > 0.0
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    # Final correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute remaining difference
        if diff > 0:
            mask = h < 1.0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        else:
            mask = h > 0.0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
    return h

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_score = _objective(best_h)
    
    # Try gradient descent with different initializations
    np.random.seed(42)
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    
    # Try several random initializations
    attempts = [
        ("uniform", np.full(N, 0.5)),
        ("random_binary", np.where(np.random.rand(N) < 0.5, 1.0, 0.0)),
        ("sine", 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / N)),
        ("triangle", 0.5 + 0.5 * np.abs(2 * (np.arange(N) / N - 0.5) - 1)),
    ]
    
    for name, init in attempts:
        if time.time() > deadline - 10:
            break
        
        h_init = _project(init.copy())
        
        # Use L-BFGS-B
        try:
            result = minimize(
                _objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = _project(result.x)
            score = _objective(h_opt)
            if score < best_score - 1e-12:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final projection to ensure exact sum
    best_h = _project(best_h)
    
    # Verify sum is exactly N/2
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-15:
        # Tiny adjustment to one element
        idx = np.argmin(best_h) if diff > 0 else np.argmax(best_h)
        best_h[idx] += diff
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
