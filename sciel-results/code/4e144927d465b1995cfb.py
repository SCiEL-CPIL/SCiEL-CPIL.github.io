"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _hump_init():
    """Create a hump-shaped initial guess."""
    x = np.linspace(-1, 1, N)
    # Triangular hump
    h = np.maximum(0, 1 - np.abs(x))
    # Normalize so max is 1
    h = h / np.max(h)
    return h

def _local_search(h0, deadline):
    """Simple coordinate descent + random restarts local search."""
    h = h0.copy()
    best_val = _compute_value(h)
    best_h = h.copy()
    
    # Coordinate descent with adaptive step size
    step = 0.1
    for i in range(N):
        if time.time() > deadline:
            break
        # Try increasing
        trial = h.copy()
        trial[i] = max(0, h[i] + step)
        val = _compute_value(trial)
        if val > best_val:
            h = trial
            best_val = val
            best_h = h.copy()
        # Try decreasing
        trial = h.copy()
        trial[i] = max(0, h[i] - step)
        val = _compute_value(trial)
        if val > best_val:
            h = trial
            best_val = val
            best_h = h.copy()
    
    # Multiple rounds of coordinate descent with shrinking step
    for round_num in range(20):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for sign in [1, -1]:
                trial = h.copy()
                trial[i] = max(0, h[i] + sign * step)
                val = _compute_value(trial)
                if val > best_val:
                    h = trial
                    best_val = val
                    best_h = h.copy()
                    improved = True
        if not improved:
            step *= 0.5
        else:
            step *= 0.8
    
    return best_h, best_val

def _random_restart(h0, deadline, max_restarts=5):
    """Try a few random perturbations from h0."""
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        # Random perturbation
        h = h0.copy()
        noise = np.random.uniform(-0.2, 0.2, N)
        h = np.clip(h + noise, 0, None)
        h_val, h_score = _local_search(h, deadline)
        if h_score > best_val:
            best_h = h_val
            best_val = h_score
    
    return best_h, best_val

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    # Start with hump
    h0 = _hump_init()
    h_best, val_best = _local_search(h0, deadline)
    
    # Try random restarts
    h_best, val_best = _random_restart(h_best, deadline, max_restarts=3)
    
    return list(h_best)
# EVOLVE-BLOCK-END
