"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
DELTA = 0.5 / N  # interval width

def compute_autoconv(h):
    """Compute autoconvolution values for lags 0 to N-1."""
    n = len(h)
    g = np.zeros(n)
    for k in range(n):
        g[k] = np.dot(h[:n-k], h[k:])
    return g

def compute_score(h):
    """Compute the normalized maximum of autoconvolution (lower is better)."""
    g = compute_autoconv(h)
    g0 = g[0]
    if g0 <= 0:
        return float('inf')
    # Normalized maximum: max off-peak / peak
    if len(g) > 1:
        return np.max(g[1:]) / g0
    return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial guess: try a few parametric families
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Try triangular
    x = np.linspace(0, 1, N)
    h_tri = 1 - np.abs(2*x - 1)
    s_tri = compute_score(h_tri)
    if s_tri < best_score:
        best_score = s_tri
        best_h = h_tri.copy()
    
    # Try Gaussian-like
    h_gauss = np.exp(-((x - 0.5)**2) / (2 * 0.1**2))
    s_gauss = compute_score(h_gauss)
    if s_gauss < best_score:
        best_score = s_gauss
        best_h = h_gauss.copy()
    
    # Try flat top with smooth edges
    h_flat = np.ones(N)
    edge = max(1, N // 10)
    for i in range(edge):
        h_flat[i] = i / edge
        h_flat[-1-i] = i / edge
    s_flat = compute_score(h_flat)
    if s_flat < best_score:
        best_score = s_flat
        best_h = h_flat.copy()
    
    # Local search: coordinate descent
    h = best_h.copy()
    step_size = 0.1
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        for i in range(0, N, 3):  # Sample every 3rd coordinate for speed
            if time.time() > deadline:
                break
            old_val = h[i]
            for delta in [-step_size, step_size]:
                new_val = max(0, old_val + delta)
                if new_val == old_val:
                    continue
                h[i] = new_val
                new_score = compute_score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
            if not improved and i % 30 == 0:
                step_size *= 0.9
        if step_size < 1e-4:
            break
    
    # Final polish: finer steps
    step_size = 0.01
    for iteration in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = h[i]
            for delta in [-step_size, step_size]:
                new_val = max(0, old_val + delta)
                if new_val == old_val:
                    continue
                h[i] = new_val
                new_score = compute_score(h)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
            if not improved and i % 50 == 0:
                step_size *= 0.9
        if step_size < 1e-5:
            break
    
    # Ensure nonnegative
    best_h = np.maximum(0, best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
