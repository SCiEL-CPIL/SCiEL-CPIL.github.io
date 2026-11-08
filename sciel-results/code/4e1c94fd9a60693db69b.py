import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_value(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15) or not np.all(np.isfinite(h)):
        return -np.inf
    h = np.clip(h, 0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    yi = y[:m+1]
    yi1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yi1 + yi1**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 < 1e-15 or norm_inf < 1e-15:
        return -np.inf
    return l2 / (norm_1 * norm_inf)

def negative_value(h):
    v = compute_value(h)
    if v == -np.inf:
        return 1e10
    return -v

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # Uniform
    initial_guesses.append(np.ones(N))
    
    # Gaussian-like with various widths
    x = np.linspace(-1, 1, N)
    for sigma in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        initial_guesses.append(np.exp(-x**2 / (2 * sigma**2)))
    
    # Plateau with various widths
    for width in [5, 10, 15, 20, 25, 30, 40, 49]:
        h = np.zeros(N)
        center = N // 2
        half = width // 2
        h[max(0, center-half):min(N, center+half)] = 1.0
        initial_guesses.append(h)
    
    # Triangular
    h = np.abs(np.linspace(-1, 1, N))
    initial_guesses.append(h)
    
    # Cosine bump
    h = np.cos(np.linspace(-np.pi/2, np.pi/2, N))
    h = np.clip(h, 0, None)
    initial_guesses.append(h)
    
    # Ramp
    initial_guesses.append(np.linspace(0, 1, N))
    initial_guesses.append(np.linspace(1, 0, N))
    
    # Two peaks
    h = np.zeros(N)
    h[N//4] = 1.0
    h[3*N//4] = 1.0
    initial_guesses.append(h)
    
    # Random
    for _ in range(20):
        initial_guesses.append(rng.random(N))
    
    # Normalize and evaluate
    scored = []
    for h in initial_guesses:
        h = np.asarray(h, dtype=float)
        s = np.sum(h)
        if s > 1e-15:
            h = h / s
        val = compute_value(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
        scored.append((val, h))
    
    # Sort by value
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Optimize from top candidates
    for val, h0 in scored:
        if time.time() - start_time > deadline:
            break
        
        h0 = h0.copy()
        if np.max(h0) > 1e-15:
            h0 = h0 / np.max(h0)
        
        try:
            res = minimize(
                negative_value,
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'maxfun': 20000}
            )
            h_opt = np.clip(res.x, 0, None)
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # More random restarts if time permits
    n_extra = 0
    while time.time() - start_time < deadline:
        h0 = rng.random(N)
        s = np.sum(h0)
        if s > 1e-15:
            h0 = h0 / s
        
        try:
            res = minimize(
                negative_value,
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'maxfun': 10000}
            )
            h_opt = np.clip(res.x, 0, None)
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
            n_extra += 1
        except Exception:
            continue
    
    return list(best_h)
# EVOLVE-BLOCK-END
