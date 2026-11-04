import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400

def _compute_value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def _parametric_objective(params):
    phi0, alpha, beta, eps, k, phi1 = params
    n = np.arange(N)
    # Chirp component
    h_chirp = np.cos(phi0 + 0.5 * alpha * n**2 + beta * n)
    # Low freq correction
    h_low = eps * np.cos(2 * np.pi * k * n / N + phi1)
    h = h_chirp + h_low
    return _compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    # 1. Parametric Global Search
    # Bounds:
    # phi0: [0, 2pi]
    # alpha: [-10, 10] (chirp rate)
    # beta: [-10, 10] (linear freq)
    # eps: [0, 2] (amplitude of low freq)
    # k: [1, 10] (integer-ish, but DE handles continuous, we can round or let it be continuous)
    # phi1: [0, 2pi]
    
    bounds = [
        (0, 2*np.pi),
        (-10, 10),
        (-10, 10),
        (0, 2),
        (1, 10),
        (0, 2*np.pi)
    ]

    # Run Differential Evolution
    # We want it to be fast. maxiter=50, popsize=10
    try:
        if time.time() < deadline - 30:
            de_res = differential_evolution(
                _parametric_objective, 
                bounds, 
                maxiter=50, 
                popsize=10,
                seed=42,
                tol=1e-5,
                polish=False # We will polish manually
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                # Reconstruct h from best params
                phi0, alpha, beta, eps, k, phi1 = de_res.x
                n = np.arange(N)
                best_h = np.cos(phi0 + 0.5 * alpha * n**2 + beta * n) + eps * np.cos(2 * np.pi * k * n / N + phi1)
    except Exception:
        pass

    # 2. Local Refinement on the full vector
    # Use Nelder-Mead on a subset or a simple coordinate descent?
    # N=400 is small enough for Nelder-Mead if we are careful, but it might be slow.
    # Let's use a custom local search that perturbs random subsets.
    
    def _local_search(h, max_iters=1000, deadline=None):
        if deadline is None:
            deadline = time.time() + 5.0
        best_h = h.copy()
        best_val = _compute_value(best_h)
        rng = np.random.default_rng(42)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            # Perturb a random block of indices to allow structural changes
            block_size = rng.integers(1, 10)
            start_idx = rng.integers(0, N - block_size + 1)
            idxs = np.arange(start_idx, start_idx + block_size)
            
            trial = best_h.copy()
            # Add small random noise to the block
            noise = rng.normal(0, 0.1, block_size)
            trial[idxs] += noise
            
            val = _compute_value(trial)
            if val < best_val:
                best_val = val
                best_h = trial
        return best_h, best_val

    # Refine the best found so far
    if time.time() < deadline - 10:
        best_h, best_val = _local_search(best_h, max_iters=500, deadline=deadline - 10)

    # 3. Try a few other random parametric starts
    for seed in [1, 2, 3]:
        if time.time() > deadline - 20:
            break
        rng = np.random.default_rng(seed)
        params = [
            rng.uniform(0, 2*np.pi),
            rng.uniform(-10, 10),
            rng.uniform(-10, 10),
            rng.uniform(0, 2),
            rng.uniform(1, 10),
            rng.uniform(0, 2*np.pi)
        ]
        phi0, alpha, beta, eps, k, phi1 = params
        n = np.arange(N)
        h_candidate = np.cos(phi0 + 0.5 * alpha * n**2 + beta * n) + eps * np.cos(2 * np.pi * k * n / N + phi1)
        val_candidate = _compute_value(h_candidate)
        
        if val_candidate < best_val:
            best_val = val_candidate
            best_h = h_candidate
            # Refine this new best
            if time.time() < deadline - 10:
                best_h, best_val = _local_search(best_h, max_iters=200, deadline=deadline - 10)

    # Final check
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h
