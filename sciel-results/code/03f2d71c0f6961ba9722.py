"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _objective(x):
    """Objective function for optimization. x is the time-domain sequence."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    Hp = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(Hp * Hp, fsize)
    return np.max(np.abs(ac)) / (s * s)

def _local_search(h, deadline):
    """Local search with random perturbations."""
    best_h = h.copy()
    best_score = _score(best_h)
    rng = np.random.default_rng(42)
    
    step_size = 0.1
    while time.time() < deadline:
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            delta = rng.normal(0, step_size)
            new_h = best_h.copy()
            new_h[idx] += delta
            new_score = _score(new_h)
            if new_score < best_score:
                best_h = new_h
                best_score = new_score
                improved = True
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    # Start with flat sequence
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Try several initializations and local search
    initializations = [
        np.ones(N) / N,  # flat
        np.sin(np.pi * np.arange(N) / N) / N,  # sine
        np.cos(np.pi * np.arange(N) / N) / N,  # cosine
        np.hanning(N) / N,  # Hann window
    ]
    
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        s = np.sum(h_init)
        if abs(s) < 1e-9:
            continue
        h_init = h_init / s  # normalize so sum = 1
        h_local, score_local = _local_search(h_init, deadline - 5)
        if score_local < best_score:
            best_score = score_local
            best_h = h_local.copy()
    
    return best_h

# EVOLVE-BLOCK-END
