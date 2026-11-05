"""
Optimized program for the Third Autocorrelation Inequality task.

The objective is to minimize V = |2 * N * max(conv(h,h)) / sum(h)^2|.
The seed (constant vector) gives V=2.0. We can do better by finding
vectors where the maximum of the autoconvolution is significantly less
than the lag-0 value, while keeping the sum large.

Strategy: Use a chirp-like complex exponential sequence h[i] = Re(exp(i*(phi + beta*i^2)))
or similar structured sequences, and optimize parameters. Also try random
search with differential evolution over structured families.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START

def _compute_value(h):
    """Compute the evaluator's exact value."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _chirp_vector(phi, beta, alpha=1.0):
    """Generate a chirp-like vector: h[i] = alpha * cos(phi + beta * i^2)."""
    i = np.arange(N, dtype=np.float64)
    return alpha * np.cos(phi + beta * i * i)

def _obj_chirp(params):
    """Objective for chirp parameters."""
    phi, beta, alpha = params
    h = _chirp_vector(phi, beta, alpha)
    return _compute_value(h)

def _two_level_vector(bits, a=1.0, b=-1.0):
    """Generate a two-level vector from a bit pattern."""
    h = np.where(bits == 1, a, b)
    return h

def _obj_two_level(params):
    """Objective for two-level vector with a, b and a seed for random bits."""
    a, b = params[0], params[1]
    seed = int(params[2]) % (2**32)
    rng = np.random.RandomState(seed)
    bits = rng.randint(0, 2, N)
    h = np.where(bits == 1, a, b)
    return _compute_value(h)

def _random_search_best(n_trials=2000, timeout_frac=0.3):
    """Do random search over chirp parameters and return best h."""
    deadline = START_TIME + timeout_frac * TIME_LIMIT
    best_val = np.inf
    best_h = np.ones(N)
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        # Random chirp parameters
        phi = np.random.uniform(0, 2 * np.pi)
        beta = np.random.uniform(-0.05, 0.05)
        alpha = np.random.uniform(0.5, 2.0)
        h = _chirp_vector(phi, beta, alpha)
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val

def _de_chirp(timeout_frac=0.4):
    """Run differential evolution on chirp parameters."""
    deadline = START_TIME + timeout_frac * TIME_LIMIT
    if time.time() > deadline:
        return np.ones(N), np.inf
    
    bounds = [(0, 2 * np.pi), (-0.1, 0.1), (0.1, 3.0)]
    try:
        result = differential_evolution(
            _obj_chirp, bounds, maxiter=50, popsize=15,
            tol=1e-6, seed=42, polish=True
        )
        if time.time() > deadline:
            return np.ones(N), np.inf
        if result.fun < np.inf:
            h = _chirp_vector(*result.x)
            val = _compute_value(h)
            return h, val
    except Exception:
        pass
    return np.ones(N), np.inf

def _nelder_mead_polish(h0, timeout_frac=0.2):
    """Polish a vector using Nelder-Mead on a low-dimensional parametrization."""
    deadline = START_TIME + timeout_frac * TIME_LIMIT
    if time.time() > deadline:
        return h0, _compute_value(h0)
    
    # Use chirp parametrization for polishing
    # Fit approximate phi, beta by least squares on phase
    i = np.arange(N, dtype=np.float64)
    # Try to extract effective parameters
    # Just do a local search over small perturbations
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    # Simple local perturbation search
    for _ in range(50):
        if time.time() > deadline:
            break
        perturbation = np.random.normal(0, 0.01, N)
        h_new = best_h + perturbation
        val = _compute_value(h_new)
        if val < best_val:
            best_val = val
            best_h = h_new
    
    return best_h, best_val

def construct():
    """
    Returns the optimal step heights.
    Uses a combination of random search, differential evolution, and local polishing
    to find a vector that minimizes the autocorrelation inequality value.
    """
    deadline = START_TIME + TIME_LIMIT - 10.0  # Safety margin
    
    # Start with seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Random search over chirp family
    if time.time() < deadline - 60:
        h_rand, val_rand = _random_search_best(n_trials=3000, timeout_frac=0.25)
        if val_rand < best_val:
            best_h = h_rand
            best_val = val_rand
    
    # Phase 2: Differential evolution on chirp parameters
    if time.time() < deadline - 40:
        h_de, val_de = _de_chirp(timeout_frac=0.35)
        if val_de < best_val:
            best_h = h_de
            best_val = val_de
    
    # Phase 3: Local polishing
    if time.time() < deadline - 15:
        h_polished, val_polished = _nelder_mead_polish(best_h, timeout_frac=0.15)
        if val_polished < best_val:
            best_h = h_polished
            best_val = val_polished
    
    # Final safety check
    if time.time() > deadline - 5:
        return np.ones(N)
    
    final_val = _compute_value(best_h)
    if final_val < 2.0:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
