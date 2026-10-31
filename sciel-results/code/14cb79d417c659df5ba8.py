"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Compute maximum absolute value of linear autoconvolution."""
    # Linear autoconvolution via FFT
    n = len(x)
    size = 2 * n
    fx = np.fft.rfft(x, size)
    ac = np.fft.irfft(fx * np.conj(fx), size)
    # Linear autoconvolution is ac[0:n] (the first n elements)
    return np.max(np.abs(ac[:n]))


def _score(x):
    """Compute max|autoconv| / integral^2."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2."""
    t0 = time.time()
    deadline = t0 + 145.0
    
    # Start with all +1 (baseline: score = 1/N)
    best = np.ones(N, dtype=np.float64)
    best_score = _score(best)
    
    # Try random signed sequences to find better starting points
    rng = np.random.default_rng(42)
    
    # Try several random initializations
    for init_iter in range(200):
        if time.time() > deadline:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x)
        if s == 0:
            continue
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best = x.copy()
    
    # Local search: flip individual signs to improve
    for flip_iter in range(50000):
        if time.time() > deadline:
            break
        
        # Try flipping a random position
        idx = rng.integers(0, N)
        x = best.copy()
        x[idx] = -x[idx]
        s = np.sum(x)
        if s == 0:
            continue
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best = x
    
    # Multi-flip local search: flip pairs of positions
    for pair_iter in range(30000):
        if time.time() > deadline:
            break
        
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        x = best.copy()
        x[i] = -x[i]
        x[j] = -x[j]
        s = np.sum(x)
        if s == 0:
            continue
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best = x
    
    return best
# EVOLVE-BLOCK-END
