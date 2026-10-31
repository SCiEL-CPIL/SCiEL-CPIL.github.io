"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)


def _autoconv_linear(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 2 * n - 1
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    C = np.fft.irfft(X * X.conj(), n=fft_size)
    return C[:size]


def _score(x):
    """Compute max|autoconv| / (sum x)^2. Return inf if sum is zero."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv_linear(x)
    return np.max(np.abs(c)) / (s * s)


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2."""
    # Start with constant vector (known feasible solution)
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try a few structured initializations
    # 1. Alternating blocks
    for block_size in [2, 4, 8, 16, 32]:
        x = np.ones(N)
        for i in range(N):
            if (i // block_size) % 2 == 1:
                x[i] = -1
        s = np.sum(x)
        if abs(s) > 1e-12:
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
    
    # 2. Random vectors with small nonzero sum
    rng = np.random.RandomState(42)
    for _ in range(50):
        x = rng.choice([-1, 1], size=N)
        s = np.sum(x)
        if abs(s) > 1e-12:
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
    
    # Local search: flip individual elements
    x = best_x.copy()
    improved = True
    step = 0
    while improved and _time_remaining() > 5.0:
        improved = False
        # Try flipping each element
        for i in range(N):
            if _time_remaining() < 2.0:
                break
            old_val = x[i]
            x[i] = -old_val
            sc = _score(x)
            if sc < best_score:
                best_score = sc
                best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
        step += 1
    
    return best_x
# EVOLVE-BLOCK-END
