"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_max_abs(x):
    """Compute max absolute value of autocorrelation (overlap definition)."""
    n = len(x)
    # Use FFT for efficiency
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
    return np.max(np.abs(acf))


def _score_seq(x):
    """Compute the ratio: max|C_x| / (sum x)^2. Returns inf if sum is 0."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    return _autocorr_max_abs(x) / (s * s)


def construct():
    deadline = time.time() + 120.0
    x = np.ones(N, dtype=np.float64)
    best_score = _score_seq(x)
    best_x = x.copy()
    
    # Local search: try flipping individual elements
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            x[i] *= -1
            s = _score_seq(x)
            if s < best_score:
                best_score = s
                best_x = x.copy()
                improved = True
            else:
                x[i] *= -1  # revert
    
    return best_x
# EVOLVE-BLOCK-END
