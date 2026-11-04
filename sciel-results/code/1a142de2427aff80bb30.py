"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
W = 1.0 / (2 * N)  # width of each interval: total length 1/2, so each is 1/(2N) = 1/800

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_peak(h):
    """Compute the L-infinity norm of the autoconvolution of step function h.
    h is an array of N signed heights. Each step has width W.
    Returns max over lags of |autoconvolution(lag)|.
    """
    # Discrete autocorrelation: R(k) = W^2 * sum_i h[i] * h[i-k] for valid i
    # For k >= 0: sum from i=k to N-1 of h[i]*h[i-k]
    # R(-k) = R(k)
    w2 = W * W
    max_val = 0.0
    for k in range(N):
        # R(k) = w2 * dot(h[k:], h[:-k]) if k > 0, else w2 * sum(h^2)
        if k == 0:
            r = w2 * np.dot(h, h)
        else:
            r = w2 * np.dot(h[k:], h[:-k])
        val = abs(r)
        if val > max_val:
            max_val = val
    return max_val


def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Start with alternating pattern (good initial structure)
    h = np.ones(N)
    h[1::2] = -1.0
    
    best_h = h.copy()
    best_val = _autoconv_peak(best_h)
    
    # Local search: flip individual bits
    while time.time() < deadline:
        # Try flipping a random bit
        i = rng.integers(0, N)
        h[i] = -h[i]
        new_val = _autoconv_peak(h)
        
        if new_val < best_val:
            best_val = new_val
            best_h = h.copy()
        else:
            h[i] = -h[i]  # revert
    
    return best_h
# EVOLVE-BLOCK-END
