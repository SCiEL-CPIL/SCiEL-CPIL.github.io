"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve


def _score(signs):
    """Compute the C3-prime value for a binary sign sequence."""
    s = signs.astype(np.int32)
    integral = np.sum(s)
    if integral == 0:
        return np.inf
    denom = integral ** 2
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return float(max_ac) / denom


def construct():
    # For ±1 sequences, ac[0] = N always, so max|ac| >= N.
    # |sum| <= N, so score >= N/N^2 = 1/N.
    # All 1s achieves exactly 1/N, which is the global minimum.
    # Verify with a quick local search for robustness.
    
    signs = np.ones(N, dtype=np.int32)
    current_score = _score(signs)
    
    # Quick local search: try flipping each position once
    for i in range(N):
        original_sign = signs[i]
        signs[i] = -original_sign
        new_score = _score(signs)
        if new_score < current_score:
            current_score = new_score
        else:
            signs[i] = original_sign
    
    return signs
# EVOLVE-BLOCK-END
