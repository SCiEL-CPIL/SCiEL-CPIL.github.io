"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _overlap_max(h, N):
    """Compute max over shifts of sum_i h[i] * (1 - h[(i+t) % N]) / N."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    # ifft(conj(H) * C)[t] = sum_j h[j] * comp[j+t]
    overlap = np.real(np.fft.ifft(np.conj(H) * C))
    return np.max(overlap)


def construct():
    N = 100
    
    # The constant sequence h = 0.5 everywhere is optimal:
    # overlap[t] = sum_i 0.5 * (1 - 0.5) = sum_i 0.25 = N * 0.25
    # For N=100, this gives 25.0 for every shift t, so max overlap = 25.0.
    # This is the theoretical minimum since by symmetry and convexity,
    # any deviation from constant increases the maximum overlap.
    
    h = np.full(N, 0.5)
    
    return h
# EVOLVE-BLOCK-END
