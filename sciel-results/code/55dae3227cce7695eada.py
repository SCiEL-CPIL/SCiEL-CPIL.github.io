"""Spectral optimization approach for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Minimizes the maximum overlap of h with shifted 1-h.
    
    Model: h_i = 0.5 + A * cos(omega * i + phase)
    Overlap M(k) = sum_i h_i * (1 - h_{i+k})
                 = N/2 - sum_i h_i * h_{i+k}
                 = N/2 - R(k)
    
    Autocorrelation R(k) for this model:
    R(k) = N/4 + (N * A^2 / 2) * cos(omega * k)
    
    We want to minimize max_k M(k).
    max_k M(k) = N/2 - min_k R(k)
    
    min_k R(k) occurs when cos(omega * k) is -1.
    min_k R(k) = N/4 - (N * A^2 / 2)
    
    Therefore, max_k M(k) = N/2 - (N/4 - N * A^2 / 2)
                         = N/4 + N * A^2 / 2
    
    To minimize this, we must minimize A^2.
    The constraint h_i in [0, 1] implies 0.5 - A >= 0 and 0.5 + A <= 1, so A <= 0.5.
    The minimum possible A is 0.
    
    With A = 0, h_i = 0.5 for all i.
    max_k M(k) = N/4.
    """
    
    # Analytical optimum for A is 0
    A = 0.0
    
    # Any omega is valid if A=0, but we pick a standard value for consistency
    omega = 2.0 * np.pi / N
    phase = 0.0
    
    i = np.arange(N)
    h = 0.5 + A * np.cos(omega * i + phase)
    
    # Ensure exact sum N/2 and bounds [0, 1]
    # With A=0, h is exactly 0.5 everywhere, sum is N/2.
    # No adjustment needed, but we include safety checks.
    
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    return h
# EVOLVE-BLOCK-END
