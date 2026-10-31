import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    # Construct geometric progression with ratio 2
    # h_i = 2^(i)
    # To avoid overflow/underflow, normalize such that max is 1
    # h_i = 2^(i - (N-1))
    
    h = np.power(2.0, np.arange(N) - (N - 1))
    
    # Verify score
    score = compute_ratio(h)
    
    # If time allows, try to optimize the ratio r slightly
    # But r=2 seems theoretically optimal for this structure.
    # Let's just return this.
    
    return list(h)
