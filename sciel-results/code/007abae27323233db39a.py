import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses a 1D search over the amplitude of a high-frequency square wave perturbation
    added to a constant baseline. This follows the strategy to reduce dimensionality.
    """
    deadline = time.time() + 110.0  # Safety margin
    
    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0
    
    # Define a square wave pattern s
    # s[i] = 1 if i % 2 == 0 else -1
    s = np.ones(N)
    s[1::2] = -1.0
    
    # Scan amplitude A
    # h = 1 + A * s
    # Sum(h) = N (since sum(s)=0)
    # We scan a range of A to find the minimum value.
    # Theoretical minimum is at A=0 (value 2.0).
    
    A_range = np.linspace(-0.5, 0.5, 100)
    
    for A in A_range:
        h = 1.0 + A * s
        h_sum = np.sum(h)
        if h_sum == 0 or not np.all(np.isfinite(h)):
            continue
        
        # Compute conv
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2 * N * max_conv / (h_sum ** 2))
        
        if val < best_value:
            best_value = val
            best_h = h.copy()
            
    return best_h
