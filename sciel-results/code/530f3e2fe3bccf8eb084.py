"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_peak(x):
    """Compute the max absolute value of circular autoconvolution."""
    # Use FFT for efficient circular convolution
    n = len(x)
    X = np.fft.fft(x)
    autoconv = np.fft.ifft(X * np.conj(X)).real
    return np.max(np.abs(autoconv))


def _greedy_local_search(x, max_iter=5000, timeout=None, start_time=None):
    """Greedy local search: try flipping each position, keep improvements."""
    if start_time is None:
        start_time = time.time()
    if timeout is None:
        timeout = 150.0
    
    best = _autoconv_peak(x)
    improved = True
    iters = 0
    
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if time.time() - start_time > timeout - 5:
            break
        
        for i in range(N):
            # Try flipping sign at position i
            old_val = x[i]
            x[i] = -old_val
            new_val = _autoconv_peak(x)
            if new_val < best:
                best = new_val
                improved = True
            else:
                x[i] = old_val
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_x = None
    best_score = float('inf')
    
    # Try different sparse patterns with varying k
    # Start with small k and increase
    for k in [4, 8, 12, 16, 20, 24, 32, 40]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            # Random sparse pattern
            x = np.zeros(N)
            positions = np.random.choice(N, size=k, replace=False)
            x[positions] = np.random.choice([-1, 1], size=k)
            
            # Greedy local search
            score = _greedy_local_search(x, max_iter=2000, timeout=deadline, start_time=start_time)
            
            if score < best_score:
                best_score = score
                best_x = x.copy()
    
    if best_x is not None:
        return best_x
    else:
        return np.ones(N)
# EVOLVE-BLOCK-END
