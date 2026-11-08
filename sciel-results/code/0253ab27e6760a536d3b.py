"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(a):
    """Compute the maximum absolute value of the autoconvolution of step function a.
    a is an array of ±1 values of length N.
    The autoconvolution at lag k (in interval units) is w * sum_i a[i]*a[i+k]
    where w is the interval width. Since we want to minimize the upper bound,
    and w is a constant factor, we can just minimize max_k |sum_i a[i]*a[i+k]|.
    """
    # Compute all lags from -(N-1) to (N-1)
    # For lag k >= 0: sum_{i=0}^{N-1-k} a[i]*a[i+k]
    # For lag k < 0: by symmetry, same as lag -k
    # We only need to compute k >= 0 and take max
    
    # Use FFT-based cross-correlation for speed
    # Cross-correlation of a with itself
    n = len(a)
    # Pad to next power of 2 for FFT
    size = 1
    while size < 2*n - 1:
        size *= 2
    fa = np.zeros(size, dtype=np.complex128)
    fa[:n] = a.astype(np.float64)
    fa = np.fft.fft(fa)
    fa2 = fa * np.conj(fa)
    corr = np.fft.ifft(fa2).real
    # corr[k] for k=0..n-1 gives sum_{i=0}^{n-1-k} a[i]*a[i+k]
    # The full correlation is symmetric
    max_val = np.max(np.abs(corr[:n]))
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, leaving 10s margin
    
    # Initial random sequence
    np.random.seed(42)
    a = np.random.choice([-1.0, 1.0], size=N)
    
    best_c3 = _compute_c3(a)
    best_a = a.copy()
    
    # Local search with multi-flip moves
    max_iter = 100000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Try flipping 1-3 random bits
        n_flips = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        idxs = np.random.choice(N, size=n_flips, replace=False)
        
        a_new = a.copy()
        a_new[idxs] *= -1
        
        c3_new = _compute_c3(a_new)
        
        if c3_new < best_c3:
            best_c3 = c3_new
            best_a = a_new.copy()
            a = a_new
        else:
            # Revert
            a = best_a.copy() if c3_new >= best_c3 else a_new
            # Actually, let's keep a as the current best
            a = best_a.copy()
    
    return best_a
# EVOLVE-BLOCK-END
