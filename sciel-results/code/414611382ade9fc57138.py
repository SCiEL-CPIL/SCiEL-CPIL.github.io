"""
Third autocorrelation inequality: minimize the upper bound on C3.

The score is the raw value (lower is better). The interface returns a sequence of
signed step heights (400 values). We use a combination of:
1. Thue-Morse initialization for low-complexity flat autocorrelation
2. Greedy coordinate descent with periodic full re-evaluation
3. Deadline guard to ensure we finish within time limits

The objective: for a signed step function f with N intervals, the autoconvolution
upper bound C3 is related to the maximum of the autocorrelation. We want to
minimize the maximum absolute autocorrelation value (excluding the trivial
self-correlation at lag 0).

Key insight from the knowledge: Thue-Morse gives good initial flat autocorrelation,
and greedy refinement can significantly improve the score (0.007 -> 0.35 improvement
was observed in another island, meaning the raw value dropped substantially).
"""

import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n: t(i) = parity of popcount(i)."""
    return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.float64)

def _autocorr_max(x):
    """
    Compute the maximum absolute value of the autocorrelation of x,
    excluding lag 0.
    
    For a step function with heights x[0..N-1], the autocorrelation at lag k
    is sum_{i} x[i] * x[i+k] (for the discrete version).
    
    We use FFT-based computation for efficiency.
    """
    # Pad to avoid circular convolution issues
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    
    X = np.fft.rfft(x, size)
    # Autocorrelation via power spectrum
    ac = np.fft.irfft(X * np.conj(X), size)[:n]
    
    # ac[0] is the self-correlation (sum of squares), skip it
    # We want max |ac[k]| for k >= 1
    if n <= 1:
        return 0.0
    
    return np.max(np.abs(ac[1:]))

def _try_flip(x, idx, current_max):
    """
    Try flipping the sign of x[idx]. Return the new max autocorrelation if it improves,
    or None if it doesn't.
    """
    old_val = x[idx]
    x[idx] = -old_val
    new_max = _autocorr_max(x)
    if new_max < current_max - 1e-12:
        return new_max
    else:
        x[idx] = old_val
        return None

def construct():
    """
    Construct a signed step function with 400 equal intervals that minimizes
    the autoconvolution upper bound C3.
    
    Strategy:
    1. Initialize with Thue-Morse sequence (±1 values)
    2. Greedy coordinate descent: repeatedly try flipping each coordinate
       and accept if it reduces the max autocorrelation
    3. Use deadline guard to stop before timeout
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin (limit is 120s, kill at 150s)
    
    # Initialize with Thue-Morse: values are +1 or -1
    x = _thue_morse(N) * 2.0 - 1.0  # Convert 0/1 to -1/+1
    
    # Compute initial max autocorrelation
    current_max = _autocorr_max(x)
    best_x = x.copy()
    best_max = current_max
    
    # Greedy coordinate descent
    improved = True
    pass_count = 0
    
    while improved:
        improved = False
        pass_count += 1
        
        # Check time
        if time.time() > deadline - 5.0:
            break
        
        # Randomize the order of coordinates to avoid cycling
        order = np.random.permutation(N)
        
        for idx in order:
            if time.time() > deadline - 5.0:
                break
            
            new_max = _try_flip(x, idx, current_max)
            if new_max is not None:
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_x = x.copy()
        
        # Early exit if max is essentially zero
        if current_max < 1e-9:
            break
    
    # If we used most of the time on greedy, we might want to return best_x
    # But x should be at least as good as best_x since we only accept improvements
    # Actually, x IS the current state which is >= best in quality (lower is better)
    # So x should be the best found. But let's be safe and return the one with lower max.
    
    if best_max <= current_max:
        return best_x
    else:
        return x

# For testing
if __name__ == "__main__":
    x = construct()
    print(f"Length: {len(x)}")
    print(f"Unique values: {np.unique(x)}")
    print(f"Max autocorrelation (excl lag 0): {_autocorr_max(x)}")
