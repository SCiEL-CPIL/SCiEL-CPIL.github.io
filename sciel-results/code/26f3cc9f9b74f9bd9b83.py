"""
Third autocorrelation inequality: minimize the upper bound on C3.
The score is seed/value where value is the raw upper bound (lower is better).
We search over signed step functions with 400 intervals on [-1/4, 1/4].
"""
import numpy as np
import time

N = 400
TIME_LIMIT = 145  # safety margin before 150s kill

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    
    For a step function f(x) = sum_i h_i * 1_{[x_i, x_{i+1}]}(x) on [-1/4, 1/4],
    the autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    
    The C3 bound relates to the L-infinity norm of the autoconvolution.
    Specifically, for the third autocorrelation inequality, the bound is:
        C3 <= max_t |(f*f)(t)|
    
    where f is the signed step function. We compute this via FFT for efficiency.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    # Interval width
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Pad h for linear autoconvolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    # The continuous autoconvolution at grid points is dx * conv[k]
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def _local_search(heights, max_iter=5000, time_budget=30, start_time=0):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    Uses best-improvement strategy with time-based stopping.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        # Try flipping each position
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def _pairwise_swap_search(heights, time_budget=10, start_time=0):
    """
    Pairwise swap local search: try swapping signs at two positions to reduce C3 bound.
    This explores a larger neighborhood than single flips.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    while improved:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        best_gain = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of pairs to keep computation manageable
        # Use random sampling of pairs
        np.random.seed(42)
        indices = np.random.choice(N, size=min(50, N), replace=False)
        
        for ii in range(len(indices)):
            i = indices[ii]
            for jj in range(ii + 1, len(indices)):
                j = indices[jj]
                if time.time() - start_time > time_budget:
                    break
                
                # Swap signs
                h[i], h[j] = -h[i], -h[j]
                new_bound = _compute_c3_bound(h)
                h[i], h[j] = -h[i], -h[j]
                
                gain = current_bound - new_bound
                if gain > best_gain:
                    best_gain = gain
                    best_i, best_j = i, j
        
        if best_i >= 0:
            h[best_i], h[best_j] = -h[best_i], -h[best_j]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(10):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 60)
        best_h, best_bound = _local_search(best_h, max_iter=5000, time_budget=local_time, start_time=start_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 20)
                h, bound = _local_search(h, max_iter=2000, time_budget=local_time, start_time=start_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Pairwise swap search for further refinement
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        swap_time = min(remaining - 5, 20)
        best_h, best_bound = _pairwise_swap_search(best_h, time_budget=swap_time, start_time=start_time)
    
    # Final verification
    final_bound = _compute_c3_bound(best_h)
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
