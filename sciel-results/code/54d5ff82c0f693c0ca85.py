"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # safety margin before 150s kill


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    For a step function f(x) with N equal intervals on [-1/4, 1/4],
    the autoconvolution (f*f)(t) is computed via FFT-based convolution.
    The bound is dx * max_k |conv[k]| where conv is the discrete autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
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


def _fast_flip_eval(h, i, dx):
    """
    Efficiently compute the new C3 bound after flipping sign at index i.
    Uses incremental update of autoconvolution via FFT.
    """
    n = len(h)
    h_new = h.copy()
    h_new[i] = -h_new[i]
    
    # Quick recompute - for N=400 this is fast enough
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h_new
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    max_abs = np.max(np.abs(conv))
    return dx * max_abs


def _local_search(heights, max_iter=2000, time_budget=10, use_pairwise=True):
    """
    Local search: try flipping individual bits and pairwise swaps to reduce the C3 bound.
    Uses best-improvement strategy with time-based stopping.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    current_bound = _compute_c3_bound(h)
    start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0
        best_move = None  # (type, i, j) where type is 'flip' or 'swap'
        
        # Try flipping each position
        for i in range(n):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _fast_flip_eval(h, i, dx)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_move = ('flip', i, -1)
        
        # Try pairwise swaps if enabled
        if use_pairwise:
            # Sample a subset of pairs to keep runtime manageable
            n_pairs_to_try = min(500, n * 2)
            pair_indices = np.random.choice(n * (n - 1) // 2, size=n_pairs_to_try, replace=False)
            for pi in pair_indices:
                # Convert linear index to (i, j) with i < j
                # Number of pairs before row i is i*(n-1) - i*(i-1)//2
                i = 0
                while True:
                    count_before = i * (n - 1) - i * (i - 1) // 2
                    if pi < count_before + (n - 1 - i):
                        j_offset = pi - count_before
                        j = i + 1 + j_offset
                        break
                    i += 1
                
                if i >= j or i >= n or j >= n:
                    continue
                
                # Try swapping signs at i and j
                h[i], h[j] = h[j], h[i]
                new_bound = _compute_c3_bound(h)
                h[i], h[j] = h[j], h[i]
                
                gain = current_bound - new_bound
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('swap', i, j)
        
        if best_move is not None:
            move_type, i, j = best_move
            if move_type == 'flip':
                h[i] = -h[i]
            else:
                h[i], h[j] = h[j], h[i]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


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
    for trial in range(15):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization with pairwise swaps
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        local_time = min(remaining - 15, 80)
        best_h, best_bound = _local_search(best_h, max_iter=3000, time_budget=local_time, use_pairwise=True)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        for trial in range(8):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            
            if bound < best_bound or trial < 3:  # Always explore early restarts
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 25)
                h, bound = _local_search(h, max_iter=1500, time_budget=local_time, use_pairwise=True)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
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
# EVOLVE-BLOCK-END
