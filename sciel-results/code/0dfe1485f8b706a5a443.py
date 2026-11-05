"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from math import log, exp

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


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def _compute_conv_state(heights):
    """
    Precompute FFT state for incremental updates.
    Returns (h, f_fft, conv, dx, n) where conv is the real autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64).copy()
    n = len(h)
    dx = 0.5 / n
    
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    fft_size = 2 * (2 * n - 1)
    f_fft = np.fft.fft(h_pad, n=fft_size)
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    return h, f_fft, conv, dx, n, fft_size


def _flip_sign_inplace(h, f_fft, conv, i, fft_size):
    """
    Flip sign at index i and update conv incrementally.
    Returns new conv (modified in place).
    """
    old_val = h[i]
    h[i] = -old_val
    
    # The change in h_pad is -2*old_val at position i
    # This changes f_fft by -2*old_val * exp(-2j*pi*i*k/fft_size)
    # And conv changes by the inverse FFT of (delta_f * conj(f_fft) + f_fft * conj(delta_f))
    # = 2*Re(delta_f * conj(f_fft))
    
    delta = -2.0 * old_val
    # delta_f[k] = delta * exp(-2j*pi*i*k/fft_size)
    # We can compute this efficiently
    
    # conv_new = conv + 2*Re(delta_f * conj(f_fft))
    # delta_f = delta * exp(-2j*pi*i*arange/fft_size)
    k = np.arange(fft_size)
    delta_f = delta * np.exp(-2j * np.pi * i * k / fft_size)
    
    conv_change = 2.0 * np.real(delta_f * np.conj(f_fft))
    conv += np.fft.ifft(conv_change).real
    
    # Update f_fft
    f_fft += delta_f
    
    return conv


def _local_search_fast(heights, max_iter=5000, time_budget=10, start_time=0):
    """
    Fast local search using incremental FFT updates.
    Each flip is O(N) instead of O(N log N).
    """
    h, f_fft, conv, dx, n, fft_size = _compute_conv_state(heights)
    current_bound = dx * np.max(np.abs(conv))
    
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
        
        # Try flipping each position using incremental update
        for i in range(N):
            conv_before = conv.copy()
            _flip_sign_inplace(h, f_fft, conv, i, fft_size)
            new_bound = dx * np.max(np.abs(conv))
            gain = current_bound - new_bound
            
            # Revert
            h[i] = -h[i]
            delta = -2.0 * h[i]  # h[i] is now the old value
            k = np.arange(fft_size)
            delta_f = delta * np.exp(-2j * np.pi * i * k / fft_size)
            conv -= np.fft.ifft(2.0 * np.real(delta_f * np.conj(f_fft))).real
            f_fft -= delta_f
            # Restore conv
            conv[:] = conv_before
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            _flip_sign_inplace(h, f_fft, conv, best_idx, fft_size)
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def _soft_max_bound(conv, beta=50.0):
    """
    Compute soft-max approximation of max(|conv|) using log-sum-exp.
    This provides a smooth surrogate for the nonsmooth max.
    """
    abs_conv = np.abs(conv)
    # Use log-sum-exp for numerical stability
    m = np.max(abs_conv)
    if m == 0:
        return 0.0
    # soft_max = m + (1/beta) * log(sum(exp(beta*(x-m))))
    soft = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (abs_conv - m))))
    return soft


def _soft_local_search(heights, time_budget=15, start_time=0, beta=50.0):
    """
    Local search using soft-max surrogate to escape shallow local minima.
    """
    h, f_fft, conv, dx, n, fft_size = _compute_conv_state(heights)
    current_soft = dx * _soft_max_bound(conv, beta)
    current_hard = dx * np.max(np.abs(conv))
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    while improved:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        best_gain = 0
        best_idx = -1
        
        for i in range(N):
            conv_before = conv.copy()
            _flip_sign_inplace(h, f_fft, conv, i, fft_size)
            new_soft = dx * _soft_max_bound(conv, beta)
            gain = current_soft - new_soft
            
            # Revert
            h[i] = -h[i]
            delta = -2.0 * h[i]
            k = np.arange(fft_size)
            delta_f = delta * np.exp(-2j * np.pi * i * k / fft_size)
            conv -= np.fft.ifft(2.0 * np.real(delta_f * np.conj(f_fft))).real
            f_fft -= delta_f
            conv[:] = conv_before
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            _flip_sign_inplace(h, f_fft, conv, best_idx, fft_size)
            current_soft -= best_gain
            current_hard = dx * np.max(np.abs(conv))
            improved = True
    
    return h, current_hard


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
    
    # Try random initializations with more trials
    np.random.seed(42)
    for trial in range(20):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Fast local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        local_time = min(remaining - 15, 60)
        best_h, best_bound = _local_search_fast(best_h, max_iter=10000, time_budget=local_time, start_time=start_time)
    
    # Soft-max local search to escape local minima
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        soft_time = min(remaining - 10, 20)
        best_h, best_bound = _soft_local_search(best_h, time_budget=soft_time, start_time=start_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        for trial in range(3):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 10, 25)
                h, bound = _local_search_fast(h, max_iter=5000, time_budget=local_time, start_time=start_time)
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
