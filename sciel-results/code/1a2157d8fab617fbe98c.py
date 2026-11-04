"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(h):
    """Compute full autoconvolution of step function h (length N)."""
    N = len(h)
    # Full convolution length is 2*N - 1
    # Use FFT for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    return conv


def _max_abs_autoconv(h):
    """Compute max absolute value of autoconvolution."""
    conv = _autoconv(h)
    return np.max(np.abs(conv))


def _local_search(h, max_time=100.0):
    """Local search by flipping individual bits."""
    start_time = time.time()
    current = h.copy().astype(np.float64)
    current_cost = _max_abs_autoconv(current)
    N = len(current)
    
    # Precompute indices
    indices = np.arange(N)
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        # Try flipping each position
        best_gain = 0
        best_idx = -1
        
        # Sample a subset if N is large for speed
        if N > 100:
            sample_size = min(50, N)
            sample_idx = np.random.choice(N, sample_size, replace=False)
        else:
            sample_idx = indices
        
        for idx in sample_idx:
            if time.time() - start_time > max_time:
                break
            
            # Flip sign at idx
            trial = current.copy()
            trial[idx] = -trial[idx]
            trial_cost = _max_abs_autoconv(trial)
            gain = current_cost - trial_cost
            
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx >= 0:
            current[best_idx] = -current[best_idx]
            current_cost -= best_gain
        else:
            break
    
    return current, current_cost


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with all +1
    h = np.ones(N, dtype=np.float64)
    
    # Do local search
    h, cost = _local_search(h, max_time=100.0)
    
    # Check if we're still within deadline
    if time.time() < deadline:
        # Try a few more restarts with different initializations
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            h_rand = np.random.choice([-1.0, 1.0], size=N)
            h_rand, cost_rand = _local_search(h_rand, max_time=15.0)
            
            if cost_rand < cost:
                h = h_rand
                cost = cost_rand
    
    return h
# EVOLVE-BLOCK-END
