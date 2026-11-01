"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_acf(h):
    """Compute the autocorrelation of h using FFT."""
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    return acf


def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags."""
    acf = _compute_acf(h)
    return np.max(np.abs(acf[1:]))


def _flip_acf_delta(h, acf, i):
    """
    Compute the change in ACF when flipping h[i].
    This is O(N) and avoids recomputing the full FFT.
    """
    n = len(h)
    delta = np.zeros(n)
    # When we flip h[i] to -h[i], the change in acf[k] is:
    # new_acf[k] = sum_j h[j]*h[j+k] with h[i] flipped
    # The terms involving h[i] change sign.
    # For lag k: acf[k] = sum_j h[j]*h[j+k]
    # Terms where j = i: h[i]*h[i+k] -> -h[i]*h[i+k] (change: -2*h[i]*h[i+k])
    # Terms where j+k = i (i.e., j = i-k): h[i-k]*h[i] -> h[i-k]*(-h[i]) (change: -2*h[i-k]*h[i])
    # So delta[k] = -2*h[i]*h[i+k] - 2*h[i-k]*h[i] for valid indices
    # Note: acf is computed as sum_j h[j]*h[j+k] for k >= 0
    
    # For k >= 0:
    # Term 1: j = i, need i+k < n: delta[k] -= 2*h[i]*h[i+k]
    # Term 2: j+k = i, j = i-k, need i-k >= 0: delta[k] -= 2*h[i-k]*h[i]
    
    # Vectorized computation
    # Term 1: for k in range(n-i), delta[k] -= 2*h[i]*h[i+k]
    if i < n:
        idx1 = np.arange(n - i)
        delta[idx1] -= 2.0 * h[i] * h[i + idx1]
    
    # Term 2: for k in range(i+1), delta[k] -= 2*h[i-k]*h[i]
    if i >= 0:
        idx2 = np.arange(i + 1)
        delta[idx2] -= 2.0 * h[i - idx2] * h[i]
    
    return delta


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(12345)
    
    # Phase 1: Initialize with a random sequence
    h = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
    acf = _compute_acf(h)
    best_val = np.max(np.abs(acf[1:]))
    best_h = h.copy()
    
    # Phase 2: Simulated Annealing
    T0 = 5.0  # Initial temperature
    T_min = 0.01
    cooling = 0.9995  # Cooling rate per step
    
    T = T0
    step = 0
    
    while time.time() < deadline - 5.0:
        # Pick a random position to flip
        i = np.random.randint(0, N)
        
        # Compute the change in ACF from flipping h[i]
        delta = _flip_acf_delta(h, acf, i)
        
        # New ACF if we accept
        new_acf = acf + delta
        new_val = np.max(np.abs(new_acf[1:]))
        
        # Decide whether to accept
        if new_val <= best_val or np.random.random() < np.exp(-(new_val - best_val) / max(T, 1e-10)):
            # Accept the flip
            h[i] = -h[i]
            acf = new_acf
            
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        
        # Cool down
        T = max(T * cooling, T_min)
        step += 1
        
        # Periodically recompute ACF from scratch to avoid drift
        if step % 10000 == 0:
            acf = _compute_acf(h)
    
    # Phase 3: Greedy local search from best_h
    h = best_h.copy()
    acf = _compute_acf(h)
    current_val = np.max(np.abs(acf[1:]))
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        best_i = -1
        best_new_val = current_val
        
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            
            delta = _flip_acf_delta(h, acf, i)
            new_val = np.max(np.abs((acf + delta)[1:]))
            
            if new_val < best_new_val:
                best_new_val = new_val
                best_i = i
        
        if best_i >= 0 and best_new_val < current_val:
            h[best_i] = -h[best_i]
            delta = _flip_acf_delta(h, acf, best_i)
            # Recompute properly since we flipped
            acf = _compute_acf(h)
            current_val = np.max(np.abs(acf[1:]))
            improved = True
    
    # Return the best sequence found
    return h
# EVOLVE-BLOCK-END
