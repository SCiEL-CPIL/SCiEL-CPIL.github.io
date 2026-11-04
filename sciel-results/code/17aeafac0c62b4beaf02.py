"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(h):
    """Compute max absolute value of the autoconvolution of step function h.
    
    h: array of N step heights on intervals of width delta.
    The autoconvolution is computed as discrete convolution scaled by delta^2.
    """
    delta = 0.5 / N  # interval width: total range 1/2, N intervals
    # Discrete autoconvolution: (h * h)[k] = sum_i h[i] * h[i+k] for lag k
    # Use FFT for efficiency
    n = len(h)
    # Pad to avoid circular convolution
    n_fft = 1 << (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)
    # The linear convolution is in the first 2n-1 elements
    conv_linear = conv[:2*n-1]
    # Scale by delta^2 for the actual autoconvolution integral
    conv_scaled = conv_linear * (delta * delta)
    return np.max(np.abs(conv_scaled))


def _local_search(h, max_iters=5000, time_limit=100.0):
    """Local search to minimize max abs autoconvolution."""
    start_time = time.time()
    current = h.copy().astype(float)
    current_score = _autoconvolution_max_abs(current)
    
    best = current.copy()
    best_score = current_score
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation: flip a random subset of signs or adjust values
        idx = np.random.randint(N)
        # Try flipping sign or small perturbation
        if np.random.random() < 0.5:
            new_h = current.copy()
            new_h[idx] = -new_h[idx]
        else:
            new_h = current.copy()
            new_h[idx] += np.random.normal(0, 0.1)
        
        new_score = _autoconvolution_max_abs(new_h)
        if new_score < current_score:
            current = new_h
            current_score = new_score
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
    
    return best, best_score


def construct():
    start_time = time.time()
    
    # Start with alternating signs pattern
    h = np.ones(N)
    h[1::2] = -1.0
    
    # Local search
    h, score = _local_search(h, max_iters=20000, time_limit=110.0)
    
    return h
# EVOLVE-BLOCK-END
