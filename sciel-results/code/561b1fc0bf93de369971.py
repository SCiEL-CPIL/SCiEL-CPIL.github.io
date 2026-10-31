"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    # Use FFT for efficient convolution
    size = 2 * n - 1
    # Pad to next power of 2 for faster FFT
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping sign
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # Try multiple random initializations with local sign-flip optimization
    np.random.seed(42)
    
    # Use more restarts since FFT is faster
    n_restarts = 100
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if r % 5 == 0:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Mostly +1 with some random -1
            h = np.ones(N)
            n_neg = np.random.randint(N//4, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 5 == 2:
            # Chirp-like pattern
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * x * (x+1) / (2*N)))
        elif r % 5 == 3:
            # Random with continuous values
            h = np.random.uniform(-1, 1, size=N)
        else:
            # Block pattern with random block sizes
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(1, N//10 + 1)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        
        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            # Flip to increase sum
            if s < 0:
                h = -h
            # Add a bias
            h = h + 0.5
            h = np.sign(h)
        
        # Local sign-flip optimization
        h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h
# EVOLVE-BLOCK-END
