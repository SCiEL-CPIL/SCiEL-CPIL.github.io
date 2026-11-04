"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """Compute max |c(k)| for k != 0, where c is the autoconvolution of f."""
    n = len(f)
    # Zero-pad to 2n for linear convolution
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # conv[k] = sum_{i=0}^{n-1-k} f[i] * f[i+k] for k=0,...,n-1
    # We want max |conv[k]| for k in 1..n-1
    if n <= 1:
        return 0.0
    return np.max(np.abs(conv[1:]))


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    best_f = None
    best_score = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with random ±1
        f = rng.choice([-1.0, 1.0], size=N)
        
        # Local search: try flipping each position
        improved = True
        while improved:
            if time.time() > deadline:
                break
            
            improved = False
            current_score = _autoconv_max_abs(f)
            
            # Try flipping each position, keep the best improvement
            best_flip = -1
            best_new_score = current_score
            
            for i in range(N):
                if time.time() > deadline:
                    break
                f[i] *= -1.0
                new_score = _autoconv_max_abs(f)
                f[i] *= -1.0  # restore
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_flip = i
            
            if best_flip >= 0 and best_new_score < current_score:
                f[best_flip] *= -1.0
                improved = True
        
        # Update best
        score = _autoconv_max_abs(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
