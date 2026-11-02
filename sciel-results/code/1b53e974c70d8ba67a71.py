import numpy as np
import time

N = 400

def _eval(f):
    """Compute the max absolute autocorrelation value excluding lag 0."""
    f_float = f.astype(np.float64)
    n = len(f_float)
    # Zero-pad for linear autocorrelation
    size = 2 * n
    fft_f = np.fft.fft(f_float, n=size)
    acf = np.fft.ifft(fft_f * np.conj(fft_f)).real[:n]
    # The autocorrelation at lag k is acf[k] for k=0..n-1
    # Exclude lag 0
    return np.max(np.abs(acf[1:]))

def construct():
    start_time = time.time()
    deadline = 130.0
    
    # Initialize with random signs
    rng = np.random.default_rng(12345)
    f = rng.choice([-1.0, 1.0], size=N)
    best_f = f.copy()
    best_val = _eval(best_f)
    current_val = best_val
    
    # Simulated annealing
    T_start = 0.5
    T_end = 0.005
    n_iterations = 300000
    
    block_sizes = [4, 6, 8, 10, 12, 16, 20, 24, 32]
    
    for i in range(n_iterations):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (i / max(n_iterations, 1))
        
        # Choose block size with bias toward smaller blocks early, larger later
        progress = i / n_iterations
        if progress < 0.3:
            bs = block_sizes[rng.integers(0, 4)]  # smaller blocks
        elif progress < 0.7:
            bs = block_sizes[rng.integers(2, 7)]  # medium blocks
        else:
            bs = block_sizes[rng.integers(0, len(block_sizes))]  # any block
        
        start = rng.integers(0, N - bs + 1)
        
        # Flip block
        f[start:start+bs] *= -1
        val_new = _eval(f)
        delta = val_new - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            current_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_f = f.copy()
        else:
            f[start:start+bs] *= -1  # revert
    
    return best_f
