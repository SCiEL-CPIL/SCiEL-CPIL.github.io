"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak_sidelobe(f):
    """Compute the max |R(k)| for k != 0 using FFT-based autocorrelation."""
    # Zero-pad to avoid circular correlation
    n = len(f)
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:2*n-1]
    # acf has length 2*n-1, with acf[n-1] being R(0)
    # Sidelobes are acf[0:n-1] and acf[n:2*n-1]
    left = acf[0:n-1]   # R(-k) for k=1..n-1
    right = acf[n:2*n-1] # R(k) for k=1..n-1
    return max(np.max(np.abs(left)), np.max(np.abs(right)))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Initial: all ones (seed)
    f = np.ones(N, dtype=np.float64)
    best_f = f.copy()
    best_val = _autocorr_peak_sidelobe(f)
    
    # Simulated annealing with block flips
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Block sizes to try
    block_sizes = [4, 8, 16, 24, 32]
    
    while T > T_min:
        if time.time() > deadline - 5.0:
            break
        
        # Choose a random block
        bs = block_sizes[np.random.randint(len(block_sizes))]
        start = np.random.randint(0, N - bs + 1)
        
        # Flip the block
        f[start:start+bs] *= -1
        
        new_val = _autocorr_peak_sidelobe(f)
        delta = new_val - best_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            if new_val < best_val:
                best_val = new_val
                best_f = f.copy()
        else:
            # Reject
            f[start:start+bs] *= -1
        
        T *= cooling
    
    return best_f
# EVOLVE-BLOCK-END
