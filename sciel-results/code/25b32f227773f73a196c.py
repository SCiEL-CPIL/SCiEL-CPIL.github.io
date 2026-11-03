"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(s):
    """Compute the maximum absolute value of the circular autoconvolution of s."""
    # Use FFT for efficient convolution
    n = len(s)
    # Circular convolution via FFT
    fft_s = np.fft.rfft(s)
    conv = np.fft.irfft(fft_s * np.conj(fft_s), n=n)
    return np.max(np.abs(conv))

def _score(s):
    """Score is the upper bound on C3, which we approximate by the max autoconvolution."""
    return _autoconv_max_abs(s)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    # Start with random signs
    np.random.seed(42)
    best_s = np.random.choice([-1.0, 1.0], size=N)
    best_score = _score(best_s)
    
    # Simulated annealing with block flips
    T = 1.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.001
    
    while time.time() < deadline and T > min_T:
        # Random block size between 4 and 32
        block_size = np.random.randint(4, 33)
        # Random starting position
        start_idx = np.random.randint(0, N)
        
        # Create flip mask
        flip_mask = np.zeros(N, dtype=bool)
        for i in range(block_size):
            idx = (start_idx + i) % N
            flip_mask[idx] = True
        
        # Flip signs in the block
        new_s = best_s.copy()
        new_s[flip_mask] *= -1
        
        new_score = _score(new_s)
        
        # Accept if better or with probability exp(-(delta)/T)
        delta = new_score - best_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            best_s = new_s
            best_score = new_score
        
        T *= cooling_rate
    
    # Final polish with local search (single bit flips)
    while time.time() < deadline:
        improved = False
        for i in range(N):
            trial_s = best_s.copy()
            trial_s[i] *= -1
            trial_score = _score(trial_s)
            if trial_score < best_score:
                best_s = trial_s
                best_score = trial_score
                improved = True
        if not improved:
            break
    
    return best_s
# EVOLVE-BLOCK-END
