"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_max_abs(s):
    """Compute the max absolute value of the circular autocorrelation of s."""
    n = len(s)
    # Use FFT for efficient autocorrelation
    F = np.fft.fft(s, n=2*n)
    acf = np.fft.ifft(F * np.conj(F)).real[:n]
    # The autocorrelation at lag 0 is the energy; we want max abs of non-zero lags
    # But for the upper bound, we might need all lags including 0
    # Let's return max abs over all lags
    return np.max(np.abs(acf))

def _score(s):
    """Compute the upper bound score (lower is better)."""
    return _autocorr_max_abs(s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill
    
    # Initialize with random signs
    rng = np.random.default_rng(42)
    best_s = rng.choice([-1, 1], size=N)
    best_score = _score(best_s)
    
    # Simulated annealing with block flips
    T = 1.0  # Temperature
    T_min = 1e-6
    cooling = 0.999
    
    current_s = best_s.copy()
    current_score = best_score
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Choose block size (4 to 32)
        block_size = rng.integers(4, 33)
        start = rng.integers(0, N - block_size + 1)
        
        # Flip the block
        new_s = current_s.copy()
        new_s[start:start+block_size] *= -1
        
        new_score = _score(new_s)
        
        # Accept or reject
        if new_score < current_score:
            current_s = new_s
            current_score = new_score
            if current_score < best_score:
                best_s = current_s.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if rng.random() < np.exp(-delta / T):
                current_s = new_s
                current_score = new_score
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Occasionally restart if stuck
        if iteration % 1000 == 0 and T < 0.01:
            T = 0.5  # Reheat
    
    return best_s
# EVOLVE-BLOCK-END

