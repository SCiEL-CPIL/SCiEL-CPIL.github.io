"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_fft(x):
    """Compute full autoconvolution via FFT."""
    n = len(x)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    F = np.fft.rfft(x, fft_size)
    ac = np.fft.irfft(F * F, fft_size)
    return ac[:2 * n - 1]


def _score(x):
    """Compute max |autoconv| / (sum x)^2."""
    ac = _autoconv_fft(x)
    max_abs = np.max(np.abs(ac))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return float('inf')
    return max_abs / (s * s)


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin
    
    # Initialize with random signs
    best_x = np.random.choice([-1.0, 1.0], size=N)
    best_score = _score(best_x)
    
    # Simulated annealing with random sign flips
    x = best_x.copy()
    current_score = best_score
    
    # Temperature schedule
    T_start = 1.0
    T_end = 1e-6
    total_steps = 200000
    
    for step in range(total_steps):
        if time.time() > deadline:
            break
        
        # Compute temperature
        progress = step / total_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random position to flip
        i = np.random.randint(0, N)
        old_val = x[i]
        x[i] = -old_val
        
        new_score = _score(x)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            # Accept with probability exp(-(new-old)/T)
            delta = new_score - current_score
            if delta > 0 and np.random.random() < np.exp(-delta / (T + 1e-12)):
                current_score = new_score
            else:
                # Reject
                x[i] = old_val
    
    return best_x
# EVOLVE-BLOCK-END
