"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _fast_autocorr_max_abs(x):
    """Compute max absolute value of linear autocorrelation using FFT."""
    n = len(x)
    # Next power of 2 >= 2n-1 for linear convolution
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fft_x = np.fft.fft(x, n=size)
    ac = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
    return np.max(np.abs(ac))

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # 5s safety margin before 120s limit
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], N)
    current_val = _fast_autocorr_max_abs(x)
    best_x = x.copy()
    best_val = current_val
    
    # Simulated Annealing parameters
    start_temp = 10.0
    end_temp = 0.01
    alpha = (end_temp / start_temp) ** (1.0 / 100000)
    temp = start_temp
    
    while time.time() < deadline and temp > end_temp:
        # Randomly pick a few indices to flip (batch size 1 for simplicity, or small batch)
        # Flipping 1 bit at a time is standard for binary search, but maybe 2-3 helps escape?
        # Let's stick to 1 bit for stability, SA handles the rest.
        i = random.randint(0, N - 1)
        x[i] = -x[i]
        
        new_val = _fast_autocorr_max_abs(x)
        
        if new_val < current_val or random.random() < np.exp(-(new_val - current_val) / temp):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[i] = -x[i] # revert
            
        temp *= alpha
        
    # Final local search polish using the fast evaluator
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            old = x[i]
            x[i] = -old
            val = _fast_autocorr_max_abs(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                improved = True
            else:
                x[i] = old
                
    return best_x
# EVOLVE-BLOCK-END
