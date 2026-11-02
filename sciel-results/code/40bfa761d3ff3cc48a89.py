"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time

# EVOLVE-BLOCK-START

def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags for a length-N ±1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k] for k=0..N-1
    # We want max |acf[k]| for k=1..N-1
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _local_search(s, max_time=5.0):
    """Greedy local search: flip individual elements to reduce max |autocorr|."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        # Try flipping each element
        for i in range(N):
            s[i] = -s[i]
            new_val = _autocorr_max_abs(s)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                s[i] = -s[i]  # revert
        
        if not improved:
            break
    
    return s, current_val

def _simulated_annealing(s, max_time=10.0):
    """Simulated annealing to escape local minima."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start_time = time.time()
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    
    T = T_start
    while time.time() - start_time < max_time and T > T_end:
        # Pick a random position to flip
        i = np.random.randint(N)
        s[i] = -s[i]
        new_val = _autocorr_max_abs(s)
        
        delta = new_val - current_val
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_val = new_val
        else:
            s[i] = -s[i]  # revert
        
        T *= cooling_rate
    
    return s, current_val

def construct():
    """Construct a signed step function to minimize max absolute autocorrelation."""
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    best_s = np.ones(N)
    best_val = _autocorr_max_abs(best_s)
    
    # Try multiple random starts
    for seed in range(50):
        if time.time() > deadline - 20:
            break
        
        # Random initial sequence
        s = np.random.choice([-1, 1], size=N)
        
        # Local search
        s_ls, val_ls = _local_search(s, max_time=5.0)
        
        if val_ls < best_val:
            best_val = val_ls
            best_s = s_ls.copy()
        
        # Simulated annealing if time permits
        if time.time() > deadline - 30:
            break
        
        s_sa, val_sa = _simulated_annealing(s_ls, max_time=10.0)
        
        if val_sa < best_val:
            best_val = val_sa
            best_s = s_sa.copy()
    
    return best_s

# EVOLVE-BLOCK-END
