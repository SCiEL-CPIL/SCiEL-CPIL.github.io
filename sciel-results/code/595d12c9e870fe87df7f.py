"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import numpy as np
import time

# EVOLVE-BLOCK-START
def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags for a length-N sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
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

def _simulated_annealing(s, max_time=60.0):
    """Simulated annealing to minimize max |autocorr|."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    best_s = s.copy()
    best_val = current_val
    start_time = time.time()
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        # Flip a random element
        i = np.random.randint(N)
        s[i] = -s[i]
        new_val = _autocorr_max_abs(s)
        
        delta = new_val - current_val
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
        else:
            s[i] = -s[i]  # revert
        
        T *= cooling
    
    return best_s, best_val

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with a random ±1 sequence
    np.random.seed(42)
    s = np.random.choice([-1, 1], size=N)
    
    # Local search first
    s, val = _local_search(s, max_time=5.0)
    
    # Simulated annealing with remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        s, val = _simulated_annealing(s, max_time=min(remaining - 5, 60.0))
    
    # Final local search
    remaining = deadline - time.time()
    if remaining > 2:
        s, val = _local_search(s, max_time=min(remaining - 1, 5.0))
    
    return s
# EVOLVE-BLOCK-END
