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

def _steepest_descent(s, max_time=5.0):
    """Steepest descent local search: find the single best flip in each pass."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        best_flip_idx = -1
        best_val = current_val
        
        # Find the flip that yields the lowest autocorrelation max abs
        for i in range(N):
            s[i] = -s[i]
            new_val = _autocorr_max_abs(s)
            if new_val < best_val:
                best_val = new_val
                best_flip_idx = i
            # Revert for next iteration of loop
            s[i] = -s[i]
        
        if best_flip_idx != -1:
            s[best_flip_idx] = -s[best_flip_idx]
            current_val = best_val
        else:
            break
            
    return s, current_val

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    best_s = None
    best_val = float('inf')
    
    # Try multiple phases of the sine wave initialization
    # The directive suggests using zero crossings of a low-frequency sine wave
    # s[i] = sign(sin(2 * pi * k * i / N + phase))
    # k=1 is the lowest frequency (one period over N).
    
    phases = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    
    for phase in phases:
        if time.time() > deadline - 5.0:
            break
            
        # Initialize with sine wave
        i = np.arange(N)
        # Use k=1 for lowest frequency. 
        # Note: sin(2*pi*i/N) has 2 zero crossings in [0, N).
        # This creates blocks of +1 and -1.
        s_init = np.sign(np.sin(2 * np.pi * i / N + phase)).astype(np.int8)
        # Handle zeros (unlikely but possible) by mapping to 1
        s_init[s_init == 0] = 1
        
        # Run a short steepest descent
        s_opt, val = _steepest_descent(s_init, max_time=5.0)
        
        if val < best_val:
            best_val = val
            best_s = s_opt
            
        # If we have time, do a longer simulated annealing on the best candidate found so far
        remaining = deadline - time.time()
        if remaining > 20:
             # Only do SA on the current best to save time
             if best_s is not None:
                 s_sa, val_sa = _simulated_annealing(best_s, max_time=min(remaining - 5, 30.0))
                 if val_sa < best_val:
                     best_val = val_sa
                     best_s = s_sa
             break # We used up the time for SA, so stop trying other phases
    
    # Final local search on the best result
    if best_s is not None:
        remaining = deadline - time.time()
        if remaining > 2:
            best_s, best_val = _steepest_descent(best_s, max_time=min(remaining - 1, 5.0))
            
    if best_s is None:
        # Fallback
        np.random.seed(42)
        best_s = np.random.choice([-1, 1], size=N).astype(np.int8)
        
    return best_s
# EVOLVE-BLOCK-END
