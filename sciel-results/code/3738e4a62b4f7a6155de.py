"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_autocorr_sidelobe(h):
    """Compute max |autocorr| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return np.max(np.abs(acf[1:]))


def _local_search(h, deadline):
    """Greedy local search: flip bits to reduce max sidelobe."""
    best_h = h.copy()
    best_val = _max_autocorr_sidelobe(best_h)
    
    while time.time() < deadline:
        improved = False
        # Try flipping each bit
        for i in range(N):
            h[i] = -h[i]
            val = _max_autocorr_sidelobe(h)
            h[i] = -h[i]  # revert
            
            if val < best_val:
                h[i] = -h[i]  # keep flip
                best_val = val
                best_h = h.copy()
                improved = True
        
        if not improved:
            break
    
    return best_h, best_val


def _random_restart(h, deadline):
    """Perform random restarts with local search."""
    best_h = h.copy()
    best_val = _max_autocorr_sidelobe(best_h)
    
    while time.time() < deadline:
        # Create a new random signed step function
        h_new = np.random.choice([-1, 1], size=N)
        
        # Run local search on the new starting point
        h_local, val_local = _local_search(h_new, deadline)
        
        if val_local < best_val:
            best_h = h_local
            best_val = val_local
    
    return best_h, best_val


def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound."""
    # Set a safety margin: aim to finish well before 120s
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Start with a random initialization
    np.random.seed(42)  # Reproducibility
    h = np.random.choice([-1, 1], size=N)
    
    # First, run local search from the initial random start
    h, best_val = _local_search(h, deadline)
    
    # Then, try random restarts if time permits
    if time.time() < deadline - 5.0:  # Keep 5s buffer
        h, best_val = _random_restart(h, deadline)
    
    return h
# EVOLVE-BLOCK-END
