"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_peak(h):
    """
    Compute the maximum absolute value of the linear autocorrelation.
    h: array of +/- 1
    """
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    # Zero-pad to avoid circular convolution artifacts
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
        
    H = np.fft.rfft(h, n=fft_size)
    # Power spectrum
    power = np.abs(H) ** 2
    # Inverse FFT to get autocorrelation
    acf = np.fft.irfft(power, n=fft_size)
    
    # The linear autocorrelation is in acf[0:n] (and symmetric part)
    # acf[k] = sum h[i] h[i+k]
    # We want max |acf[k]| for k != 0
    # acf[0] is N (or close to it)
    # We exclude lag 0
    return float(np.max(np.abs(acf[1:n])))

def _construct_random():
    """Generate a random +/- 1 sequence."""
    return np.random.choice([-1, 1], size=N)

def _local_search(h, time_limit=10.0):
    """
    Greedy local search to minimize peak autocorrelation.
    """
    best_h = h.copy()
    best_val = _autoconv_peak(best_h)
    n = len(best_h)
    start_time = time.time()
    
    # Greedy: try flipping each position, keep if it improves
    # To be fast, we limit the number of passes or use random order
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Random order to escape local minima
        order = np.random.permutation(n)
        for i in order:
            # Flip sign
            best_h[i] *= -1
            val = _autoconv_peak(best_h)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] *= -1  # Revert
    
    return best_h, best_val

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with a random sequence
    # Use a fixed seed for reproducibility if needed, but random is fine
    np.random.seed(42)
    current_h = _construct_random()
    current_val = _autoconv_peak(current_h)
    
    best_h = current_h
    best_val = current_val
    
    # Run local search
    if time.time() < deadline - 20.0:
        current_h, current_val = _local_search(current_h, time_limit=110.0)
        if current_val < best_val:
            best_h = current_h
            best_val = current_val
            
    # Optional: Try a few more random restarts if time permits
    while time.time() < deadline - 30.0:
        trial_h = _construct_random()
        trial_val = _autoconv_peak(trial_h)
        if trial_val < best_val:
            best_h = trial_h
            best_val = trial_val
            # Quick local search
            if time.time() < deadline - 30.0:
                best_h, best_val = _local_search(best_h, time_limit=20.0)
        else:
            # If random restarts aren't improving, break early
            # (This is a heuristic; might miss better basins, but saves time)
            pass

    # Ensure return type is float or int compatible with evaluator
    # The seed returned np.ones (float). We return int +/- 1.
    # To be safe, cast to float if the seed was float.
    return best_h.astype(np.float64)

# EVOLVE-BLOCK-END
