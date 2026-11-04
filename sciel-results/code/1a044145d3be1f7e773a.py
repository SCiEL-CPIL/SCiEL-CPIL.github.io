"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 array."""
    tm = np.array([1], dtype=np.float64)
    while len(tm) < n:
        tm = np.concatenate([tm, -tm])
    return tm[:n]


def _autoconv_peak(x):
    """Compute the maximum absolute value of the autoconvolution of x."""
    # Autoconvolution via FFT for efficiency
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * np.conj(f), n=2*n)
    # The full autoconvolution has length 2*n-1, but irfft gives 2*n
    # The peak is the max absolute value
    return np.max(np.abs(ac))


def _local_search(x, max_iters=5000, time_limit=120.0):
    """Greedy local search to minimize autoconvolution peak."""
    start_time = time.time()
    n = len(x)
    current_peak = _autoconv_peak(x)
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Try flipping each position and keep the best improvement
        best_pos = -1
        best_peak = current_peak
        
        # Sample a subset of positions for speed
        if n > 200:
            candidates = np.random.choice(n, size=50, replace=False)
        else:
            candidates = range(n)
        
        for pos in candidates:
            x[pos] = -x[pos]
            peak = _autoconv_peak(x)
            x[pos] = -x[pos]  # revert
            
            if peak < best_peak:
                best_peak = peak
                best_pos = pos
        
        if best_pos >= 0:
            x[best_pos] = -x[best_pos]
            current_peak = best_peak
        else:
            # No improvement found, try random restarts of the search
            if np.random.random() < 0.1:
                # Perturb a few random positions
                for _ in range(3):
                    pos = np.random.randint(n)
                    x[pos] = -x[pos]
                current_peak = _autoconv_peak(x)
    
    return x, current_peak


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initialize with Thue-Morse sequence
    x = _thue_morse(N).copy()
    current_peak = _autoconv_peak(x)
    
    # Run local search
    x, current_peak = _local_search(x, max_iters=10000, time_limit=130.0)
    
    # If we have time left, do a few more passes
    if time.time() < deadline - 10.0:
        x, current_peak = _local_search(x, max_iters=5000, time_limit=120.0)
    
    # Ensure we return within deadline
    if time.time() > deadline:
        # Return what we have
        pass
    
    return x
# EVOLVE-BLOCK-END
