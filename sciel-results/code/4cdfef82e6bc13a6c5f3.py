"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(x):
    """Compute peak sidelobe level of autocorrelation (max abs at non-zero lags)."""
    n = len(x)
    # Use FFT for efficient autocorrelation
    size = 2 * n
    fft_x = np.fft.rfft(x, size)
    acf = np.fft.irfft(fft_x * np.conj(fft_x), size)
    # acf[0] is lag 0, acf[1:n] are positive lags 1..n-1
    # The peak sidelobe is max of |acf[k]| for k != 0
    if n > 1:
        peak = np.max(np.abs(acf[1:n]))
    else:
        peak = 0.0
    return peak


def _local_search(x, time_limit):
    """Greedy local search: flip signs to reduce peak sidelobe."""
    x = x.copy()
    n = len(x)
    size = 2 * n
    current = _autocorr_peak(x)
    start = time.time()
    
    # Precompute FFT size
    # Iterate: try flipping each position, keep if improves
    while time.time() - start < time_limit - 5:
        improved = False
        for i in range(n):
            if time.time() - start > time_limit - 5:
                break
            old = x[i]
            x[i] = -old
            new = _autocorr_peak(x)
            if new < current:
                current = new
                improved = True
            else:
                x[i] = old
        if not improved:
            break
    
    return x, current


def construct():
    t0 = time.time()
    deadline = t0 + 140  # Safety margin
    
    # Initialize with a structured pattern: alternating +1, -1
    # This gives a good starting point for low autocorrelation
    x = np.ones(N)
    x[1::2] = -1.0
    
    # Run local search with time budget
    remaining = deadline - time.time()
    if remaining > 10:
        x, _ = _local_search(x, remaining)
    
    return x
# EVOLVE-BLOCK-END
