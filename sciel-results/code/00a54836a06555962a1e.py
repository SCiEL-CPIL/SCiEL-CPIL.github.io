N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute max |autocorrelation| at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is lag 0, we want max |acf[k]| for k=1..n-1
    return np.max(np.abs(acf[1:n]))


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound (max absolute autocorrelation at non-zero lags).
    
    Strategy: 
    1. Initialize with random ±1 pattern
    2. Use greedy local search to flip positions that reduce the peak
    3. Time-budgeted to finish well within 120s
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Initialize with random ±1 pattern
    np.random.seed(42)
    h = np.random.choice([-1.0, 1.0], size=N)
    
    best_peak = _autocorr_peak(h)
    best_h = h.copy()
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    
    def compute_peak(h):
        H = np.fft.rfft(h, n=fft_size)
        acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
        return np.max(np.abs(acf[1:N]))
    
    # Greedy local search: iterate through positions, flip if it improves
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > time_limit:
                break
            # Try flipping position i
            h[i] = -h[i]
            new_peak = compute_peak(h)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]  # revert
    
    return best_h
# EVOLVE-BLOCK-END
