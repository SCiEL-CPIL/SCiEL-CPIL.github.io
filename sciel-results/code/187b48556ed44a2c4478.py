"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _c3_bound(x):
    """Compute the C3 upper bound: max absolute autocorrelation sidelobe."""
    n = len(x)
    # Compute autocorrelation using FFT for efficiency
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2 * n:
        size *= 2
    x_padded = np.zeros(size)
    x_padded[:n] = x
    fft_x = np.fft.fft(x_padded)
    autocorr = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
    # Sidelobes are lags 1 to n-1
    if n > 1:
        return np.max(np.abs(autocorr[1:]))
    return 0.0


def construct():
    deadline = time.time() + 110  # Safety margin: 110s out of 120s limit
    
    # Start with random ±1 sequence
    np.random.seed(42)
    x = np.random.choice([-1, 1], size=N)
    best_x = x.copy()
    best_val = _c3_bound(best_x)
    
    # Local search: flip individual elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping element i
            x[i] *= -1
            val = _c3_bound(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                improved = True
            else:
                x[i] *= -1  # revert
    
    return best_x
# EVOLVE-BLOCK-END
