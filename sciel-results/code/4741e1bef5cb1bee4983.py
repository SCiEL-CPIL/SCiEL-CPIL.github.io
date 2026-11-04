"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max |f*f| over all lags using FFT."""
    n = len(f)
    conv = np.fft.ifft(np.fft.fft(f, 2*n) ** 2).real
    return np.max(np.abs(conv))


def construct():
    t0 = time.time()
    deadline = t0 + 105.0  # safety margin before 120s
    
    # Start with alternating ±1
    f = np.ones(N)
    f[::2] = -1
    
    best_f = f.copy()
    best_val = _autoconvolution_max_abs(f)
    
    # Local search with limited sweeps
    for sweep in range(100):  # cap sweeps
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            f[i] *= -1
            val = _autoconvolution_max_abs(f)
            if val < best_val:
                best_val = val
                best_f = f.copy()
                improved = True
            else:
                f[i] *= -1  # revert
        if not improved:
            break
    
    return best_f
# EVOLVE-BLOCK-END
