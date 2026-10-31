"""Rewritten program: real-valued window shapes with non-zero sum to minimize
the autoconvolution sidelobe energy (the C3 upper bound)."""
import time
import numpy as np

N = 400

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 6.0 - (time.time() - _START)   # 6 s safety margin


def _sidelobe_energy(f):
    """C3 = sum over non-DC lags of |autoconvolution lag|.
    autoconv lag k = sum_i f_i * f_{i+k}  (linear, zero-padded)."""
    N = len(f)
    ac = np.correlate(f, f, mode='full')   # length 2N-1, center at N-1 is lag 0
    ac = np.asarray(ac, dtype=float)
    c = N - 1
    # zero out DC (lag 0)
    ac2 = ac.copy()
    ac2[c] = 0.0
    return float(np.sum(np.abs(ac2)))


def _gaussian_window(sigma_frac=0.1):
    """Generate a Gaussian window of length N. 
    Does NOT subtract the mean, so sum is non-zero."""
    n = np.arange(N)
    center = (N - 1) / 2.0
    sigma = sigma_frac * N
    w = np.exp(-((n - center) ** 2) / (2.0 * sigma ** 2))
    return w


def construct():
    # Generate a Gaussian window. 
    # A Gaussian is smooth (low-frequency content) and its autoconvolution 
    # is also a Gaussian, which decays rapidly, minimizing the L1 norm of the tail.
    # We do not center it (subtract mean) to ensure the sum is non-zero.
    
    # Try a few sigmas to see which gives the best (lowest) sidelobe energy
    best_val = np.inf
    best_f = None
    
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    for sig in sigmas:
        if _remaining() < 1.0:
            break
        w = _gaussian_window(sig)
        # Normalize to unit L2 norm to have a consistent scale
        norm = np.linalg.norm(w)
        if norm == 0:
            continue
        w = w / norm
        
        # Check constraint: sum must not be zero
        if np.isclose(np.sum(w), 0.0):
            # Add a tiny constant to break symmetry if needed, though Gaussian is positive
            w += 1e-6
            w = w / np.linalg.norm(w)
            
        val = _sidelobe_energy(w)
        if val < best_val:
            best_val = val
            best_f = w

    if best_f is None:
        # Fallback: a simple constant vector (sum is non-zero)
        best_f = np.ones(N) / np.sqrt(N)

    return best_f
