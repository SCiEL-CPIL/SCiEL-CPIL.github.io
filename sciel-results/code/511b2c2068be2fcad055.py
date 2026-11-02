"""Signed step function for autoconvolution upper bound C3."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autocorr(s):
    """Compute autocorrelation of sequence s (length N) using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(f * np.conj(f), n=n_fft)[:N]
    return acf


def _peak_sidelobe(s):
    """Return max |acf[k]| for k = 1..N-1."""
    acf = _autocorr(s)
    return float(np.max(np.abs(acf[1:])))


def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s
    
    rng = np.random.default_rng(777)
    
    # Multiple random initializations
    best_s = None
    best_val = np.inf
    
    for i in range(20):
        s = rng.choice([-1.0, 1.0], size=N)
        val = _peak_sidelobe(s)
        if val < best_val:
            best_val = val
            best_s = s.copy()
    
    # Simulated annealing with single-bit and multi-bit flips
    s = best_s.copy()
    current_val = best_val
    T = 8.0
    T_min = 0.001
    alpha = 0.9997
    
    while time.time() < deadline:
        # Decide move type
        if rng.random() < 0.85:
            # Single flip
            idx = rng.integers(0, N)
            s[idx] *= -1.0
            new_val = _peak_sidelobe(s)
            delta = new_val - current_val
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_s = s.copy()
            else:
                s[idx] *= -1.0  # revert
        else:
            # Multi-flip (2-10 bits)
            k = rng.integers(2, 11)
            idxs = rng.choice(N, size=k, replace=False)
            s[idxs] *= -1.0
            new_val = _peak_sidelobe(s)
            delta = new_val - current_val
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_s = s.copy()
            else:
                s[idxs] *= -1.0  # revert
        
        T = max(T * alpha, T_min)
    
    return best_s
# EVOLVE-BLOCK-END
