"""Minimize max |autoconvolution| of ±1 sequence via local search."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_autocorr(x):
    """Compute max |autocorrelation| of a ±1 sequence using FFT."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return np.max(np.abs(acf))

def construct():
    t0 = time.time()
    deadline = t0 + 128.0  # 2s safety margin
    
    rng = np.random.default_rng(7)
    best_x = None
    best_val = np.inf
    
    # Multiple random restarts with steepest-descent local search
    for restart in range(200):
        if time.time() > deadline:
            break
        
        x = rng.choice([-1.0, 1.0], size=N)
        val = _max_autocorr(x)
        
        if val < best_val:
            best_x = x.copy()
            best_val = val
        
        # Steepest descent: repeatedly flip the bit that most reduces peak autocorr
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            best_idx = -1
            best_new_val = val
            for idx in range(N):
                if time.time() > deadline:
                    break
                old = x[idx]
                x[idx] = -old
                nv = _max_autocorr(x)
                x[idx] = old
                if nv < best_new_val:
                    best_new_val = nv
                    best_idx = idx
            if best_idx >= 0:
                x[best_idx] = -x[best_idx]
                val = best_new_val
                improved = True
                if val < best_val:
                    best_x = x.copy()
                    best_val = val
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
