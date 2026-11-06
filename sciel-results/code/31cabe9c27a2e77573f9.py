"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    ac = np.convolve(h, h)
    return abs(2.0 * N * np.max(ac) / (s * s))

def _fft_convolve(h):
    n = 2 * len(h)
    H = np.fft.rfft(h, n)
    ac = np.fft.irfft(H * H, n)[:2 * len(h) - 1]
    return ac

def _value_fast(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    ac = _fft_convolve(h)
    return abs(2.0 * N * np.max(ac) / (s * s))

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin
    
    # Start with a smooth Gaussian-like shape
    x = np.linspace(-1, 1, N)
    
    best_h = np.ones(N)
    best_val = _value_fast(best_h)
    
    # Try various Gaussian widths
    for sigma in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
        h = np.exp(-x**2 / (2 * sigma**2))
        # Ensure sum is not zero
        if abs(np.sum(h)) < 1e-10:
            continue
        v = _value_fast(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # Coordinate descent polishing on the best Gaussian
    h = best_h.copy()
    step = 0.1
    improved = True
    iters = 0
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        iters += 1
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            old = h[i]
            h[i] = old + step
            v1 = _value_fast(h)
            h[i] = old - step
            v2 = _value_fast(h)
            h[i] = old
            
            if v1 < best_val:
                h[i] = old + step
                best_val = v1
                improved = True
            elif v2 < best_val:
                h[i] = old - step
                best_val = v2
                improved = True
        
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
        else:
            # Keep step or reduce slightly
            step *= 0.8
    
    # Try a few more structural variations if time permits
    if time.time() < deadline - 10.0:
        # Raised cosine
        h_rc = 0.5 * (1 + np.cos(np.pi * x))
        v_rc = _value_fast(h_rc)
        if v_rc < best_val:
            best_val = v_rc
            best_h = h_rc.copy()
            # Polish
            h = best_h.copy()
            step = 0.05
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    old = h[i]
                    h[i] = old + step
                    v1 = _value_fast(h)
                    h[i] = old - step
                    v2 = _value_fast(h)
                    h[i] = old
                    if v1 < best_val:
                        h[i] = old + step
                        best_val = v1
                        improved = True
                    elif v2 < best_val:
                        h[i] = old - step
                        best_val = v2
                        improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
                else:
                    step *= 0.8
    
    return best_h

# EVOLVE-BLOCK-END
