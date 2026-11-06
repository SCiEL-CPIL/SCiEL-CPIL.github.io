import time
import numpy as np

N = 400
_t0 = time.time()
_DEADLINE = 145.0

def _value(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return np.inf
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve_max(h):
    """Compute max of full autocorrelation using FFT."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * np.conj(f), size)[:n]
    return float(np.max(conv))

def _fast_value(h):
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    m = _fft_convolve_max(h)
    return abs(2.0 * N * m / (s * s))

def construct():
    global _t0
    _t0 = time.time()
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Start with constant vector (known optimal by Cauchy-Schwarz)
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    # Try coordinate descent to see if we can improve
    h = np.ones(N).copy()
    
    # Try various initial patterns
    rng = np.random.default_rng(42)
    
    patterns = [
        np.ones(N),
        np.sign(np.cos(2 * np.pi * np.arange(N) / N)),  # Alternating-ish
        np.sin(np.pi * np.arange(N) / N),  # Sine
        np.cos(2 * np.pi * np.arange(N) / N),  # Cosine
        rng.standard_normal(N),  # Random
        np.ones(N) * (1 + 0.1 * rng.standard_normal(N)),  # Perturbed constant
    ]
    
    for pat in patterns:
        if not deadline_ok():
            break
        h = pat.astype(float).copy()
        s = np.sum(h)
        if abs(s) < 1e-10:
            continue
        
        # Coordinate descent
        for iteration in range(2000):
            if not deadline_ok():
                break
            improved = False
            for i in range(N):
                if not deadline_ok():
                    break
                old = h[i]
                # Try a few perturbations
                for delta in [0.01, -0.01, 0.1, -0.1, 0.5, -0.5]:
                    h[i] = old + delta
                    v = _fast_value(h)
                    if v < best_v - 1e-12:
                        best_v = v
                        best_h = h.copy()
                        improved = True
                        break
                h[i] = old
            
            if not improved:
                break
    
    # Final verification with exact convolve
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    v_final = _value(best_h)
    if v_final < 2.0:
        return best_h
    else:
        return np.ones(N)
