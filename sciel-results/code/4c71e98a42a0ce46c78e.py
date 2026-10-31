"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _fft_max_overlap(h):
    """Fast computation of max overlap using FFT.
    
    overlap(s) = sum_i h[i] * (1 - h[(i-s) mod N])
               = sum_i h[i] - sum_i h[i]*h[(i-s) mod N]
               = sum_h - autocorrelation at shift s
    """
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    sum_h = np.sum(h)
    # autocorr[k] = sum_i h[i] * h[(i+k) mod N]
    # We want C[s] = sum_i h[i] * h[(i-s) mod N] = autocorr[(-s) mod N]
    overlaps = sum_h - autocorr[np.mod(-np.arange(N), N)]
    return np.max(overlaps)

def _project_to_valid(h):
    """Project h to satisfy: sum = N/2, each element in [0, 1]."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    diff = target - np.sum(h)
    if abs(diff) < 1e-14:
        return h
    # Distribute diff among elements not at bounds
    mask = (h > 1e-10) & (h < 1.0 - 1e-10)
    if np.sum(mask) > 0:
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    # If still off, adjust any element
    diff2 = target - np.sum(h)
    if abs(diff2) > 1e-12:
        # Find elements with room
        if diff2 > 0:
            for i in range(N):
                room = 1.0 - h[i]
                if room > 1e-12:
                    add = min(diff2, room)
                    h[i] += add
                    diff2 -= add
                    if abs(diff2) < 1e-12:
                        break
        else:
            for i in range(N):
                if h[i] > 1e-12:
                    sub = min(-diff2, h[i])
                    h[i] -= sub
                    diff2 += sub
                    if abs(diff2) < 1e-12:
                        break
    return h

def _soft_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    sum_h = np.sum(h)
    overlaps = sum_h - autocorr[np.mod(-np.arange(N), N)]
    # Soft max
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _eval_periodic(pattern, period):
    """Evaluate a periodic pattern tiled to length N."""
    n_reps = N // period
    h = np.tile(pattern, n_reps)
    h = _project_to_valid(h)
    return _fft_max_overlap(h), h

def construct():
    deadline = time.time() + 110.0
    
    best_h = np.full(N, 0.5)
    best_val = _fft_max_overlap(best_h)
    
    # Phase 1: Try periodic patterns (lower dimensional optimization)
    for period in [2, 4, 5, 10, 20, 25, 50]:
        if time.time() > deadline - 60:
            break
        if N % period != 0:
            continue
        n_reps = N // period
        
        # Start from uniform and a few structured inits
        inits = [np.full(period, 0.5)]
        # Alternating high/low
        alt = np.zeros(period)
        alt[::2] = 0.75
        alt[1::2] = 0.25
        inits.append(alt)
        # Cosine wave
        for k in [1, 2]:
            wave = 0.5 + 0.3 * np.cos(2 * np.pi * k * np.arange(period) / period)
            inits.append(wave)
        
        for x0 in inits:
            if time.time() > deadline - 50:
                break
            
            def objective(x):
                h = np.tile(x, n_reps)
                h = _project_to_valid(h)
                return _soft_max_overlap(h, beta=50.0)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                h_cand = _project_to_valid(np.tile(res.x, n_reps))
                val = _fft_max_overlap(h_cand)
                if val < best_val:
                    best_val = val
                    best_h = h_cand.copy()
            except Exception:
                pass
    
    # Phase 2: Fourier-based constructions with optimization
    # h[i] = 0.5 + sum_k a_k * cos(2*pi*k*i/N + phi_k)
    def _fourier_h(params):
        """params: [a1, phi1, a2, phi2, ..., ak, phik]"""
        h = np.full(N, 0.5)
        for k in range(len(params) // 2):
            a = params[2*k]
            phi = params[2*k + 1]
            h += a * np.cos(2 * np.pi * (k+1) * np.arange(N) / N + phi)
        return _project_to_valid(h)
    
    # Optimize 2-harmonic Fourier params
    if time.time() < deadline - 30:
        n_harmonics = 2
        n_params = 2 * n_harmonics
        x0 = np.zeros(n_params)
        
        def obj_fourier(x):
            h = _fourier_h(x)
            return _soft_max_overlap(h, beta=40.0)
        
        bounds = [(-0.45, 0.45), (0, 2*np.pi)] * n_harmonics
        try:
            res = minimize(obj_fourier, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
            h_cand = _fourier_h(res.x)
            val = _fft_max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass
        
        # Try 3 harmonics
        n_harmonics = 3
        n_params = 2 * n_harmonics
        x0 = np.zeros(n_params)
        
        def obj_fourier3(x):
            h = _fourier_h(x)
            return _soft_max_overlap(h, beta=40.0)
        
        bounds = [(-0.3, 0.3), (0, 2*np.pi)] * n_harmonics
        try:
            res = minimize(obj_fourier3, x0, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9})
            h_cand = _fourier_h(res.x)
            val = _fft_max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass
    
    # Phase 3: Local optimization on full h with soft-max surrogate
    if time.time() < deadline - 15:
        def obj_full(x):
            h = _project_to_valid(x)
            return _soft_max_overlap(h, beta=60.0)
        
        try:
            res = minimize(obj_full, best_h, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11})
            h_cand = _project_to_valid(res.x)
            val = _fft_max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        def obj_polish(x):
            h = _project_to_valid(x)
            return _soft_max_overlap(h, beta=100.0)
        
        try:
            res = minimize(obj_polish, best_h, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            h_cand = _project_to_valid(res.x)
            val = _fft_max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    best_h = _project_to_valid(best_h)
    return best_h
# EVOLVE-BLOCK-END
