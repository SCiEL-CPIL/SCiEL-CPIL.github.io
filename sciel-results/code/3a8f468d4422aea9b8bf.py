"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = np.array(h, dtype=float)
    for _ in range(20):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-15:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _h_from_coeffs(c):
    """
    Reconstruct h from Fourier coefficients.
    c[0] is the DC offset (mean).
    c[1] is cos coeff for k=1, c[2] is sin coeff for k=1, etc.
    h[i] = c[0] + sum_{k=1}^{K} (c[2k-1] cos(2pi k i/N) + c[2k] sin(2pi k i/N))
    """
    i = np.arange(N)
    h = np.full(N, c[0])
    for k in range(1, len(c)//2 + 1):
        h += c[2*k-1] * np.cos(2 * np.pi * k * i / N)
        h += c[2*k] * np.sin(2 * np.pi * k * i / N)
    return h

def _optimize_sinusoidal(h_init, K, deadline):
    """
    Optimize low-dimensional Fourier coefficients.
    """
    # Initialize coefficients from h_init
    i = np.arange(N)
    c = np.zeros(2*K + 1)
    c[0] = np.mean(h_init)
    for k in range(1, K+1):
        c[2*k-1] = (2.0/N) * np.sum(h_init * np.cos(2 * np.pi * k * i / N))
        c[2*k] = (2.0/N) * np.sum(h_init * np.sin(2 * np.pi * k * i / N))
    
    def _obj(c):
        if time.time() > deadline:
            return 1.0
        h = _h_from_coeffs(c)
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=100.0)
        # Penalty for sum constraint
        sum_pen = 100.0 * (np.sum(h) - N/2.0)**2
        # Penalty for bounds
        bound_pen = 100.0 * (np.sum(np.clip(h, 0, 1) - h)**2)
        return smax + sum_pen + bound_pen

    # Bounds for coefficients to keep h in [0,1] roughly
    # |c[0]| <= 1, |c[2k]| <= 1 is a loose bound
    bounds = [(0.0, 1.0)] + [(-1.0, 1.0)] * (2*K)
    
    try:
        res = minimize(_obj, c, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
        h_opt = _h_from_coeffs(res.x)
        h_opt = _normalize(h_opt)
        return h_opt, _score(h_opt)
    except Exception:
        return h_init, _score(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # 1. Generate diverse initial guesses
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    
    candidates = [h0]
    
    # Add random smooth candidates
    np.random.seed(42)
    for _ in range(3):
        h_rand = 0.5 + 0.2 * np.sin(np.random.uniform(0, 2*np.pi) + np.arange(N)*0.1)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = _normalize(h_rand)
        candidates.append(h_rand)

    best_h = h0.copy()
    best_score = _score(h0)

    # 2. Structural Change: Optimize low-dimensional Fourier coefficients directly
    for K in [5, 10, 15]:
        if time.time() < deadline - 30.0:
            try:
                h_opt, s_opt = _optimize_sinusoidal(best_h, K, deadline)
                if s_opt < best_score:
                    best_h = h_opt
                    best_score = s_opt
            except Exception:
                pass

    # 3. Final Local Polish on the best candidate found
    if time.time() < deadline - 5.0:
        try:
            # Use SLSQP on the full vector for final refinement
            def _final_obj(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return _smooth_max(overlap, alpha=200.0)
            
            res = minimize(_final_obj, best_h, method='SLSQP', bounds=[(0,1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
            h_final = _normalize(res.x)
            s_final = _score(h_final)
            if s_final < best_score:
                best_h = h_final
                best_score = s_final
        except Exception:
            pass

    return best_h
