import time
import numpy as np
from scipy.optimize import dual_annealing

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _autoconv_peak_ratio(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    Metric: N * max_k (sum_i h_i * h_{i+k}) / (sum_i h_i)^2
    For uniform, this equals 1.0.
    """
    h = np.asarray(heights, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Use FFT for fast convolution
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    max_conv = np.max(c)
    return N * max_conv / (total * total)

def _score(h):
    """Objective function for optimization."""
    h = np.maximum(h, 0.0)
    return _autoconv_peak_ratio(h)

def _construct_from_coeffs(coeffs):
    """
    Construct height vector from Fourier cosine coefficients.
    coeffs = [a1, a2, ..., aM]
    h_i = 1 + sum_k a_k cos(2*pi*k*i/N)
    """
    M = len(coeffs)
    i = np.arange(N)
    h = np.ones(N)
    for k in range(1, M + 1):
        h += coeffs[k-1] * np.cos(2 * np.pi * k * i / N)
    return h

def _objective(coeffs):
    """
    Objective for scipy optimizer.
    Returns the peak ratio, or a large penalty if h has negative values.
    """
    h = _construct_from_coeffs(coeffs)
    if np.min(h) < 0:
        # Penalty for non-negativity violation
        # Scale penalty to be larger than typical objective values
        return 10.0 + np.sum(np.minimum(h, 0)**2)
    return _autoconv_peak_ratio(h)

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _autoconv_peak_ratio(best_h)
    
    # Structural Change: Optimize Fourier coefficients
    # We try a few different numbers of harmonics M
    # and use dual_annealing for global search
    
    # Define bounds for coefficients
    # h_i >= 0 implies sum |a_k| <= 1 roughly, but depends on correlations.
    # Let's be conservative: |a_k| <= 0.5 / k
    Ms = [10, 20, 40]
    
    for M in Ms:
        if _time_left() < 10.0:
            break
            
        bounds = []
        for k in range(1, M + 1):
            # Heuristic bound
            b = 0.5 / k
            bounds.append((-b, b))
            
        # Initial guess: all zeros (uniform)
        x0 = np.zeros(M)
        
        try:
            # dual_annealing is a global optimizer
            # maxiter and maxfev control the budget
            # We have ~10-20 seconds per M if we are lucky
            res = dual_annealing(
                _objective,
                bounds,
                x0=x0,
                maxiter=50,
                maxfev=2000,
                seed=42,
                restarts=5,
                # Time limit is handled by our deadline check in the wrapper if we could, 
                # but dual_annealing doesn't accept a time limit directly.
                # We rely on maxfev and maxiter to keep it fast.
                # 2000 evals * ~1ms (FFT) = ~2s. This is safe.
            )
            
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_h = _construct_from_coeffs(res.x)
                
        except Exception as e:
            pass
    
    # Refine the best Fourier solution with gradient descent on the full h vector?
    # Or just return the Fourier result.
    # Let's do a quick local refinement on the full h vector starting from the Fourier result
    if _time_left() > 20.0:
        # Use a simple gradient descent step on the full vector to polish
        # This reuses the logic from the original program but starts from a better point
        h0 = best_h.copy()
        # We can't easily import the old _gradient_descent if we replaced it.
        # Let's implement a simple local search.
        
        # Actually, let's just stick to the Fourier result if it's good.
        # Or use a few steps of Nelder-Mead on the coefficients again?
        # The Fourier parameterization is low-dimensional, so local methods are fast.
        
        # Let's try a second pass with more harmonics if time permits
        if _time_left() > 15.0:
            M2 = 60
            bounds2 = []
            for k in range(1, M2 + 1):
                b = 0.5 / k
                bounds2.append((-b, b))
            
            x0 = np.zeros(M2)
            # If we have a good solution from M=40, we can project it?
            # No, just start from 0 or use the previous best if it fits.
            # Previous best had M coeffs. We can pad with zeros.
            if len(res.x) < M2:
                x0 = np.zeros(M2)
                x0[:len(res.x)] = res.x
            else:
                x0 = res.x[:M2]
                
            try:
                res2 = dual_annealing(
                    _objective,
                    bounds2,
                    x0=x0,
                    maxiter=50,
                    maxfev=2000,
                    seed=42,
                    restarts=5
                )
                if res2.success and res2.fun < best_score:
                    best_score = res2.fun
                    best_h = _construct_from_coeffs(res2.x)
            except Exception:
                pass

    # Safety: never return worse than uniform
    uniform_score = _autoconv_peak_ratio(np.ones(N))
    if best_score > uniform_score + 1e-9:
        best_h = np.ones(N)
        best_score = uniform_score
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0.0)
    # Normalize to ensure sum is N? 
    # The metric is scale invariant: N * max(c) / sum(h)^2.
    # If we scale h by k, c scales by k^2, sum(h) scales by k, so ratio is invariant.
    # So we don't strictly need to normalize, but it's good practice.
    s = best_h.sum()
    if s > 0:
        best_h = best_h * (N / s)
        
    return [float(v) for v in best_h]

_start_time = 0.0
