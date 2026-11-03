"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600

# EVOLVE-BLOCK-START

def _autoconv_max_normalized(heights):
    """
    Compute the normalized maximum of the autoconvolution of the step function.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    F = np.fft.fft(padded)
    G = F * np.conj(F)
    conv = np.fft.ifft(G).real
    max_conv = np.max(conv[:2*n-1])
    total = np.sum(h)
    if total == 0:
        return 0.0
    dx = 1.0 / (2 * n)
    normalized_max = max_conv / (total * total * dx)
    return normalized_max

def _compute_ratio_fast(h):
    """Fast computation of normalized max autoconvolution ratio."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    F = np.fft.fft(padded)
    conv = np.fft.ifft(F * np.conj(F)).real
    max_conv = np.max(conv[:2*n-1])
    total = np.sum(h)
    if total == 0:
        return 0.0
    dx = 1.0 / (2 * n)
    return max_conv / (total * total * dx)

def _parametric_heights(params):
    """
    Generate heights from a parametric smooth model.
    params = [center, sigma, cos_weight, gauss_weight, asymmetry]
    - center: position of peak (0 to N-1)
    - sigma: width parameter
    - cos_weight: weight of raised cosine component
    - gauss_weight: weight of Gaussian component  
    - asymmetry: asymmetric skew parameter (-1 to 1)
    """
    center = params[0]
    sigma = params[1]
    cos_weight = params[2]
    gauss_weight = params[3]
    asymmetry = params[4]
    
    t = np.arange(N, dtype=np.float64)
    t_norm = (t - center) / (N - 1)
    
    # Apply asymmetry by shifting the effective center based on t
    if abs(asymmetry) > 1e-6:
        t_shifted = t - asymmetry * (t - center) * 0.5
    else:
        t_shifted = t
    
    # Raised cosine component
    h_cos = 0.5 * (1.0 + np.cos(np.pi * t_norm))
    
    # Gaussian component
    sigma_eff = max(sigma, 1e-6)
    h_gauss = np.exp(-0.5 * ((t_shifted - center) / sigma_eff) ** 2)
    
    # Blend
    h = cos_weight * h_cos + gauss_weight * h_gauss
    
    # Ensure non-negativity and normalize scale
    h = np.maximum(h, 0.0)
    if np.sum(h) > 0:
        h = h / np.sum(h) * N  # Scale so that average height is 1
    
    return h

def _objective_parametric(params):
    """Objective function for parametric optimization."""
    h = _parametric_heights(params)
    return _compute_ratio_fast(h)

def _objective_direct(heights):
    """Direct objective on heights."""
    h = np.maximum(heights, 0.0)
    return _compute_ratio_fast(h)

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initial best: uniform
    best_heights = np.ones(N, dtype=np.float64)
    best_score = _compute_ratio_fast(best_heights)
    
    t = np.arange(N, dtype=np.float64)
    
    # Try several parametric models
    # Raised cosine
    h_cos = 0.5 * (1.0 + np.cos(np.pi * t / (N - 1)))
    h_cos = np.maximum(h_cos, 0.0)
    if np.sum(h_cos) > 0:
        h_cos = h_cos / np.sum(h_cos) * N
    score_cos = _compute_ratio_fast(h_cos)
    if score_cos < best_score:
        best_score = score_cos
        best_heights = h_cos.copy()
    
    # Gaussian with various sigmas
    center = N / 2.0
    for sigma_frac in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        sigma = sigma_frac * N
        h_gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        h_gauss = np.maximum(h_gauss, 0.0)
        if np.sum(h_gauss) > 0:
            h_gauss = h_gauss / np.sum(h_gauss) * N
        score_gauss = _compute_ratio_fast(h_gauss)
        if score_gauss < best_score:
            best_score = score_gauss
            best_heights = h_gauss.copy()
    
    # Use Differential Evolution for global search in parametric space
    if time.time() < deadline - 30.0:
        try:
            # Bounds: [center, sigma, cos_weight, gauss_weight, asymmetry]
            bounds = [
                (0, N - 1),           # center
                (N * 0.05, N * 0.6),  # sigma
                (0.0, 1.0),           # cos_weight
                (0.0, 1.0),           # gauss_weight
                (-0.5, 0.5)           # asymmetry
            ]
            
            result_de = differential_evolution(
                _objective_parametric,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='sobol'
            )
            
            if result_de.fun < best_score:
                h_opt = _parametric_heights(result_de.x)
                h_opt = np.maximum(h_opt, 0.0)
                score = _compute_ratio_fast(h_opt)
                if score < best_score:
                    best_score = score
                    best_heights = h_opt.copy()
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B on the full height vector
    if time.time() < deadline - 10.0:
        try:
            h0 = best_heights.copy()
            
            def _smooth_objective(h):
                h_pos = np.maximum(h, 0.0)
                score = _compute_ratio_fast(h_pos)
                # Light smoothness penalty
                diff = np.diff(h_pos)
                penalty = 1e-4 * np.sum(diff**2)
                return score + penalty
            
            result_local = minimize(
                _smooth_objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            
            if result_local.fun < best_score + 1e-6:
                h_opt = np.maximum(result_local.x, 0.0)
                score = _compute_ratio_fast(h_opt)
                if score < best_score:
                    best_score = score
                    best_heights = h_opt.copy()
        except Exception:
            pass
    
    return best_heights.tolist()

# EVOLVE-BLOCK-END
