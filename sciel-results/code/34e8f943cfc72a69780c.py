"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy: Represent the step function as a mixture of Gaussian basis functions
with variable centers, widths, and weights. Optimize these parameters using
differential evolution to exploit smooth continuous structure.
Added: FFT-based autoconvolution for speed, and a final polish pass with
tighter tolerance and more iterations to squeeze out additional improvement.
Improvements: More Gaussian components, smooth max surrogate (log-sum-exp)
for differentiable optimization, and direct per-interval refinement.
"""

import time
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 600
DELTA = 0.5 / N
DEADLINE = 145.0  # safety margin before 150s kill

_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _midpoints():
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    return x


def _autoconv_fft(h):
    """Compute full autoconvolution via FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    H = np.fft.rfft(h, size)
    C = np.fft.irfft(H * H, size)
    return C[:2 * n - 1]


def _evaluate_step_heights(h):
    """Compute the evaluator's score: max(autoconv) / (sum h)^2."""
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = float(np.sum(h))
    if s < 1e-12:
        return float('inf')
    
    # Compute autoconvolution via FFT
    c = _autoconv_fft(h)
    peak = float(np.max(c))
    
    # L1-based normalization: peak / (sum h)^2
    return peak / (s * s)


def _smooth_max_surrogate(h, temperature=0.01):
    """Smooth max of autoconvolution using log-sum-exp for differentiability.
    
    Returns a smooth approximation to max(c) / (sum h)^2.
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = float(np.sum(h))
    if s < 1e-12:
        return float('inf')
    
    c = _autoconv_fft(h)
    
    # Log-sum-exp for smooth max
    c_max = np.max(c)
    if temperature > 0:
        # soft max: exp((c - c_max)/t) normalized
        shifted = (c - c_max) / temperature
        # Clip to avoid overflow
        shifted = np.clip(shifted, -100, 100)
        log_sum_exp = c_max + temperature * np.log(np.sum(np.exp(shifted)))
    else:
        log_sum_exp = c_max
    
    return log_sum_exp / (s * s)


def _gaussian_mixture_to_heights(params, x_mid):
    """Convert Gaussian mixture parameters to step heights.
    
    params: [w1, c1, s1, w2, c2, s2, ..., wK, cK, sK]
    where K is number of Gaussians.
    """
    K = len(params) // 3
    h = np.zeros_like(x_mid)
    for k in range(K):
        w = params[3*k]
        c = params[3*k+1]
        s = params[3*k+2]
        s = max(s, 1e-6)  # avoid zero width
        h += w * np.exp(-0.5 * ((x_mid - c) / s) ** 2)
    return np.clip(h, 0.0, None)


def _refine_heights_direct(h, max_iter=50, tol=1e-8):
    """Direct per-interval refinement: adjust each height slightly to reduce peak.
    
    Uses coordinate descent on individual heights.
    """
    h = np.array(h, dtype=float)
    h = np.clip(h, 0.0, None)
    current_score = _evaluate_step_heights(h)
    
    for iteration in range(max_iter):
        if _time_left() < 5.0:
            break
        improved = False
        for i in range(N):
            # Try small perturbations
            base = h[i]
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_trial = h.copy()
                h_trial[i] = max(0.0, base + delta)
                score = _evaluate_step_heights(h_trial)
                if score < current_score - tol:
                    h[i] = h_trial[i]
                    current_score = score
                    improved = True
                    break
        if not improved:
            break
    
    return h, current_score


def construct():
    global _t0
    _t0 = time.time()
    
    x_mid = _midpoints()
    
    # Start with uniform as baseline
    uniform = np.ones(N)
    best_h = uniform.copy()
    best_score = _evaluate_step_heights(uniform)
    
    # Try Gaussian mixture with K=1, 2, 3, 4, 5 Gaussians
    for K in [1, 2, 3, 4, 5]:
        if _time_left() < 15.0:
            break
        
        # Bounds for each parameter: weight in [0, 2], center in [-0.25, 0.25], sigma in [0.01, 0.25]
        bounds = [(0.0, 2.0), (-0.25, 0.25), (0.01, 0.25)] * K
        
        def objective(params):
            if _time_left() < 5.0:
                return float('inf')
            h = _gaussian_mixture_to_heights(params, x_mid)
            return _evaluate_step_heights(h)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=30,
                popsize=12,
                tol=0.02,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if result.fun < best_score:
                h_candidate = _gaussian_mixture_to_heights(result.x, x_mid)
                score = _evaluate_step_heights(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Also try a simple parametric family: h(x) = 1 + eps*cos(2*pi*x/0.5)
    for eps in np.linspace(-0.5, 0.5, 21):
        if _time_left() < 5.0:
            break
        h = 1.0 + eps * np.cos(2 * np.pi * x_mid / 0.5)
        h = np.clip(h, 0.0, None)
        score = _evaluate_step_heights(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Polish with smooth surrogate using L-BFGS-B
    if _time_left() > 15.0:
        # Try K=3 and K=4 with smooth objective
        for K in [3, 4]:
            if _time_left() < 8.0:
                break
            bounds = [(0.0, 2.0), (-0.25, 0.25), (0.01, 0.25)] * K
            
            def objective_smooth(params):
                if _time_left() < 2.0:
                    return float('inf')
                h = _gaussian_mixture_to_heights(params, x_mid)
                return _smooth_max_surrogate(h, temperature=0.005)
            
            # Try multiple random starts
            for trial in range(5):
                if _time_left() < 3.0:
                    break
                x0_trial = np.random.uniform(
                    low=[b[0] for b in bounds],
                    high=[b[1] for b in bounds],
                    size=len(bounds)
                )
                try:
                    res = minimize(
                        objective_smooth,
                        x0_trial,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.fun < float('inf'):
                        h_candidate = _gaussian_mixture_to_heights(res.x, x_mid)
                        score = _evaluate_step_heights(h_candidate)
                        if score < best_score:
                            best_score = score
                            best_h = h_candidate.copy()
                except Exception:
                    pass
    
    # Final direct refinement pass
    if _time_left() > 10.0:
        h_refined, score_refined = _refine_heights_direct(best_h, max_iter=30, tol=1e-9)
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined
    
    # Ensure nonnegative
    best_h = np.clip(best_h, 0.0, None)
    
    return [float(v) for v in best_h]


_t0 = time.time()
