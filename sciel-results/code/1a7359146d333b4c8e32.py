"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Strategy: Represent the step function as a low-rank piecewise-linear spline
with 600 knots. The heights are parameterized by a low-rank decomposition
(rank 2-4) to reduce dimensionality while allowing non-smooth, non-Gaussian
shapes. Optimize using multi-start differential evolution.
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


def _low_rank_spline_to_heights(params, x_mid, rank):
    """Convert low-rank spline parameters to step heights.
    
    params: [U_11, U_12, ..., U_1r, V_11, V_12, ..., V_1r, ...]
    where U is (N x rank) and V is (rank x rank)
    
    Actually, let's use a simpler parameterization:
    h_i = sum_{k=1}^{rank} a_k * phi_k(x_i)
    where phi_k are basis functions (cosines or piecewise linear)
    
    For piecewise linear with rank r, we use r control points and interpolate.
    """
    # We'll use a cosine basis for flexibility
    # h(x) = sum_{k=0}^{rank-1} a_k * cos(2*pi*k*x/0.5)
    # But we also want to allow asymmetric shapes, so add sine terms too
    
    # Let's use: h(x) = a_0 + sum_{k=1}^{rank-1} (a_k * cos(2*pi*k*x/0.5) + b_k * sin(2*pi*k*x/0.5))
    # This gives 1 + 2*(rank-1) parameters
    
    num_params = len(params)
    # params[0] = a_0 (DC term)
    # params[1:1+rank] = a_1..a_rank (cosine coefficients)
    # params[1+rank:1+2*rank] = b_1..b_rank (sine coefficients)
    
    a0 = params[0]
    a = params[1:1+rank]
    b = params[1+rank:1+2*rank]
    
    h = np.full_like(x_mid, a0, dtype=float)
    for k in range(rank):
        freq = 2.0 * np.pi * (k + 1) / 0.5
        h += a[k] * np.cos(freq * x_mid)
        if k < len(b):
            h += b[k] * np.sin(freq * x_mid)
    
    return np.clip(h, 0.0, None)


def construct():
    global _t0
    _t0 = time.time()
    
    x_mid = _midpoints()
    
    # Start with uniform as baseline
    uniform = np.ones(N)
    best_h = uniform.copy()
    best_score = _evaluate_step_heights(uniform)
    
    # Try low-rank spline with different ranks
    for rank in [2, 3, 4, 5]:
        if _time_left() < 15.0:
            break
        
        # Number of parameters: 1 (DC) + 2*rank (cos + sin for each frequency)
        num_params = 1 + 2 * rank
        
        # Bounds: DC term in [0.5, 1.5], coefficients in [-0.5, 0.5]
        bounds = [(0.5, 1.5)] + [(-0.5, 0.5)] * (2 * rank)
        
        def objective(params):
            if _time_left() < 3.0:
                return float('inf')
            h = _low_rank_spline_to_heights(params, x_mid, rank)
            return _evaluate_step_heights(h)
        
        try:
            # Multi-start: try 3 different random seeds
            for seed in [42, 123, 456]:
                if _time_left() < 5.0:
                    break
                
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=30,
                    popsize=12,
                    tol=0.01,
                    seed=seed,
                    polish=False,
                    init='latinhypercube'
                )
                
                if result.fun < best_score:
                    h_candidate = _low_rank_spline_to_heights(result.x, x_mid, rank)
                    score = _evaluate_step_heights(h_candidate)
                    if score < best_score:
                        best_score = score
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Also try a simple parametric family: h(x) = 1 + eps*cos(2*pi*x/0.5)
    for eps in np.linspace(-0.5, 0.5, 21):
        if _time_left() < 3.0:
            break
        h = 1.0 + eps * np.cos(2 * np.pi * x_mid / 0.5)
        h = np.clip(h, 0.0, None)
        score = _evaluate_step_heights(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final polish: use scipy.optimize.minimize on the best parameters
    if _time_left() > 8.0:
        # Determine which rank gave the best result
        best_rank = None
        best_params = None
        
        for rank in [2, 3, 4, 5]:
            if _time_left() < 3.0:
                break
            
            # We need to recover the parameters that gave best_h
            # For now, just do a local optimization around the current best
            # by trying different initializations
            
            num_params = 1 + 2 * rank
            bounds = [(0.5, 1.5)] + [(-0.5, 0.5)] * (2 * rank)
            
            def objective_polish(params):
                if _time_left() < 2.0:
                    return float('inf')
                h = _low_rank_spline_to_heights(params, x_mid, rank)
                return _evaluate_step_heights(h)
            
            # Try a few random starts
            for trial in range(2):
                if _time_left() < 2.0:
                    break
                x0_trial = np.random.uniform(
                    low=[b[0] for b in bounds],
                    high=[b[1] for b in bounds],
                    size=num_params
                )
                try:
                    res = minimize(
                        objective_polish,
                        x0_trial,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.fun < best_score:
                        h_candidate = _low_rank_spline_to_heights(res.x, x_mid, rank)
                        score = _evaluate_step_heights(h_candidate)
                        if score < best_score:
                            best_score = score
                            best_h = h_candidate.copy()
                except Exception:
                    pass
    
    # Ensure nonnegative
    best_h = np.clip(best_h, 0.0, None)
    
    return [float(v) for v in best_h]


_t0 = time.time()
