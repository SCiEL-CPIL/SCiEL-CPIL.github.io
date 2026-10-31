"""Program using low-rank Gaussian mixture basis with differential evolution optimization."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

N = 600

# Precompute grid points for the step function on [-1/4, 1/4]
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

# Precompute fine grid for convolution to speed up evaluation
_n_fine_global = 4096
_t_fine_global = np.linspace(-0.5, 0.5, _n_fine_global)
_dt_global = _t_fine_global[1] - _t_fine_global[0]
_idx_global = np.floor((_t_fine_global + 0.25) / _dx).astype(int)
_idx_global = np.clip(_idx_global, 0, N - 1)
_in_support_global = (_t_fine_global >= -0.25) & (_t_fine_global < 0.25)

def _gaussian_mixture_heights(params, n_basis=6):
    """
    params: array of length 3 * n_basis
            [c1, w1, a1, c2, w2, a2, ...]
    Returns: array of N nonnegative step heights.
    """
    n = n_basis
    heights = np.zeros(N)
    for i in range(n):
        c = params[3 * i]
        w = params[3 * i + 1]
        a = params[3 * i + 2]
        # Gaussian: a * exp(-(x - c)^2 / (2 * w^2))
        heights += a * np.exp(-(( _centers - c) ** 2) / (2.0 * w ** 2))
    return np.maximum(heights, 0.0)

def _compute_score(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(f*f) / ||f||_2^2
    """
    # Compute L2 norm squared
    l2_sq = np.sum(heights ** 2) * _dx
    
    if l2_sq <= 1e-15:
        return 1e10  # Avoid division by zero
    
    # Evaluate the step function on the precomputed fine grid
    f_fine = np.zeros(_n_fine_global)
    f_fine[_in_support_global] = heights[_idx_global[_in_support_global]]
    
    # Compute autoconvolution on the fine grid
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * _dt_global
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def construct():
    n_basis = 6
    # Parameter bounds:
    # Centers c_i in [-0.24, 0.24]
    # Widths w_i in [0.005, 0.05] (narrow to separate peaks)
    # Amplitudes a_i in [0.1, 1.0] (positive, similar scale)
    
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.24, 0.24))  # center
        bounds.append((0.005, 0.05))  # width
        bounds.append((0.1, 1.0))     # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    # Use differential evolution with tuned parameters
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,  # Reduced iterations to fit in time with finer grid
        tol=1e-6,
        popsize=10,   # Reduced population
        init='sobol',
        polish=True,
        updating='immediate',
        workers=1
    )
    
    best_params = result.x
    best_heights = _gaussian_mixture_heights(best_params, n_basis)
    
    # Ensure nonnegative
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
