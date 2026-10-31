"""Program using low-rank Gaussian mixture basis with differential evolution optimization."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

N = 600

# Precompute grid points for the step function on [-1/4, 1/4]
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

# Precompute fine grid details for fast evaluation
N_FINE = 16384
_t_fine = np.linspace(-0.5, 0.5, N_FINE)
_dt = _t_fine[1] - _t_fine[0]
# Precompute mapping from fine grid indices to coarse step indices
# x_fine is in [-0.5, 0.5]. Steps are in [-0.25, 0.25].
_idx_map = np.floor((_t_fine + 0.25) / _dx).astype(int)
_idx_map = np.clip(_idx_map, 0, N - 1)
_in_support = (_t_fine >= -0.25) & (_t_fine < 0.25)
# We only need to fill the part of f_fine that is in support, but fftconvolve needs the whole array.
# However, we can create f_fine by scattering heights into the pre-allocated array.
# To avoid creating a new zero array every time, we can't easily reuse it if we modify it in place
# because the heights change. But we can create a view or just do the assignment.
# Actually, creating a zeros array of size 16384 is very fast.

def _gaussian_mixture_heights(params, n_basis=8):
    """
    params: array of length 3 * n_basis
            [c1, w1, a1, c2, w2, a2, ...]
    Returns: array of N nonnegative step heights.
    """
    n = n_basis
    heights = np.zeros(N)
    # Vectorized computation for speed
    # Reshape params to (n_basis, 3)
    p = params.reshape(-1, 3)
    c = p[:, 0]
    w = p[:, 1]
    a = p[:, 2]
    
    # Expand centers to match _centers shape for broadcasting
    # _centers is (N,), c is (n_basis,)
    # diff is (N, n_basis)
    diff = _centers[:, np.newaxis] - c[np.newaxis, :]
    # Compute exponent
    # w is (n_basis,), so w**2 is (n_basis,)
    # We need to divide by 2*w^2 for each basis
    exp_term = np.exp(-diff**2 / (2.0 * w[np.newaxis, :]**2))
    # Multiply by amplitudes
    # a is (n_basis,), exp_term is (N, n_basis)
    heights = np.sum(a[np.newaxis, :] * exp_term, axis=1)
    
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
    
    # Evaluate the step function on the fine grid
    # f_fine is zero everywhere except where _in_support is True
    # For indices where _in_support is True, f_fine[i] = heights[_idx_map[i]]
    f_fine = np.zeros(N_FINE)
    # Use the precomputed mask and index map
    f_fine[_in_support] = heights[_idx_map[_in_support]]
    
    # Compute autoconvolution on the fine grid
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * _dt
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def construct():
    n_basis = 8
    # Parameter bounds:
    # Centers c_i in [-0.25, 0.25]
    # Widths w_i in [0.01, 0.25] # Reduced max width as it's half the support width
    # Amplitudes a_i in [0, 5]
    
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.01, 0.25))   # width
        bounds.append((0.0, 5.0))     # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    # Use differential evolution with tuned parameters
    # Increased basis count to 8, so more dimensions.
    # Keep popsize and maxiter moderate to fit in time.
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=150,
        tol=1e-6,
        popsize=12,
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
