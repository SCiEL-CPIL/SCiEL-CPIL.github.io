"""Program using low-rank Gaussian mixture basis with differential evolution optimization."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

N = 600

# Precompute grid points for the step function on [-1/4, 1/4]
# The step function is constant on each of the N equal intervals.
# We represent it as a sum of Gaussian basis functions sampled at interval centers.
# Interval width: 1/2 / N = 1/1200
# Interval centers: -1/4 + (i + 0.5)/N * (1/2) for i in 0..N-1
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

def _gaussian_mixture_heights(params, n_basis=8):
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
    The step function f is defined on [-1/4, 1/4] with N equal intervals.
    The autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    For a step function with heights h_i on intervals of width dx,
    the discrete approximation of the convolution is:
    conv[k] = sum_i h_i * h_{k-i} * dx^2  (approximately)
    
    But we need to be careful about the normalization.
    
    The task says "minimize the normalized maximum of its autoconvolution".
    The seed program returns [1.0]*N. Let's think about what the score is for the seed.
    
    For f(x) = 1 on [-1/4, 1/4], the autoconvolution is:
    (f*f)(t) = length of intersection of [-1/4, 1/4] and [t-1/4, t+1/4]
    This is a triangular function with peak 1/2 at t=0, support [-1/2, 1/2].
    
    The "normalized maximum" likely means: max(f*f) / (something)
    
    Given the context of C1 (first autocorrelation inequality), and that the seed scores 1.0,
    let's assume the score is max(f*f) where f*f is computed in a normalized way.
    
    Let's compute it numerically:
    - f is a step function with heights h_i on intervals of width dx.
    - The continuous autoconvolution can be approximated by:
      (f*f)(t) ≈ sum_{i,j} h_i * h_j * dx^2 * [indicator that t is in the support]
    
    Actually, for a step function, the autoconvolution at a point t is:
    integral f(s) f(t-s) ds
    
    Let's discretize: f(s) ≈ h_i for s in interval i.
    Then (f*f)(t) ≈ sum_i h_i * (integral over s in interval i of f(t-s) ds)
    
    This is complex. Let's use FFT-based convolution on the discrete heights.
    
    The discrete convolution of the heights array with itself, scaled appropriately,
    approximates the continuous autoconvolution.
    
    For the seed [1.0]*N:
    The discrete convolution is a triangle of length 2N-1 with max value N.
    The continuous autoconvolution max is 1/2 (as computed above for f=1 on [-1/4,1/4]).
    
    The ratio: (1/2) / N * something? 
    
    Let's think differently. The "normalized" part likely means we normalize the step function
    so that its L1 norm is fixed, or its L2 norm is fixed.
    
    Given that the task is about an upper bound on C1, and the seed is the constant function,
    let's assume the score is:
    max(f*f) / (||f||_1^2) or max(f*f) / (||f||_2^2) or similar.
    
    For f=1 on [-1/4,1/4]:
    ||f||_1 = 1/2
    ||f||_1^2 = 1/4
    max(f*f) = 1/2
    So max(f*f) / ||f||_1^2 = (1/2) / (1/4) = 2.
    
    But the seed scores 1.0. So maybe it's max(f*f) / ||f||_2^2?
    ||f||_2^2 = integral f^2 = 1/2
    max(f*f) / ||f||_2^2 = (1/2) / (1/2) = 1.0. Yes!
    
    So the score is max(f*f) / ||f||_2^2 where ||f||_2^2 = integral f(x)^2 dx.
    
    For a step function with heights h_i on intervals of width dx:
    ||f||_2^2 ≈ sum_i h_i^2 * dx
    max(f*f) needs to be computed.
    
    The autoconvolution (f*f)(t) = integral f(s) f(t-s) ds.
    Using the step function approximation:
    (f*f)(t) ≈ sum_i h_i * h_j * dx^2 where the intervals i and j are such that...
    
    Actually, let's just compute it numerically using the discrete heights.
    The continuous autoconvolution can be approximated by:
    conv(t) = sum_i h_i * (integral over s in interval i of f(t-s) ds)
    
    A simpler approach: resample the step function on a fine grid, compute the convolution,
    and find the max. Then divide by the L2 norm squared.
    """
    # Compute L2 norm squared
    l2_sq = np.sum(heights ** 2) * _dx
    
    if l2_sq <= 1e-15:
        return 1e10  # Avoid division by zero
    
    # Compute autoconvolution using FFT
    # The step function is defined on [-1/4, 1/4]. The autoconvolution is on [-1/2, 1/2].
    # We'll compute the discrete convolution of the heights array with itself.
    # The result will have 2N-1 elements, corresponding to the convolution of two sequences of length N.
    
    conv = fftconvolve(heights, heights, mode='full')
    # The convolution values are in units of h_i * h_j (no dx factor yet).
    # The continuous autoconvolution at a point is approximately:
    # sum_{i,j} h_i * h_j * dx^2 * [something]
    
    # Actually, for a step function with height h_i on interval [x_i, x_i+dx],
    # the autoconvolution at t is:
    # integral f(s) f(t-s) ds = sum_i h_i * (measure of s in interval i such that t-s is in the support)
    
    # Let's use a more direct numerical approach:
    # Create a fine grid on [-1/2, 1/2] and evaluate the step function.
    n_fine = 4096
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    dt = t_fine[1] - t_fine[0]
    
    # Evaluate the step function on the fine grid
    # The step function is defined on [-1/4, 1/4]. Outside this range, f=0.
    # For each point t in t_fine, f(t) = h_i where i is the interval containing t, if t in [-1/4, 1/4], else 0.
    # Interval i covers [-1/4 + i*dx, -1/4 + (i+1)*dx)
    
    # Map fine grid points to interval indices
    # x in [-1/4, 1/4] -> index = floor((x + 1/4) / dx)
    # Clip to [0, N-1]
    
    x_fine = t_fine  # The fine grid is on [-1/2, 1/2], but f is only nonzero on [-1/4, 1/4]
    idx = np.floor((x_fine + 0.25) / _dx).astype(int)
    idx = np.clip(idx, 0, N - 1)
    
    # Only points in [-1/4, 1/4] have nonzero f
    in_support = (x_fine >= -0.25) & (x_fine < 0.25)
    f_fine = np.zeros(n_fine)
    f_fine[in_support] = heights[idx[in_support]]
    
    # Compute autoconvolution on the fine grid
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * dt
    # The full convolution has length 2*n_fine - 1.
    # The center corresponds to t=0.
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def construct():
    n_basis = 6
    # Parameter bounds:
    # Centers c_i in [-0.25, 0.25]
    # Widths w_i in [0.01, 0.5] (log scale might be better, but let's try linear first)
    # Amplitudes a_i in [0, 5]
    
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.01, 0.5))     # width
        bounds.append((0.0, 5.0))      # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        popsize=10,
        init='sobol',
        polish=False,
        updating='immediate',
        workers=1
    )
    
    best_params = result.x
    best_heights = _gaussian_mixture_heights(best_params, n_basis)
    
    # Ensure nonnegative
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
