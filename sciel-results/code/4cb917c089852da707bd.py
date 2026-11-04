"""
Design: Represent 50 heights as squared modulus of a sum of two orthogonal,
disjointly supported windowed basis functions.

h_k = |a * w1(k) + b * w2(k)|^2

where w1, w2 are non-overlapping Gaussian-like pulses separated by a large gap.
Since they are disjointly supported, |a*w1 + b*w2|^2 = a^2 * w1^2 + b^2 * w2^2
(no cross terms). This makes the autoconvolution additive in a useful way.

We optimize: separation distance, widths (sigma1, sigma2), and amplitudes (a, b)
to maximize the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_inf)

Note: The heights are on equal intervals of [-1/4, 1/4], so 50 points.
The step size is (1/2)/50 = 1/100 = 0.01.

We use a global search (random restarts + local refinement) over the parameters.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50

def _compute_score(params):
    """
    params: [center1, center2, sigma1, sigma2, a, b]
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    c1, c2, s1, s2, a, b = params
    
    # Ensure positive sigmas and amplitudes
    s1 = max(abs(s1), 0.005)
    s2 = max(abs(s2), 0.005)
    a = abs(a)
    b = abs(b)
    
    if a < 1e-10 or b < 1e-10:
        return -1.0
    
    # Grid points on [-1/4, 1/4] with 50 points
    x = np.linspace(-0.25, 0.25, N)
    
    # Ensure centers are within or near the domain
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    # Ensure disjoint supports: require |c1 - c2| > (s1 + s2) * 3
    if abs(c1 - c2) < (s1 + s2) * 3.0:
        return -1.0
    
    # Gaussian-like pulses (using exp(-x^2/(2*sigma^2)))
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    # Heights: h_k = |a*w1 + b*w2|^2 = a^2*w1^2 + b^2*w2^2 (disjoint)
    h = a**2 * w1**2 + b**2 * w2**2
    
    # Ensure nonnegativity (should be, but clip for safety)
    h = np.maximum(h, 0.0)
    
    # Check if any height is zero - that's fine, nonnegative allows zeros
    
    # L2 norm squared
    # The L2 norm on the discrete grid: sum of h_k^2 * dx? 
    # Actually, for step functions on equal intervals, the L2 norm squared is sum(h_k^2) * dx
    # But since we're taking a ratio, dx cancels out. Let's use the raw sums.
    # Actually, let's think about this more carefully.
    # The autoconvolution (h * h)(t) = integral h(s) h(t-s) ds
    # For step functions, this is a piecewise constant function.
    # The supremum norm is the max value.
    # L2 norm squared = integral h(x)^2 dx = dx * sum(h_k^2)
    # L1 norm = integral h(x) dx = dx * sum(h_k)
    # Autoconvolution: (h*h)(t) for step functions on grid...
    # Let's compute the autoconvolution properly.
    
    # For discrete step functions with step size dx, the autoconvolution at grid points:
    # (h*h)[i] = sum_j h[j] * h[i-j] * dx (approximately, with proper indexing)
    # Actually, the continuous autoconvolution of step functions:
    # If h(x) = h_k for x in [x_k, x_{k+1}), then
    # (h*h)(t) = integral h(s) h(t-s) ds
    # This is a continuous function. Its supremum can be found by evaluating
    # on a fine grid or by noting it's piecewise quadratic.
    
    # For simplicity and speed, let's compute the autoconvolution on a fine grid.
    dx = 0.01  # step size
    
    # Compute autoconvolution using FFT or direct method
    # Use direct method for accuracy
    # The support of h*h is [-0.5, 0.5]
    # Evaluate on a fine grid
    
    # Actually, let's use the property that for step functions, the autoconvolution
    # can be computed efficiently. Let's use np.convolve with the right scaling.
    
    # h is defined on N points with spacing dx.
    # The autoconvolution (h*h)(t) = sum_k h_k * h_k' * dx where we sum over overlaps.
    # Using discrete convolution: conv(h, h) * dx gives the values at grid points
    # of the autoconvolution, but we need the supremum which may occur between grid points.
    
    # For Gaussian-like pulses, the autoconvolution is also smooth, so evaluating
    # on a fine grid should give a good approximation of the supremum.
    
    # Compute autoconvolution via FFT for efficiency
    h_ext = np.zeros(2 * N)
    h_ext[:N] = h
    
    # Linear convolution
    conv = np.convolve(h_ext, h_ext, mode='full') * dx
    
    # The autoconvolution is a continuous function. The discrete conv gives values
    # at points spaced dx apart, starting from -0.5 to 0.5.
    # The supremum of the continuous function is at most the max of the discrete
    # samples if the function is smooth enough. For Gaussians, this should be fine.
    
    sup_norm = np.max(conv)
    
    if sup_norm <= 0:
        return -1.0
    
    l2_sq = np.sum(h**2) * dx
    l1 = np.sum(h) * dx
    
    if l1 <= 0:
        return -1.0
    
    ratio = l2_sq / (l1 * sup_norm)
    
    return ratio


def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s


def construct():
    """
    Returns a sequence of 50 nonnegative step heights.
    """
    # Global search using differential evolution
    # Parameters: [center1, center2, sigma1, sigma2, a, b]
    
    bounds = [
        (-0.25, 0.25),  # center1
        (-0.25, 0.25),  # center2
        (0.005, 0.15),  # sigma1
        (0.005, 0.15),  # sigma2
        (0.1, 5.0),     # a
        (0.1, 5.0),     # b
    ]
    
    best_params = None
    best_score = -np.inf
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _neg_score, bounds, 
            maxiter=200, popsize=20,
            tol=1e-6, seed=42,
            polish=True
        )
        if result.success or result.fun < 1e10:
            s = -result.fun
            if s > best_score:
                best_score = s
                best_params = result.x
    except Exception:
        pass
    
    # Also try some random restarts
    np.random.seed(123)
    for _ in range(50):
        params = []
        for (lo, hi) in bounds:
            params.append(np.random.uniform(lo, hi))
        s = _compute_score(params)
        if s > best_score:
            best_score = s
            best_params = np.array(params)
    
    # Local refinement around best
    if best_params is not None:
        try:
            result = minimize(
                _neg_score, best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < 1e10:
                s = -result.fun
                if s > best_score:
                    best_score = s
                    best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: uniform
        return [1.0] * N
    
    c1, c2, s1, s2, a, b = best_params
    s1 = max(abs(s1), 0.005)
    s2 = max(abs(s2), 0.005)
    a = abs(a)
    b = abs(b)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    x = np.linspace(-0.25, 0.25, N)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    h = a**2 * w1**2 + b**2 * w2**2
    h = np.maximum(h, 0.0)
    
    # Normalize: the ratio is scale-invariant, so we can scale h arbitrarily.
    # But we need nonnegative heights. Let's keep them as is.
    
    return [float(v) for v in h]
