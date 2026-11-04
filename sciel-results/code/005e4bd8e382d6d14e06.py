"""Optimized program for first autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 600
# Interval width for each step
dx = (0.25 - (-0.25)) / N  # 0.5 / 600

def _compute_score(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    heights: array of N nonnegative step heights on [-1/4, 1/4].
    
    The autoconvolution (discrete) is computed via FFT.
    We need to normalize appropriately.
    
    The "normalized maximum" typically refers to:
    max_k (autoconv[k]) / (integral of f)^2 * (some factor related to dx)
    
    Actually, let's think about the continuous case:
    If f(x) is a step function on [-1/4, 1/4] with 600 intervals of width dx,
    then f(x) = h_i for x in [x_i, x_{i+1}).
    
    The autoconvolution is (f * f)(t) = ∫ f(x) f(t-x) dx.
    
    The discrete approximation:
    Let's define the continuous function values. The autoconvolution at time t
    can be approximated by summing over the intervals.
    
    A common normalization for this type of problem (upper bound on C1) is:
    C1 <= max_t (f*f)(t) / (∫f)^2 * (range of support or similar)
    
    Actually, for the first autocorrelation inequality, the bound is often:
    C1 <= sup_x (f*f)(x) / (∫ f)^2
    
    Let's compute:
    1. integral = sum(h_i) * dx
    2. autoconv = fftconvolve(h, h) * dx  (since each bin has width dx)
       Wait, the convolution of two step functions:
       (f*f)(t) ≈ sum_i sum_j h_i h_j dx^2 if the intervals overlap in a certain way.
       
       Actually, let's use the standard discrete convolution approach:
       Let g = h (the heights). The autoconvolution of the step function
       can be computed as the convolution of the height array with itself,
       then scaled by dx (because the continuous convolution integral
       ∫ f(x)f(t-x)dx, when f is piecewise constant with width dx,
       the contribution from each pair of bins is h_i h_j dx^2, but the
       number of overlapping bins depends on the shift).
       
       Let's think more carefully. If we have a step function f(x) = h_i for
       x in [i*dx, (i+1)*dx), i=0..N-1 (after shifting to start at 0), then:
       (f*f)(t) = ∫ f(x) f(t-x) dx
       
       For a given shift k (in units of dx), the overlap is:
       (f*f)(k*dx) ≈ sum_i h_i h_{i-k} dx  (with appropriate bounds)
       
       Actually, the standard way: if we define the discrete signal h[0..N-1],
       then the continuous autoconvolution sampled at multiples of dx is:
       (f*f)(k*dx) = dx * sum_i h[i] h[i-k]  (for k where indices are valid)
       
       Wait, let's verify. If f(x) = h_i for x in [i dx, (i+1) dx), then:
       (f*f)(t) = ∫_{-∞}^{∞} f(x) f(t-x) dx
       
       For t = k dx, the integral becomes a sum over x where both f(x) and
       f(t-x) are nonzero. This is equivalent to the discrete convolution
       of h with itself, multiplied by dx.
       
       So: autoconv[k] = dx * (h * h)[k] where (h*h) is the discrete linear convolution.
       
       The normalization is: max(autoconv) / (integral)^2
       where integral = dx * sum(h).
       
       So the score = max(dx * conv) / (dx * sum(h))^2
                    = max(conv) / (dx * (sum(h))^2)
    """
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < 0):
        return np.inf
    
    # Compute discrete autoconvolution
    conv = fftconvolve(h, h, mode='full')
    
    # Integral of f
    integral = np.sum(h) * dx
    
    if integral <= 0:
        return np.inf
    
    # Normalized max
    score = np.max(conv) / (dx * integral * integral)
    
    return score

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    # Start with a reasonable initial guess
    # A tent function might be a good starting point
    x = np.linspace(-0.25, 0.25, N)
    # Initial: a triangular pulse
    initial = np.maximum(0, 1 - 2 * np.abs(x))
    # Normalize so that integral is 1
    initial = initial / (np.sum(initial) * dx)
    
    best_score = _compute_score(initial)
    best_heights = initial.copy()
    
    # Try differential evolution on a reduced parameterization
    # Use Fourier coefficients for a smoother representation
    # But we need to ensure nonnegativity. Let's use a different approach.
    
    # Strategy: Optimize directly in the height space with constraints,
    # but that's 600 dimensions which is too high for DE.
    
    # Alternative: Use a parametric family with more parameters than 2.
    # Let's use a sum of a few Gaussian-like bumps or a polynomial.
    
    # Actually, let's try optimizing with scipy.optimize.minimize
    # using a lower-dimensional parameterization.
    
    # Use a Fourier series representation with nonnegativity constraint
    # h(x) = a0 + sum_{k=1}^{K} (a_k cos(2*pi*k*x/0.5) + b_k sin(2*pi*k*x/0.5))
    # But ensuring nonnegativity is tricky.
    
    # Instead, let's use a simpler approach: optimize a few key parameters
    # of a piecewise linear function.
    
    # Let's try a different strategy: use Nelder-Mead or L-BFGS-B on
    # a reduced set of parameters.
    
    # For now, let's try a grid search over simple parametric families
    # and then refine with local optimization.
    
    # Family 1: Tent functions with variable width and height
    # h(x) = H * max(0, 1 - |x|/W) for |x| <= W, 0 otherwise
    
    best_tent_score = np.inf
    best_tent_heights = None
    
    # Grid search over width W (in units of dx) and height H
    # W can range from 1 to N/2 intervals
    # H is determined by normalization if we fix integral, but we don't need to fix integral
    
    # Actually, the score is scale-invariant! 
    # If we scale h by c, then conv scales by c^2 and integral scales by c,
    # so score = max(conv) / (dx * integral^2) is unchanged.
    # So we can fix the integral to 1 and just optimize the shape.
    
    # Let's fix integral = 1 and optimize the shape.
    
    def score_fixed_integral(heights):
        """Score with integral normalized to 1."""
        h = np.asarray(heights, dtype=np.float64)
        if np.any(h < 0):
            return np.inf
        integral = np.sum(h) * dx
        if integral <= 0:
            return np.inf
        # Normalize to integral = 1
        h_norm = h / integral
        conv = fftconvolve(h_norm, h_norm, mode='full')
        # Now score = max(conv) / dx
        return np.max(conv) / dx
    
    # For tent functions:
    # h(x) = H * (1 - |x|/W) for |x| <= W
    # integral = H * W (area of triangle)
    # To fix integral = 1: H = 1/W
    # So h(x) = (1/W) * (1 - |x|/W) for |x| <= W
    # This is determined entirely by W.
    
    # Let's grid search over W
    W_values = np.linspace(0.01, 0.25, 100)  # W in units of x (from 0 to 0.25)
    
    for W in W_values:
        if time.time() > deadline:
            break
        
        # Create tent function
        h = np.zeros(N)
        for i in range(N):
            x_i = -0.25 + (i + 0.5) * dx  # center of interval i
            if abs(x_i) <= W:
                h[i] = (1.0 / W) * (1.0 - abs(x_i) / W)
        
        s = score_fixed_integral(h)
        if s < best_tent_score:
            best_tent_score = s
            best_tent_heights = h.copy()
    
    if best_tent_heights is not None:
        if best_tent_score < best_score:
            best_score = best_tent_score
            best_heights = best_tent_heights
    
    # Now try to refine with local optimization on the full height vector
    # Use L-BFGS-B with nonnegativity constraints
    
    if time.time() < deadline - 10:
        # Convert to optimization problem
        # We want to minimize score_fixed_integral(h) subject to h >= 0
        
        # Use a log-barrier or just bound constraints
        bounds = [(0, None)] * N
        
        def objective(h):
            return score_fixed_integral(h)
        
        # Start from the best tent
        x0 = best_heights.copy()
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_heights = result.x
        except Exception:
            pass
    
    # Return the best heights found
    return list(best_heights)
