"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def _autoconvolution_max(heights):
    """Compute the maximum of the autoconvolution of the step function."""
    # The autoconvolution of a step function with heights h on N equal intervals
    # is given by the convolution of h with itself, scaled by the interval width.
    # For a step function f(x) = h_i on interval i, the autoconvolution is:
    # (f*f)(t) = integral f(x)f(t-x) dx
    # For discrete approximation, this is the convolution of h with itself, 
    # multiplied by the interval width (1/4 - (-1/4)) / N = 0.5/N
    # But actually, the autoconvolution in the continuous sense for a step function
    # is computed as follows: if h[i] is the height on interval i, then the 
    # autoconvolution at a point is the sum over all pairs of intervals that 
    # overlap when shifted.
    
    # A simpler approach: the autoconvolution of a step function is a piecewise 
    # linear function. Its maximum can be found by computing the convolution 
    # of the heights with themselves and taking the appropriate scaling.
    
    # The autoconvolution (f*f)(t) for a step function f with N intervals of width w:
    # The convolution is a piecewise linear function with 2N-1 segments.
    # The values at the breakpoints can be computed from the heights.
    
    # Let's use the fact that the autoconvolution is the convolution of h with h,
    # where h is the sequence of heights, and the result is scaled by w^2 for 
    # the area, but we need the actual function values.
    
    # Actually, for a step function f(x) = h_i for x in [x_i, x_{i+1}), 
    # the autoconvolution (f*f)(t) = integral f(x)f(t-x) dx.
    # This is a piecewise linear function. The maximum occurs at one of the 
    # breakpoints or in the interior of a segment.
    
    # A practical approach: compute the convolution of h with h, which gives 
    # the values at the "center" of each segment of the autoconvolution.
    
    h = np.array(heights, dtype=float)
    # The autoconvolution of a step function with heights h and N intervals of width w
    # The convolution of h with h (linear convolution) gives a sequence of length 2N-1.
    # Each element represents the integral contribution for a specific overlap.
    # The actual autoconvolution function value at the midpoint of each segment 
    # is w * conv[i], where conv = np.convolve(h, h)
    
    w = 0.5 / N  # interval width
    conv = np.convolve(h, h)
    # The autoconvolution is a piecewise linear function. The maximum value 
    # is bounded by the maximum of w * conv[i] plus some adjustment.
    # Actually, for a step function, the autoconvolution is piecewise linear, 
    # and its maximum is at most max(w * conv) + w * max(h)^2 / 4 or something.
    # Let's just use w * max(conv) as an upper bound, or compute more carefully.
    
    # More carefully: the autoconvolution (f*f)(t) for a step function is:
    # For t in [k*w, (k+1)*w) for k = 0, ..., 2N-2:
    # (f*f)(t) = sum_i h_i * h_{i+k} * (w - |t - k*w|) + ... 
    # This is getting complex. Let's use a simpler upper bound.
    
    # The L-infinity norm of the autoconvolution is at most the L1 norm of f squared,
    # but that's too loose. A better bound: the autoconvolution is piecewise linear,
    # and its maximum is at most max(w * conv[i]) where conv = np.convolve(h, h).
    
    # Actually, I think the correct interpretation is:
    # The autoconvolution (f*f)(t) = integral f(x) f(t-x) dx
    # For a step function, this is a piecewise linear function.
    # The maximum value can be computed exactly.
    
    # Let me use a different approach: the peak of the autoconvolution is at most
    # the maximum of the "discrete autoconvolution" times w, where the discrete 
    # autoconvolution is conv[i] = sum_j h[j] h[j+i] (with appropriate bounds).
    
    # I'll compute the exact piecewise linear autoconvolution and find its maximum.
    
    # The autoconvolution of a step function with N intervals of width w:
    # The support is [-N*w, N*w] = [-0.5, 0.5]
    # It's a piecewise linear function with breakpoints at k*w for k = -N, ..., N
    
    # The value at breakpoint k*w (for k = -N+1, ..., N-1) is:
    # (f*f)(k*w) = w * sum_{j} h[j] h[j+k] (with h[j] = 0 for j outside [0, N-1])
    
    # Between breakpoints, it's linear. The maximum is at a breakpoint or 
    # at the peak of a linear segment.
    
    # For simplicity and speed, I'll compute the values at breakpoints and use 
    # the maximum as an approximation (it's actually an upper bound for the 
    # continuous case if the function is unimodal in each segment, which it is 
    # for nonnegative heights).
    
    h = np.array(heights, dtype=float)
    n = len(h)
    # Compute the autoconvolution values at breakpoints
    # (f*f)(k*w) = w * sum_j h[j] h[j+k] for k = -(n-1), ..., n-1
    conv_vals = np.array([w * np.dot(h[max(0, -k):], h[max(0, k):]) for k in range(-(n-1), n)])
    # The maximum of the piecewise linear function is at most the max of these 
    # values plus a small amount, but for nonnegative h, the function is 
    # concave in each segment, so the max is at a breakpoint or at the peak.
    # For nonnegative h, the autoconvolution is nonnegative and the max is 
    # actually at a breakpoint for the "center" region.
    
    return np.max(conv_vals)

def _score(heights):
    """Compute the score: normalized maximum of autoconvolution."""
    h = np.array(heights, dtype=float)
    # Normalize: the task says "normalized maximum of its autoconvolution"
    # I'll interpret this as the maximum of the autoconvolution divided by 
    # the L1 norm of f, or some other normalization.
    # For an upper bound on C1, the normalization is likely by the L2 norm 
    # or L1 norm of f.
    # Let's use max(autoconv) / (integral f)^2 as the normalization, which 
    # gives a dimensionless quantity.
    l1 = np.sum(h) * (0.5 / N)
    if l1 == 0:
        return 1e10
    max_ac = _autoconvolution_max(h)
    return max_ac / (l1 * l1)

def construct():
    """Construct a step function to minimize the normalized max autoconvolution."""
    t_start = time.time()
    deadline = t_start + 130  # Safety margin before 150s limit
    
    # Initial guess: uniform
    x0 = np.ones(N)
    
    # Use scipy minimize with L-BFGS-B
    def objective(x):
        return _score(x)
    
    # Try a few different initializations
    best_x = x0.copy()
    best_score = _score(x0)
    
    # Uniform initialization
    res = minimize(objective, x0, method='L-BFGS-B', 
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    if res.success and res.fun < best_score:
        best_x = res.x
        best_score = res.fun
    
    if time.time() > deadline - 10:
        return list(best_x)
    
    # Try a cosine initialization
    x1 = 0.5 + 0.5 * np.cos(np.pi * np.arange(N) / N)
    res = minimize(objective, x1, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    if res.success and res.fun < best_score:
        best_x = res.x
        best_score = res.fun
    
    if time.time() > deadline - 10:
        return list(best_x)
    
    # Try a Gaussian initialization
    x2 = np.exp(-0.5 * ((np.arange(N) - N/2) / (N/4))**2)
    res = minimize(objective, x2, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    if res.success and res.fun < best_score:
        best_x = res.x
        best_score = res.fun
    
    if time.time() > deadline - 10:
        return list(best_x)
    
    # Refine the best solution with more iterations
    res = minimize(objective, best_x, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
    if res.success and res.fun < best_score:
        best_x = res.x
        best_score = res.fun
    
    return list(best_x)
# EVOLVE-BLOCK-END
