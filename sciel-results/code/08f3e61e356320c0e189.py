import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width

def _autoconv_max_normalized(h):
    """Compute normalized max of autoconvolution for step heights h.
    Normalization: max(f*f) / (integral f)^2
    """
    h = np.asarray(h, dtype=float)
    integral = np.sum(h) * DELTA
    if integral <= 0:
        return np.inf
    # Compute autoconvolution of the step function
    # For piecewise constant f with heights h_i on intervals of width DELTA,
    # (f*f)(t) at the knot points can be computed via discrete convolution
    # The autoconvolution at t=0 is sum_i h_i^2 * DELTA
    # More generally, we need the full autoconvolution profile
    
    # Use FFT-based convolution for the discrete approximation
    # The step function sampled at interval centers, convolved with itself
    # Actually, for piecewise constant, the convolution is trapezoidal.
    # The maximum occurs at t=0 for nonnegative functions.
    
    # For nonnegative f, the autoconvolution is maximized at t=0
    # (f*f)(0) = integral f^2 = sum h_i^2 * DELTA
    # (f*f)(0) / (integral f)^2 = (sum h_i^2 * DELTA) / (sum h_i * DELTA)^2
    # = (sum h_i^2) / (DELTA * (sum h_i)^2)
    
    # Wait, but the problem says "normalized maximum of its autoconvolution"
    # For nonnegative functions, max autoconvolution is at t=0.
    # So we minimize (sum h_i^2) / (sum h_i)^2 * (1/DELTA)
    # This is minimized when all h_i are equal.
    
    # But that gives the seed answer. So maybe the normalization is different,
    # or the autoconvolution is defined differently.
    
    # Let me reconsider: maybe it's the L-infinity norm of the autoconvolution
    # normalized by something else, or maybe the function isn't necessarily
    # nonnegative in the autoconvolution sense... but the task says nonnegative.
    
    # Actually, re-reading: "minimize the normalized maximum of its autoconvolution"
    # Perhaps the normalization is by the L2 norm squared, not the L1 norm squared.
    # Or perhaps it's max(f*f) / ||f||_2^2
    
    # Let me just compute the full autoconvolution to be safe.
    
    # For a step function, the autoconvolution at the midpoints of intervals
    # can be computed. Let me use the discrete approach.
    
    # The autoconvolution (f*f)(t) for piecewise constant f:
    # At t = k*DELTA (k integer), (f*f)(k*DELTA) = DELTA * sum_i h_i * h_{i-k}
    # (with h_j = 0 for j outside [0, N-1])
    # This is DELTA times the discrete autoconvolution of h.
    
    # The maximum of the continuous autoconvolution occurs at one of the knot points
    # for piecewise constant nonnegative functions.
    
    # Compute discrete autoconvolution
    h_pad = np.zeros(2 * N - 1)
    h_pad[:N] = h
    autoconv = np.correlate(h_pad, h_pad, mode='full')  # This gives sum h_i h_{i+k}
    # Actually np.correlate(h, h, 'full') gives the cross-correlation which for
    # auto is the same as convolution reversed. Let me use FFT.
    
    # Simpler: use the fact that for nonnegative step functions, max is at t=0
    # (f*f)(0) = sum h_i^2 * DELTA
    # Normalized by (sum h_i * DELTA)^2:
    val = (np.sum(h**2) * DELTA) / (np.sum(h) * DELTA)**2
    return val

def _autoconv_max_normalized_full(h):
    """Full computation of normalized max autoconvolution."""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s <= 0:
        return np.inf
    # For nonnegative h, max autoconvolution is at t=0
    # (f*f)(0) = DELTA * sum(h^2)
    # Normalization by (integral f)^2 = (DELTA * sum(h))^2
    return (DELTA * np.sum(h**2)) / (DELTA * s)**2

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    def objective(x):
        # x are the heights, must be nonnegative
        # Use softplus or just clip
        h = np.maximum(x, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        return (np.sum(h**2)) / (s**2)
    
    def gradient(x):
        h = np.maximum(x, 0)
        s = np.sum(h)
        if s < 1e-10:
            return np.zeros_like(x)
        # d/dh_i of (sum h^2)/(sum h)^2
        # = (2*h_i * s^2 - sum(h^2)*2*s) / s^4
        # = 2*(h_i*s - sum(h^2)) / s^3
        sum_h2 = np.sum(h**2)
        grad = 2.0 * (h * s - sum_h2) / (s**3)
        # Only for positive h (subgradient at 0)
        grad[h <= 0] = 0.0
        return grad
    
    # The objective (sum h^2)/(sum h)^2 is minimized when all h are equal.
    # This is a convex problem! The minimum is 1/N when all h_i = 1/N.
    # But wait, that would give a better score than the seed.
    
    # Let me check: seed has all h_i = 1, so objective = (N*1)/(N^2) = 1/N
    # If we scale all h_i to 1/N, objective = (N*(1/N)^2)/(N*(1/N))^2 = (1/N)/(1) = 1/N
    # Same value! Because the objective is scale-invariant.
    
    # So the seed already achieves the minimum of this particular objective.
    # This means my understanding of the problem is wrong.
    
    # Let me reconsider. Maybe the "normalized maximum of autoconvolution" is not
    # max(f*f)/(integral f)^2 but something else.
    
    # Perhaps it's max(f*f) / ||f||_2^2 where ||f||_2^2 = integral f^2
    # Then for uniform f: max(f*f) = (1/2)*1 = 1/2, ||f||_2^2 = 1/2, ratio = 1
    # For a different f, this could be different.
    
    # Or maybe it's the L-infinity norm of the autoconvolution divided by the L1 norm squared,
    # but the autoconvolution is not just at t=0.
    
    # Actually, I think the issue is that for nonnegative functions, the autoconvolution
    # IS maximized at t=0, so any normalization that's scale-invariant would give the
    # same result for uniform vs. any other function only if the function shape doesn't matter.
    
    # Wait, (sum h^2)/(sum h)^2 DOES depend on the shape. If all h_i = c, it's N*c^2/(N*c)^2 = 1/N.
    # If only one h_i = c and rest 0, it's c^2/c^2 = 1. So uniform is better (1/N < 1).
    
    # So the seed with all 1s gives 1/N = 1/600 ≈ 0.00167.
    # The score is seed/value where seed is the seed program's value.
    # If the seed value is 1/600, and we can't do better, we get score 1.0.
    
    # This suggests the problem is more subtle. Maybe the autoconvolution is defined
    # on the real line and the "maximum" is the supremum, and for step functions it's
    # not necessarily at t=0? No, for nonnegative it is.
    
    # OR maybe the normalization involves the L2 norm of f, not L1.
    # Let's try: max(f*f) / ||f||_2^2
    # For uniform: (1/2) / (1/2) = 1
    # For a spike: if f = c on one interval, max(f*f) = c^2*DELTA, ||f||_2^2 = c^2*DELTA, ratio = 1
    # Same!
    
    # Hmm. Let me try yet another normalization.
    # Maybe it's max(f*f) / (max f)^2 * something?
    
    # I think I need to just implement the full autoconvolution computation and
    # use a reasonable normalization, then optimize.
    
    # Let me use: score = max_t (f*f)(t) / (∫f)^2
    # And compute the full autoconvolution to find the true max (even though for
    # nonnegative it should be at t=0).
    
    # Actually, let me just try a different approach: use scipy to minimize
    # the L-infinity norm of the autoconvolution (normalized) using a
    # log-sum-exp smoothing and L-BFGS.
    
    def full_objective(x):
        h = np.maximum(x, 0)
        s = np.sum(h) * DELTA
        if s < 1e-10:
            return 1e10
        # Compute autoconvolution via FFT
        # The step function values at interval midpoints
        # Convolution of two piecewise constant functions
        # Use the discrete convolution of the height vector
        h_conv = np.convolve(h, h, mode='full')
        # The autoconvolution at the knot points is DELTA * h_conv (approximately)
        # Actually, the continuous autoconvolution at t=k*DELTA is DELTA * sum_i h_i h_{i-k}
        # which is DELTA * h_conv[k] for appropriate indexing
        max_autoconv = DELTA * np.max(h_conv)
        return max_autoconv / (s**2)
    
    # Let's just run a simple optimization with a time limit
    best_h = h0.copy()
    best_val = full_objective(best_h)
    
    # Try L-BFGS with log-sum-exp smoothing
    def smooth_max_autoconv(h, temperature=0.01):
        h_conv = np.convolve(h, h, mode='full')
        s = np.sum(h) * DELTA
        if s < 1e-10:
            return 1e10, np.zeros_like(h)
        # Log-sum-exp of DELTA*h_conv / s^2
        vals = DELTA * h_conv / (s**2)
        max_val = np.max(vals)
        if max_val < 1e-30:
            return 0, np.zeros_like(h)
        # logsumexp
        shifted = vals / temperature - max_val / temperature
        logsumexp = max_val / temperature + np.log(np.sum(np.exp(shifted))) * temperature
        # Actually, let me just use a simple approximation
        return max_val, None
    
    # Given time constraints, let me try a few simple strategies
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best_h)
    
    # Try L-BFGS on the smoothed objective
    def obj_and_grad(x):
        h = np.maximum(x, 1e-10)
        h_conv = np.convolve(h, h, mode='full')
        s = np.sum(h) * DELTA
        if s < 1e-10:
            return 1e10, np.zeros_like(x)
        max_ac = DELTA * np.max(h_conv)
        val = max_ac / (s**2)
        # Gradient is complex, use numerical if needed
        return val, None
    
    # Use Nelder-Mead or similar derivative-free method on a reduced problem
    # Or just try a few heuristic patterns
    
    # Try: Gaussian-like profile
    t = np.linspace(-1, 1, N)
    for sigma in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        h_trial = np.exp(-t**2 / (2*sigma**2))
        val = full_objective(h_trial)
        if val < best_val:
            best_val = val
            best_h = h_trial.copy()
    
    # Try: triangular
    h_trial = np.abs(1 - 2*np.linspace(0, 1, N))
    val = full_objective(h_trial)
    if val < best_val:
        best_val = val
        best_h = h_trial.copy()
    
    # Try: cosine
    h_trial = 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N)))
    val = full_objective(h_trial)
    if val < best_val:
        best_val = val
        best_h = h_trial.copy()
    
    # Try to optimize with scipy
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        def obj_scipy(x):
            h = np.maximum(x, 0)
            return full_objective(h)
        
        try:
            res = minimize(obj_scipy, h0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'maxfun': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_h = np.maximum(res.x, 0)
        except:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        # Try from best_h
        try:
            res = minimize(obj_scipy, best_h, method='L-BFGS-B',
                          options={'maxiter': 100, 'maxfun': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_h = np.maximum(res.x, 0)
        except:
            pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-10:
        best_h = np.ones(N)
    
    return list(best_h)
