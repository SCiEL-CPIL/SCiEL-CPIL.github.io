"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
# Interval width: total length is 1/2, divided into N equal intervals
# [-1/4, 1/4] has length 1/2, so each interval has width 1/(2*N)
dx = 1.0 / (2.0 * N)

# Precompute the autoconvolution kernel structure
# For a step function f with values h[0..N-1] on intervals of width dx,
# the autoconvolution (f * f)(t) = integral f(t-s) f(s) ds
# On a discrete grid, we can compute this via FFT-based convolution.
# The autoconvolution will have 2N-1 points (for lags from -(N-1) to (N-1)).
# Actually, for discrete convolution of two length-N arrays, we get 2N-1 points.
# But we need to be careful about the scaling.

# Let's think about this more carefully:
# f(x) = h[i] for x in [i*dx - 1/4, (i+1)*dx - 1/4) for i = 0..N-1
# The autoconvolution g(t) = ∫ f(t-s) f(s) ds
# This is a continuous function. For the "normalized maximum", we need to figure out
# what normalization is used. The task says "normalized maximum of its autoconvolution".
# 
# A common normalization: max(g) / (∫f)^2 or max(g) / ∫f^2 or similar.
# Given that the uniform distribution gives score 1.0 (the seed), let's figure out what
# normalization makes the uniform case give a specific value.
#
# For uniform h[i] = 1 for all i:
# f(x) = 1 on [-1/4, 1/4], 0 elsewhere.
# ∫f dx = 1/2
# ∫f^2 dx = 1/2
# g(t) = ∫ f(t-s) f(s) ds
# For t in [-1/4, 1/4]: g(t) = 1/2 - |t| (triangle function)
# max g(t) = 1/2 at t=0
#
# If normalized max = max(g) / (∫f)^2 = (1/2) / (1/4) = 2
# If normalized max = max(g) / ∫f^2 = (1/2) / (1/2) = 1
# If normalized max = max(g) * N / (sum h)^2 ... let's think in discrete terms.
#
# Let's use discrete convolution. Let h be the array of heights.
# discrete_conv = np.convolve(h, h, mode='full') gives 2N-1 values.
# Each value represents the sum of products over overlapping intervals, scaled by dx.
# Actually, the continuous autoconvolution at discrete points can be approximated.
#
# For the uniform case h[i]=1:
# discrete_conv[i] for i=0..2N-2
# discrete_conv[0] = 1, discrete_conv[1] = 2, ..., discrete_conv[N-1] = N, ..., discrete_conv[2N-2] = 1
# The continuous autoconvolution max is at t=0, which corresponds to index N-1 in the full convolution.
# The value at index N-1 is N (sum of N ones times ones).
# The actual continuous value is N * dx = N * (1/(2N)) = 1/2. Correct.
#
# So the continuous autoconvolution max ≈ discrete_conv[N-1] * dx
# For uniform: N * dx = 1/2.
#
# Normalization options:
# Option A: max_cont / (sum_h * dx)^2 = (N*dx) / (N*dx)^2 = 1/(N*dx) = 1/(1/2) = 2 for uniform
# Option B: max_cont / (sum(h^2) * dx) = (N*dx) / (N*dx) = 1 for uniform
# Option C: max_cont / max_cont_uniform = 1 for uniform
#
# Since the seed program returns [1.0]*N and gets score 1.0, and the score is "upper bound on C1",
# and the search score is seed/value where seed=1.0, the raw value for the seed must be 1.0.
# So the normalized maximum for the uniform distribution must be 1.0.
# This suggests normalization = max_cont / ∫f^2 dx = 1.0 for uniform.
# Or equivalently, max_cont / (sum(h^2) * dx).
#
# Let's verify: for uniform, max_cont = 1/2, ∫f^2 = 1/2, ratio = 1. Yes.
#
# So the objective is: minimize max(g) / (∫f^2) where g = f*f.
# In discrete terms: minimize (max(discrete_conv) * dx) / (sum(h^2) * dx) = max(discrete_conv) / sum(h^2)
# Wait, that's not quite right because the max of the continuous function may not be exactly at the grid point.
# But for a step function, the autoconvolution is piecewise linear, and the maximum is at t=0 (by symmetry and unimodality for non-negative functions).
# At t=0, g(0) = ∫ f(s)^2 ds = sum(h^2) * dx.
# So max(g) = g(0) = sum(h^2) * dx for any non-negative function? No, that's only if the function is symmetric and the peak is at 0.
# Actually, for non-negative f, g(0) = ∫f(s)^2 ds, and g(t) for t≠0 may be smaller.
# By Cauchy-Schwarz, g(0) >= g(t) for all t? Not necessarily.
# But for a step function that is non-negative, the autoconvolution is maximized at t=0.
# Let me verify: g(t) = ∫ f(t-s) f(s) ds. For t>0, the overlap region is smaller than at t=0.
# Since f is non-negative, the integral over a smaller region is smaller. So yes, max(g) = g(0).
#
# Therefore, max(g) = sum(h^2) * dx.
# And the normalization is max(g) / ∫f^2 = (sum(h^2) * dx) / (sum(h^2) * dx) = 1.
# That can't be right because then all functions would have score 1.
#
# I must be misunderstanding the normalization. Let me re-read: "normalized maximum of its autoconvolution".
# Perhaps it's max(g) / (∫f)^2? For uniform: (1/2) / (1/4) = 2. But seed gives 1.0.
# Or max(g) / max(g_uniform) = 1 for uniform. That would make sense.
# So the score = max(g) / max(g_uniform) = max(g) / (1/2) = 2 * max(g).
# For uniform, this is 2 * (1/2) = 1. 
#
# So we want to minimize 2 * max(g) = 2 * g(0) = 2 * sum(h^2) * dx.
# But wait, if we scale h by c, then g scales by c^2, so the score scales by c^2.
# To minimize, we'd make h arbitrarily small. That can't be right.
#
# There must be a constraint. Perhaps ∫f = 1 or sum(h) * dx = 1 or sum(h) = 1.
# If sum(h) = 1, then for uniform, h[i] = 1/N, sum(h^2) = N * (1/N^2) = 1/N.
# max(g) = (1/N) * dx = 1/(N*dx) = 1/(1/2) = 2. That doesn't give 1.
#
# Let's try: the normalized maximum is max(g) / (∫f)^2 with the constraint that the function is a probability density or something.
# Actually, let's just assume the score is computed as:
# score = max(np.convolve(h, h, 'full')) / (sum(h)**2)
# For uniform h=[1,1,...,1], max_conv = N, sum(h)^2 = N^2, ratio = 1/N. That's not 1.
#
# Or score = max_conv * dx / (sum(h)*dx)^2 = (max_conv * dx) / (sum(h)*dx)^2
# For uniform: (N*dx) / (N*dx)^2 = 1/(N*dx) = 2. Not 1.
#
# Let's try: the evaluator computes the actual continuous autoconvolution max and normalizes by something.
# Given the ambiguity, let's look at the inspiration: "minimize the normalized maximum of its autoconvolution".
# A common choice in harmonic analysis is to normalize by the L1 norm squared or L2 norm.
#
# Let me just try to minimize max_conv / sum(h^2) with a fixed sum(h) constraint, or minimize max_conv with sum(h)=1.
# Actually, the simplest interpretation that gives 1.0 for uniform:
# score = max_conv / (N * sum(h^2)) * N ... I'm going in circles.
#
# Let's just use: minimize max(np.convolve(h,h,'full')) subject to sum(h)=N (so average height is 1).
# For uniform, this gives max_conv = N. The score would be N. But seed is 1.0.
#
# I'll go with: the score is max_conv / (sum(h)**2 / N) = N * max_conv / (sum(h)**2).
# For uniform: N * N / N^2 = 1. Yes!
# So score = N * max_conv / (sum(h)**2).
# We want to minimize this.
#
# Let's verify: if h is concentrated at one point, h[0]=N, rest 0.
# max_conv = N^2 (at index 0). sum(h) = N. score = N * N^2 / N^2 = N. Much worse.
# If h is uniform, score = 1. So uniform is optimal? That contradicts the task.
#
# I think the normalization is different. Let me try: score = max_conv / max_conv_uniform.
# max_conv_uniform = N. So score = max_conv / N.
# For uniform, score = 1. For concentrated, score = N. We want to minimize.
# But we can make h small, making max_conv small. So there must be a constraint.
#
# The task says "nonnegative step function". No explicit constraint. But the score must be scale-invariant or there's a hidden constraint.
# Given the confusion, I'll assume the score is computed as:
# score = max(g_cont) / (∫f)^2 * (∫f^2) ... no.
#
# Let me just use L-BFGS-B to minimize max_conv with the constraint sum(h) = N, and see.
# Actually, let's use the formulation: minimize max_conv / (sum(h)/N)^2 * (1/N) ... 
#
# I'll go with: minimize max_conv subject to sum(h) = N, h >= 0.
# The score returned will be max_conv / N (normalized by uniform's max_conv).
# This should give 1.0 for uniform and potentially lower for other distributions.

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    def objective(h):
        # Compute discrete autoconvolution
        conv = np.convolve(h, h, mode='full')
        # The max is at the center (index N-1) for non-negative h
        # But let's take the max to be safe
        return np.max(conv)
    
    def gradient(h):
        # Gradient of max(conv) w.r.t. h
        # conv[i] = sum_j h[j] * h[i-j] (with appropriate bounds)
        # d(conv[i])/d(h[k]) = h[i-k] + h[k-i] (if within bounds)
        # The max is at index N-1 (center)
        # d(conv[N-1])/d(h[k]) = h[N-1-k] + h[k-(N-1)] ... this is getting complex.
        # Use numerical gradient or just let L-BFGS-B use finite differences.
        return None
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N,
                   'jac': lambda h: np.ones(N)}]
    
    bounds = [(0, None)] * N
    
    best_h = h0.copy()
    best_score = objective(h0) / N  # normalized
    
    # Try L-BFGS-B
    try:
        res = minimize(objective, h0, method='L-BFGS-B', 
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8,
                               'maxfun': 50000})
        if time.time() < deadline:
            h_opt = res.x
            # Ensure non-negativity and normalize
            h_opt = np.maximum(h_opt, 0)
            if np.sum(h_opt) > 0:
                h_opt = h_opt * N / np.sum(h_opt)
            score_opt = objective(h_opt) / N
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        # Return best found
        return list(best_h)
    
    # Try a few more restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Random initial guess
        h_rand = np.random.exponential(1.0, size=N)
        h_rand = h_rand * N / np.sum(h_rand)
        
        try:
            res = minimize(objective, h_rand, method='L-BFGS-B',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                   'maxfun': 20000})
            if time.time() > deadline - 10:
                break
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            if np.sum(h_opt) > 0:
                h_opt = h_opt * N / np.sum(h_opt)
            score_opt = objective(h_opt) / N
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
        except Exception:
            pass
    
    return list(best_h)

# EVOLVE-BLOCK-END
