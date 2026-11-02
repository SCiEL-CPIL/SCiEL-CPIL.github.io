"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize the normalized maximum of the autoconvolution of a nonnegative
    step function with N equal intervals on [-1/4, 1/4].
    
    The step heights h[0..N-1] define a function f(x) = h[i] on interval i.
    The autoconvolution (f*f)(t) = integral f(x)f(t-x) dx.
    
    For a step function on N equal intervals of width w = (1/2)/N = 1/(2N),
    the autoconvolution is a piecewise linear function. The values at the
    "midpoints" of the convolution support correspond to discrete autoconvolution
    coefficients.
    
    Specifically, if we let w = 1/(2N), then the autoconvolution at time t = k*w
    (for integer k) involves sums of products of h[i]*h[j] where i+j = k or i+j = k-1,
    weighted by the overlap lengths.
    
    Actually, let's think more carefully. The function f is defined on [-1/4, 1/4]
    with N intervals of width w = (1/2)/N. Let the intervals be [a_i, a_{i+1}]
    where a_i = -1/4 + i*w.
    
    The autoconvolution g(t) = integral f(x) f(t-x) dx.
    
    Since f is a step function, g(t) is a piecewise linear function (actually,
    it's the convolution of two step functions, so it's piecewise quadratic... 
    no wait, convolution of two step functions is piecewise linear).
    
    Let's use a discrete approximation. If we sample the autoconvolution at the
    centers of the N+1 possible "peak" positions, or more precisely, the autoconvolution
    of a step function with N steps has a support of length 2*(1/2) = 1, and it's
    piecewise linear with at most 2N-1 linear pieces.
    
    A cleaner approach: discretize the autoconvolution. Let w = 1/(2N). The 
    autoconvolution g(t) for t in [-1/2, 1/2] can be computed. At t = m*w for
    m = -N, ..., N, the value g(m*w) can be expressed as a linear combination
    of products h[i]*h[j].
    
    Actually, for the purpose of minimizing the maximum, we can bound the 
    autoconvolution by its values at specific points. The autoconvolution of a
    nonnegative step function is maximized at t=0 (by symmetry and nonnegativity,
    actually not necessarily, but for the maximum over t, we need to check all t).
    
    Let's use the following: the autoconvolution g(t) = integral f(x)f(t-x)dx.
    For a step function, this is a piecewise linear function. Its maximum occurs
    at one of the "knot points" where the derivative changes, which are at t = 
    (i - j)*w for integers i, j. So we need to check g at t = k*w for k = -N+1, ..., N-1
    (or similar range).
    
    Let's compute g(k*w) for k = -(N-1), ..., N-1.
    
    g(k*w) = sum over i,j of h[i]*h[j] * length of intersection of [a_i, a_{i+1}] 
             and [a_j - k*w + a_0, ...]
    
    This is getting complex. Let me use a simpler discrete model.
    
    Alternative: Approximate the autoconvolution by the discrete autoconvolution
    of the heights. If we think of h as a vector, the discrete autoconvolution
    c[k] = sum_i h[i]*h[i+k] (for appropriate k). The continuous autoconvolution
    is related to this but with weighting by w.
    
    Specifically, g(t) ≈ w * sum_i h[i]*h[j] where the intervals overlap appropriately.
    
    For a more tractable problem, let's consider that the maximum of the continuous
    autoconvolution is bounded by a constant times the maximum of the discrete 
    autoconvolution coefficients. To be safe, let's directly formulate constraints
    on the discrete autoconvolution.
    
    Let c[k] = sum_{i} h[i] * h[i+k] for k = -(N-1), ..., N-1.
    The continuous autoconvolution's maximum is at most w * max_k |c[k]| * some factor.
    Actually, g(t) <= w * max_i h[i] * max_j h[j] * N in the worst case, but we want
    to minimize the actual max.
    
    Let's just minimize T subject to c[k] <= T for all k, and h[i] >= 0, and 
    sum h[i] = 1 (normalization). The "normalized maximum" likely means we normalize
    so that the L1 norm or integral is fixed. Let's assume sum h[i] = N (so the 
    average height is 1) or sum h[i] * w = 1 (integral = 1).
    
    The task says "normalized maximum of its autoconvolution". Let's assume the 
    normalization is such that the integral of f is 1, i.e., w * sum h[i] = 1, 
    so sum h[i] = 1/w = 2N.
    
    Actually, let's just minimize max_k c[k] / (sum h[i])^2 * w or something. 
    Let's keep it simple: minimize T such that c[k] <= T for all k, h[i] >= 0, 
    and sum h[i] = 1. Then the score is T.
    
    Let's use bisection on T and solve a linear feasibility problem.
    """
    # Discrete autoconvolution: c[k] = sum_i h[i]*h[i+k] for k = -(N-1),...,N-1
    # This is nonlinear (quadratic) in h. So linprog won't work directly.
    
    # We need to handle the quadratic nature. Let's use a different approach:
    # Minimize T subject to h^T A_k h <= T for all k, h >= 0, sum h = 1.
    # This is a quadratic constraint, not linear.
    
    # Alternative: Use the fact that for nonnegative h, the maximum of the 
    # autoconvolution is achieved at k=0 by some inequality? No, that's not true.
    
    # Let's try a simpler heuristic: spread the mass as evenly as possible.
    # If h[i] = 1/N for all i, then c[0] = sum (1/N)^2 = N*(1/N^2) = 1/N.
    # c[k] = (N-|k|)/N^2 for |k| < N.
    # So max c[k] = c[0] = 1/N.
    
    # The seed program returns [1.0]*N, so sum h = N. If we normalize by sum h,
    # then the normalized autoconvolution max would be (1/N) / N^2 * ... 
    
    # Let's just return a uniform distribution and see. The seed already does this.
    # We need to do better.
    
    # Idea: Use a triangular or other shape that reduces the peak of the 
    # autoconvolution. For example, a triangular pulse has a triangular 
    # autoconvolution, which might have a lower peak relative to its mass.
    
    # Let's try a triangular shape: h[i] proportional to distance from center, 
    # or a raised cosine, etc.
    
    # Actually, let's use scipy to solve a quadratic programming problem.
    # Minimize max_k h^T A_k h subject to sum h = 1, h >= 0.
    # This can be formulated as: minimize T subject to h^T A_k h <= T, sum h = 1, h >= 0.
    # This is a convex quadratic program (since A_k are positive semidefinite? 
    # Actually A_k for autoconvolution: A_k[i,j] = 1 if j-i = k. This is a 
    # permutation matrix, which is orthogonal, so h^T A_k h is not necessarily 
    # convex).
    
    # Let's just try a few candidate shapes and pick the best.
    
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Triangular (peak at center)
    center = N // 2
    tri = np.zeros(N)
    for i in range(N):
        dist = abs(i - center)
        tri[i] = max(0, 1 - dist / (center + 1))
    if tri.sum() > 0:
        candidates.append(tri / tri.sum())
    
    # Raised cosine
    x = np.linspace(0, 1, N)
    cos_wave = 0.5 * (1 + np.cos(2 * np.pi * x))
    candidates.append(cos_wave / cos_wave.sum())
    
    # Gaussian-like
    mu = N / 2
    sigma = N / 4
    gauss = np.exp(-0.5 * ((np.arange(N) - mu) / sigma) ** 2)
    candidates.append(gauss / gauss.sum())
    
    # Compute the discrete autoconvolution max for each candidate
    best_h = candidates[0]
    best_score = np.inf
    
    for h in candidates:
        # Compute autoconvolution
        c = np.correlate(h, h, mode='full')
        score = np.max(c)
        if score < best_score:
            best_score = score
            best_h = h
    
    return list(best_h)
# EVOLVE-BLOCK-END
