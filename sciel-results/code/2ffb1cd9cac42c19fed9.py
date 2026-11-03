import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(x):
    """
    Compute the normalized maximum of the autoconvolution of a step function
    with heights x on N equal intervals over [-1/4, 1/4].
    
    The step function f(t) = x[i] for t in [-1/4 + i*h, -1/4 + (i+1)*h) where h = 1/(2N).
    
    The autoconvolution (f * f)(s) = ∫ f(t) f(s-t) dt.
    
    For step functions, this can be computed via discrete convolution.
    Let h = 1/(2N). The continuous autoconvolution at point s is:
    (f*f)(s) = sum_{i,j} x[i] x[j] * length of {t: t in [i*h - 1/4, (i+1)*h - 1/4) and s-t in [j*h - 1/4, (j+1)*h - 1/4)}
    
    Actually, let's think in terms of the standard discrete autoconvolution.
    If we define the discrete autoconvolution c[k] = sum_i x[i] x[k-i] (with appropriate bounds),
    then the continuous autoconvolution's maximum is related to max(c) * h^2 / h = max(c) * h? 
    
    Let's be more careful. Let f(t) = sum_i x[i] * 1_{[a_i, a_{i+1})}(t) where a_i = -1/4 + i*h, h=1/(2N).
    Then (f*f)(s) = ∫ f(t) f(s-t) dt.
    
    This is a piecewise linear function. Its maximum occurs at one of the breakpoints.
    
    Alternatively, we can use the fact that for step functions, the L-infinity norm of the autoconvolution
    is bounded by the L2 norm squared (by Cauchy-Schwarz: ||f*f||_∞ <= ||f||_2^2).
    Actually, ||f*f||_∞ <= ||f||_∞ * ||f||_1.
    
    The "normalized maximum" likely means ||f*f||_∞ / ||f||_1^2 or similar.
    
    Let's assume the score is max_k (c[k]) * h where c is the discrete autoconvolution,
    normalized by (sum(x) * h)^2.
    
    Actually, let's just compute it properly.
    """
    if np.allclose(x, 0):
        return 0.0
    # Discrete autoconvolution
    c = np.convolve(x, x, mode='full')
    # The continuous autoconvolution maximum is approximately max(c) * h
    # where h = 1/(2N)
    h = 1.0 / (2 * N)
    # The L1 norm of f is sum(x) * h
    l1 = np.sum(x) * h
    # The L-infinity norm of f*f is max(c) * h (approximately, for step functions)
    # Actually, for step functions, the maximum of the continuous autoconvolution
    # is exactly max(c) * h^2 / h = max(c) * h? No.
    
    # Let's think again. f(t) = x[i] on interval of length h.
    # (f*f)(s) = sum_{i,j} x[i] x[j] * overlap_length(interval_i, s - interval_j)
    # The maximum of this is at most max_i x[i] * sum_j x[j] * h = max(x) * l1
    # And also at most (sum_i x[i] * h)^2 / h = l1^2 / h when all mass is concentrated.
    
    # The normalized maximum is likely ||f*f||_∞ / ||f||_1^2
    # ||f||_1 = sum(x) * h
    # ||f*f||_∞ <= max(x) * ||f||_1
    # So the ratio is <= max(x) / ||f||_1 = max(x) / (sum(x) * h)
    
    # For the constant function x = [1]*N:
    # sum(x) = N, ||f||_1 = N*h = 1/2
    # max(x) = 1
    # ||f*f||_∞ = max(x) * ||f||_1 = 1 * 1/2 = 1/2
    # Ratio = (1/2) / (1/2)^2 = 2
    
    # But the autoconvolution of a constant on [-1/4, 1/4] is a triangle:
    # (f*f)(s) = length of overlap of [-1/4, 1/4] and [s-1/4, s+1/4]
    # = max(0, 1/2 - |s|) for |s| <= 1/2
    # Max is 1/2 at s=0.
    # ||f||_1 = 1/2
    # So ||f*f||_∞ / ||f||_1^2 = (1/2) / (1/4) = 2.
    
    # So the score for constant is 2.0.
    
    # Let's compute the actual maximum of the continuous autoconvolution.
    # For step functions, it's a piecewise linear function. The maximum is at a breakpoint.
    # The breakpoints are at s = (i - j) * h for i, j in 0..N-1.
    
    # Simpler: use the discrete convolution to approximate.
    # The continuous autoconvolution at s = k*h is approximately c[k] * h^2 / h = c[k] * h?
    
    # Let's just use the bound: ||f*f||_∞ <= max(x) * ||f||_1
    # And the normalized score is ||f*f||_∞ / ||f||_1^2 <= max(x) / ||f||_1
    
    # To minimize this, we want to minimize max(x) / sum(x)
    # Which is achieved by making x as uniform as possible.
    # But wait, the actual maximum might be less than the bound.
    
    # Let's compute the actual max of the continuous autoconvolution.
    # For step functions, (f*f)(s) is piecewise linear with breakpoints at (i-j)*h.
    # The value at s = k*h can be computed from the discrete convolution.
    
    # Actually, for the purpose of this task, let's assume the score is:
    # max_k c[k] * h / (sum(x) * h)^2 = max(c) / (sum(x)^2 * h)
    
    # For constant x: c = [1, 2, 3, ..., N, N-1, ..., 1]
    # max(c) = N
    # sum(x) = N
    # score = N / (N^2 * h) = 1/(N*h) = 1/(N * 1/(2N)) = 2.
    
    # This matches! So the score is max(c) / (sum(x)^2 * h) where h = 1/(2N).
    
    c = np.convolve(x, x, mode='full')
    max_c = np.max(c)
    sum_x = np.sum(x)
    if sum_x == 0:
        return 0.0
    h = 1.0 / (2 * N)
    score = max_c / (sum_x * sum_x * h)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with a uniform distribution
    x_best = np.ones(N)
    score_best = compute_score(x_best)
    
    # Try a few different initializations and local optimizations
    # 1. Uniform (already done)
    # 2. Concentrated in the middle
    # 3. Random
    
    # Let's try to minimize the score using gradient-based optimization
    # The score is a piecewise linear function, so we can use subgradient methods
    
    # Try a few random initializations
    best_x = x_best.copy()
    best_score = score_best
    
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        # Random initialization
        x = np.random.rand(N)
        # Normalize to have the same L1 norm as the uniform case
        x = x / np.sum(x) * N
        
        # Local optimization using Nelder-Mead or similar
        # Since the function is piecewise linear, we can use subgradient descent
        
        # Simple subgradient descent
        lr = 0.1
        for step in range(1000):
            if time.time() > deadline - 10:
                break
            
            score = compute_score(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
            
            # Compute subgradient
            c = np.convolve(x, x, mode='full')
            max_idx = np.argmax(c)
            
            # The subgradient with respect to x[i] is:
            # d(score)/d(x[i]) = (c[max_idx] * 2 * x[j] / (sum_x^2 * h) - 2 * sum_x * max_c * x[i] / (sum_x^4 * h))
            # where j is such that max_idx = i + j - (N-1)
            
            # This is complex. Let's just use a simple heuristic:
            # Reduce the maximum element and increase smaller elements
            
            max_val = np.max(x)
            min_val = np.min(x)
            if max_val - min_val > 1e-6:
                # Move mass from max to min
                idx_max = np.argmax(x)
                idx_min = np.argmin(x)
                delta = (max_val - min_val) * 0.1
                x[idx_max] -= delta
                x[idx_min] += delta
                x = np.maximum(x, 0)
        
        # Also try a concentrated distribution
        if trial == 1:
            x = np.zeros(N)
            center = N // 2
            x[center-5:center+5] = 1.0
            x = x / np.sum(x) * N
            score = compute_score(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
    
    return list(best_x)
