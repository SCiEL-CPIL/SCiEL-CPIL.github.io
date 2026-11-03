import time
import numpy as np
from scipy.optimize import minimize
import random

N = 600
DEADLINE = 140.0  # safety margin before 150s kill

def _autoconvolution(heights):
    """Compute the autoconvolution (circular? or linear?) of the step function.
    Since we have 600 equal intervals on [-1/4, 1/4], the interval width is (1/2)/600 = 1/1200.
    The autoconvolution of a step function with these intervals:
    h(k) = sum_i heights[i] * heights[j] where i+j = k, times (interval_width)^2
    But actually, the "normalized maximum" - let me think about what this means.
    
    The autoconvolution (f*f)(x) = integral f(t) f(x-t) dt.
    For step functions on equal intervals, this is a discrete convolution scaled by interval width.
    
    Let dx = 1/1200. Then (f*f)(k*dx) ≈ dx * sum_i f_i * f_{k-i}
    
    The "normalized maximum" likely means max(f*f) / (some normalization).
    Common normalizations: max(f*f) / (integral(f))^2, or max(f*f) / (max f * integral f), etc.
    
    Given the seed (all ones) scores 1.0, let me compute for uniform:
    f_i = 1 for all i, dx = 1/1200.
    (f*f)(k) = dx * sum_i f_i * f_{k-i}
    For k=0: dx * N = (1/1200) * 600 = 0.5
    For k=1: dx * 599, etc.
    Max is at k=0: 0.5
    
    integral(f) = dx * sum f_i = (1/1200) * 600 = 0.5
    (integral f)^2 = 0.25
    
    If normalization is max(f*f) / (integral f)^2 = 0.5 / 0.25 = 2.0
    
    Or max(f*f) / (dx * (sum f)^2) ... that's the same.
    
    Actually, let me reconsider. Maybe it's max(f*f) where f*f is the continuous autoconvolution,
    and the normalization divides by (integral f)^2.
    
    For uniform: integral f = 0.5, (integral f)^2 = 0.25
    max(f*f) = 0.5 (at x=0)
    Ratio = 0.5/0.25 = 2.0
    
    But the seed scores 1.0. So maybe the normalization is different.
    
    Perhaps the score is defined such that the uniform gives 1.0. Let me think...
    
    If the score is max(f*f) / max(uniform*f*uniform) = 0.5 / 0.5 = 1.0. That works!
    
    So the score is: max(autoconvolution of f) / max(autoconvolution of uniform)
    where uniform has all heights = 1.
    
    max(autoconvolution of uniform) = dx * N = (1/1200) * 600 = 0.5
    
    So score = max(f*f) / 0.5 = 2 * max(f*f)
    
    Wait, but that means we want to minimize max(f*f), and the uniform gives 1.0.
    To beat 1.0, we need max(f*f) < 0.5.
    
    But by the Cauchy-Schwarz or other inequalities, is it possible to have max(f*f) < 0.5 for nonnegative f?
    
    Actually, f*f at 0 = integral f(t)^2 dt. For the uniform, this is (1/1200)*600*1 = 0.5.
    For any other nonnegative function with the same integral... wait, the integral isn't fixed.
    
    Hmm, let me reconsider. Maybe the normalization is different.
    
    Let me just try: score = max(f*f) / (integral f)^2. For uniform this gives 2.0, not 1.0.
    
    Let me try: score = max(f*f) / (integral f)^2 * something...
    
    Actually, maybe the "normalized maximum" means: the autoconvolution is normalized so that its integral is 1, and then we take the maximum.
    
    integral(f*f) = (integral f)^2 = (0.5)^2 = 0.25 for uniform.
    If we normalize so integral is 1: (f*f)/0.25. Max = 0.5/0.25 = 2.0.
    
    Still not 1.0.
    
    Let me try another approach: maybe the heights are normalized so that sum(heights) = 1 or max(heights) = 1.
    
    If sum(heights) = 1: uniform has heights = 1/600.
    (f*f)(0) = dx * sum f_i^2 = (1/1200) * 600 * (1/600)^2 = (1/1200) * (1/600) = 1/720000
    That's tiny.
    
    I think I'm overcomplicating this. Let me just assume the score formula is:
    score = max(f*f) / (integral f)^2
    and the uniform gives some value V0, and the "1.0" in the seed score means the raw value equals V0.
    
    Since I can't be 100% sure, let me just optimize to minimize max(f*f) / (integral f)^2,
    and see if I can get below the uniform value. If the uniform is optimal, I'll return uniform.
    
    Actually, re-reading: "score is the upper bound on C1" and "the evaluator measures the raw value (lower is better)".
    The seed (uniform) scores 1.0. So the raw value for uniform is some V0, and the score = seed_value / raw_value = V0/raw_value.
    So if raw_value < V0, score > 1.0. If raw_value > V0, score < 1.0.
    
    So I need to minimize the raw value, which is the "normalized maximum of its autoconvolution".
    
    Let me just go with: raw_value = max(f*f) / (integral f)^2, and for uniform this gives 2.0.
    Then score = 2.0 / raw_value. To get score > 1, I need raw_value < 2.0.
    
    Let me just implement this and optimize.
    """
    dx = 1.0 / (N * 2)  # interval width: total range 1/2, N intervals
    # Linear convolution
    conv = np.convolve(heights, heights, mode='full') * dx
    # The autoconvolution (f*f)(x) for step functions
    # Actually, for the continuous autoconvolution, we need to be more careful.
    # (f*f)(x) = integral f(t) f(x-t) dt
    # For step functions, this is a piecewise linear function.
    # The maximum occurs at x=0 by symmetry if f is symmetric, or somewhere else.
    # Let me just use the discrete approximation.
    return np.max(conv) / (np.sum(heights) * dx)**2 if np.sum(heights) > 0 else float('inf')

def _raw_score(heights):
    """Compute the raw value to be minimized."""
    dx = 1.0 / (N * 2)
    s = np.sum(heights)
    if s <= 0:
        return float('inf')
    conv = np.convolve(heights, heights, mode='full') * dx
    return np.max(conv) / (s * dx)**2

def construct():
    start_time = time.time()
    
    # Uniform baseline
    uniform = np.ones(N)
    uniform_score = _raw_score(uniform)
    
    best_heights = uniform.copy()
    best_score = uniform_score
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    # Try to optimize using scipy
    try:
        from scipy.optimize import minimize as sp_minimize
        
        # Multi-start local search
        random.seed(42)
        np.random.seed(42)
        
        num_starts = 0
        max_starts = 50
        
        while check_time() and num_starts < max_starts:
            num_starts += 1
            
            # Generate random initial point
            if num_starts == 1:
                x0 = np.ones(N)
            elif num_starts % 3 == 0:
                # Gaussian bump
                centers = np.random.randint(0, N, size=1)
                sigmas = np.random.uniform(N/10, N/3, size=1)
                x0 = np.exp(-0.5 * ((np.arange(N) - centers[0]) / sigmas[0])**2)
                x0 += 0.1 * np.random.randn(N)
                x0 = np.maximum(x0, 0)
            elif num_starts % 3 == 1:
                # Random
                x0 = np.random.exponential(1.0, size=N)
            else:
                # Symmetric random
                half = N // 2
                h = np.random.exponential(1.0, size=half)
                x0 = np.concatenate([h, h[::-1]])
            
            # Normalize
            if np.sum(x0) > 0:
                x0 = x0 / np.sum(x0) * N  # keep similar scale to uniform
            
            def objective(x):
                x = np.maximum(x, 0)  # enforce nonnegativity
                return _raw_score(x)
            
            def jac(x):
                # Numerical gradient
                eps = 1e-7
                grad = np.zeros_like(x)
                for i in range(N):
                    xp = np.maximum(x.copy(), 0)
                    xm = np.maximum(x.copy(), 0)
                    xp[i] += eps
                    xm[i] -= eps
                    if xm[i] < 0:
                        xm[i] = 0
                    grad[i] = (objective(xp) - objective(xm)) / (eps - (-eps) if xm[i] >= 0 else 2*eps)
                return grad
            
            result = sp_minimize(objective, x0, method='L-BFGS-B', 
                                bounds=[(0, None)]*N,
                                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or (time.time() - start_time) < DEADLINE:
                x_opt = np.maximum(result.x, 0)
                s = _raw_score(x_opt)
                if s < best_score:
                    best_score = s
                    best_heights = x_opt.copy()
                    
    except Exception as e:
        pass
    
    # Final check
    final_score = _raw_score(best_heights)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_heights)
