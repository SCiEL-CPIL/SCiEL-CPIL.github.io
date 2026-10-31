"""Program to minimize the normalized maximum autoconvolution of a nonnegative step function."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 600
# Intervals on [-1/4, 1/4]
# Total length = 1/2. 600 intervals.
# Width of each interval = (1/2) / 600 = 1/1200
# Centers of intervals:
# x_i = -1/4 + (i + 0.5) * (1/2) / 600 for i = 0..599
# Let's verify:
# i=0: -0.25 + 0.5/1200 = -0.25 + 0.000416...
# i=599: -0.25 + 599.5/1200 = -0.25 + 0.499583... = 0.249583...
# Max shift is 1/2. Autoconvolution of support [-1/4, 1/4] is support [-1/2, 1/2].

# Precompute coordinates
x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N) / 0.5
# Wait, let's just use np.linspace(-0.25, 0.25, N) and treat them as sample points or centers.
# For step functions, the value is constant on the interval.
# Let's define the grid properly.
# Interval width h = 1/1200.
# Centers c_i = -0.25 + (i + 0.5)*h.
h = 1.0 / (2 * N)
centers = -0.25 + (np.arange(N) + 0.5) * h

def compute_autoconv_peak(f):
    """
    Compute the normalized maximum autoconvolution.
    Score = max_k (f * f)(k) / (sum(f))**2
    Note: The convolution sum is sum_i f[i] f[i+k].
    The L1 norm squared is (sum f[i])^2.
    For discrete step functions, the continuous autoconvolution peak is approximated by
    max_k (1/h) * sum_i f[i] f[i+k] ?
    Actually, let's look at the definition.
    (f * f)(t) = integral f(x) f(t-x) dx.
    For step function f(x) = sum_j f_j 1_{[x_j, x_{j+1})}(x).
    (f * f)(t) = sum_{i,j} f_i f_j * measure({x | x in [x_i, x_{i+1}) and t-x in [x_j, x_{j+1})})
    = sum_{i,j} f_i f_j * length of intersection of [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j).
    
    The "normalized maximum" usually refers to:
    C1 = max_t (f*f)(t) / (integral f)^2
    Since integral f = sum f_i * h, (integral f)^2 = h^2 (sum f_i)^2.
    
    Let's compute the discrete approximation of the peak.
    The maximum of the continuous convolution occurs at some t.
    For step functions, the convolution is a piecewise linear function (if f is step, f*f is piecewise quadratic? No, f is step, f*f is piecewise linear? No.
    If f is a step function, f*f is a piecewise linear function?
    Let's check. f(x) = 1 on [0,1], 0 elsewhere. f*f(t) = t for 0<t<1, 2-t for 1<t<2. Piecewise linear.
    Generally, the convolution of two step functions is piecewise linear.
    The maximum occurs at one of the grid points where the "slopes" change, or at the peaks.
    The peaks are at t = x_i + x_j - x_k?
    Actually, the maximum of f*f for non-negative f is at t=0 if f is unimodal? Not necessarily.
    However, we can compute the values at the centers or at the boundaries.
    
    A robust way:
    Compute the discrete convolution of the sequence f_i.
    Let g_k = sum_i f_i f_{i-k}. This is the unnormalized correlation/convolution sum.
    The continuous value at t corresponding to shift k is roughly h * g_k?
    Let's derive.
    (f*f)(t) = integral f(x) f(t-x) dx.
    Approximate integral as sum_j f_j h * f(t - c_j).
    This is h * sum_j f_j f(t - c_j).
    If t is a center c_k, then f(t - c_j) = f(c_k - c_j).
    The distance c_k - c_j = (k-j)h.
    f(c_k - c_j) is the value of the step function at distance (k-j)h.
    Since f is defined on intervals, the value depends on which interval (k-j)h falls into.
    If we assume f is bandlimited or smooth, we might just use the discrete autocorrelation.
    However, the "normalized maximum" is the score.
    Score = max_t (f*f)(t) / (sum f_i * h)^2.
    
    Let's use the fact that for step functions, the maximum of f*f is achieved at one of the points t = c_i + c_j - c_k?
    Actually, the function f*f is piecewise linear. Its extrema are at the breakpoints.
    The breakpoints of f*f are at t = x_i - x_j (where x_i are the boundaries of the steps).
    The boundaries are at -0.25 + i*h for i=0..N.
    So the candidates for t are {x_i - x_j}.
    Due to symmetry and translation invariance of the score (wait, the score is not translation invariant?
    "minimize the normalized maximum of its autoconvolution".
    If we shift f, f*f shifts. The maximum value is the same.
    The L1 norm is the same.
    So the score is translation invariant.
    We can fix the position or just compute it generally.
    
    Efficient computation:
    1. Compute the discrete autocorrelation r_k = sum_i f_i f_{i+k} for k = -(N-1)..(N-1).
    2. The continuous peak is approximated by max_k (h * r_k) ?
       Let's check with a box function of width W containing M intervals.
       f_i = 1 for M intervals.
       integral f = M*h = W.
       (integral f)^2 = W^2.
       (f*f)(0) = W.
       Score = W / W^2 = 1/W.
       Discrete r_0 = sum 1*1 = M.
       h * r_0 = M * h = W.
       So h * r_0 / (sum f_i * h)^2 = W / W^2 = 1/W.
       
       What about the peak at other t?
       For a box, the peak is at t=0.
       If we use a smooth function, is the peak at t=0?
       If f is unimodal and symmetric, yes.
       But we don't enforce symmetry.
       However, the maximum of f*f for non-negative f is always >= (integral f)^2 / (length of support)?
       
       Let's assume the peak is well-approximated by max_k (h * r_k).
       Is this exact?
       For step functions, the convolution is piecewise linear between the breakpoints.
       The maximum of a piecewise linear function is at a breakpoint or at infinity (if slope doesn't change sign).
       Here it's 0 outside.
       The breakpoints are at t = x_i - x_j.
       The values at these points are NOT simply h * r_k.
       r_k corresponds to t = k*h.
       The breakpoints are at t = (i-j)*h.
       So the set of breakpoints IS {k*h}.
       Wait, x_i = -0.25 + i*h.
       x_i - x_j = (i-j)*h.
       So the breakpoints are exactly at the grid points of the discrete convolution shifts.
       
       However, the value of the continuous convolution at t = k*h is:
       (f*f)(k*h) = sum_{i,j} f_i f_j * length( [x_i, x_{i+1}) ∩ [k*h - x_{j+1}, k*h - x_j) ).
       
       If we just take h * r_k, that assumes the "overlap" is h for each pair (i, j) such that c_i - c_j = k*h?
       No. c_i - c_j = (i-j)h.
       If i-j = k, then c_i - c_j = k*h.
       The intervals [x_i, x_{i+1}) and [k*h - x_{j+1}, k*h - x_j)
       Note k*h - x_j = k*h - (-0.25 + j*h) = 0.25 + (k-j)h.
       And k*h - x_{j+1} = 0.25 + (k-j-1)h.
       So the second interval is [0.25 + (k-j-1)h, 0.25 + (k-j)h).
       The first interval is [-0.25 + i*h, -0.25 + (i+1)h).
       
       Let m = i-j.
       Interval 1: [-0.25 + j*h + m*h, -0.25 + j*h + (m+1)h)
       Interval 2: [0.25 + m*h - h, 0.25 + m*h) = [0.25 + (m-1)h, 0.25 + m*h)
       
       We need the intersection length of A and B.
       A = [-0.25 + (j+m)h, -0.25 + (j+m+1)h)
       B = [0.25 + (m-1)h, 0.25 + m*h)
       
       Let's check if A and B overlap.
       Length of A is h. Length of B is h.
       If m=0:
       A = [-0.25 + jh, -0.25 + (j+1)h)
       B = [0.25 - h, 0.25)
       Do they overlap?
       -0.25 + jh < 0.25 < -0.25 + (j+1)h ?
       0.25 + 0.25 < (j+1)h => 0.5 < (j+1)h.
       0.25 - h < -0.25 + jh => 0.5 < (j-1)h + 2h = (j+1)h.
       So if (j+1)h > 0.5, they overlap.
       Since total range is 0.5, h = 0.5/600.
       (j+1)h > 0.5 implies j+1 > 600, which is impossible.
       So for m=0, A and B do NOT overlap?
       Wait.
       A is in [-0.25, 0.25).
       B is in [-0.25, 0.25).
       For m=0, j is arbitrary.
       If j=0: A=[-0.25, -0.25+h). B=[0.25-h, 0.25). No overlap.
       If j=599: A=[0.25-h, 0.25). B=[0.25-h, 0.25). Overlap is h.
       
       So the discrete correlation r_0 = sum f_i^2.
       The term i=j contributes h * f_i^2 only if the intervals overlap.
       For i=j, A=[x_i, x_{i+1}), B=[x_{i+1}-h, x_{i+1}) = [x_i, x_{i+1}).
       Wait.
       B for j=i, m=0:
       B = [0.25 - h, 0.25). This is fixed?
       No, my derivation of B depended on m.
       B = [0.25 + (m-1)h, 0.25 + m*h).
       If m=0, B=[0.25-h, 0.25).
       This means B is always at the right end?
       That's because I fixed the support to [-0.25, 0.25].
       The convolution (f*f)(t) is non-zero only if t is in [-0.5, 0.5].
       The formula I derived calculates the contribution to (f*f)(t) where t = k*h.
       
       Actually, there is a simpler way.
       The function f is defined on a grid.
       We can use FFT to compute the convolution efficiently?
       The number of points is 600. 600^2 is 360,000. This is very fast in Python/NumPy.
       We can just compute the full discrete autocorrelation and check the peaks.
       
       But is h * max(r) the correct score?
       Let's test with a box.
       f = 1 for i=0..M-1.
       r_0 = M.
       h * M = W.
       Score = W / W^2 = 1/W.
       
       What if the peak is not at k=0?
       For non-negative f, is the maximum of f*f always at t=0?
       No. Consider f(x) = 1 on [0,1] and [3,4].
       f*f has peaks at 0, 2, 3, 4, 6, 7, 8?
       f*f(0) = 1.
       f*f(3) = 1 (overlap of [0,1] and [3,4] shifted? No.
       (f*f)(t) = integral f(x) f(t-x).
       If t=3, f(x) non-zero on [0,1] U [3,4].
       f(3-x) non-zero when 3-x in [0,1] U [3,4] => x in [2,3] U [-1,0].
       Intersection of [0,1]U[3,4] and [2,3]U[-1,0] is empty?
       [0,1] and [2,3] -> empty.
       [3,4] and [2,3] -> {3} (measure 0).
       [0,1] and [-1,0] -> {0}.
       So f*f(3) = 0.
       
       It seems for "separated" components, the cross terms might be small.
       But if we have a structure like a comb, the peak might be at the periodicity.
       However, we are minimizing the max.
       
       Given the time constraint and the need for a robust score,
       I will compute the discrete autocorrelation r_k = sum_i f_i f_{i-k}.
       I will assume the peak is max_k (h * r_k).
       This is an upper bound on the true peak?
       Actually, the true peak of the piecewise linear function is at the breakpoints.
       The value at the breakpoints is exactly h * r_k?
       Let's check the box case again.
       Box of width W=M*h.
       r_0 = M. h*M = W. True peak W.
       r_1 = M-1. h*(M-1) = W-h.
       True value at t=h is W-h.
       It matches.
       
       So Score = max_k (h * r_k) / (h * sum f_i)^2 = max(r_k) / (h * (sum f_i)^2).
       
       Wait.
       (integral f)^2 = (h * sum f_i)^2.
       Peak = h * max(r_k).
       Score = h * max(r_k) / (h^2 * (sum f_i)^2) = max(r_k) / (h * (sum f_i)^2).
       
       So we want to minimize max(r_k) / (sum f_i)^2, and then divide by h?
       No, h is a constant.
       Minimize S = max(r_k) / (sum f_i)^2.
       Then return S / h.
       
       Let's implement this.
    """
    # f is numpy array
    # Compute autocorrelation
    # np.correlate(f, f, mode='full')
    # This gives r_k for k = -(N-1) to N-1.
    # r_0 is at index N-1.
    r = np.correlate(f, f, mode='full')
    max_r = np.max(r)
    sum_f = np.sum(f)
    if sum_f == 0:
        return 0.0
    score = max_r / (h * sum_f**2)
    return score

def construct():
    start_time = time.time()
    deadline = 110.0  # 10 seconds buffer
    
    # Initial guess: Gaussian
    # x = centers
    sigma = 0.1
    f0 = np.exp(-centers**2 / (2 * sigma**2))
    
    best_score = compute_autoconv_peak(f0)
    best_f = f0.copy()
    
    # Use Differential Evolution or L-BFGS-B?
    # DE is robust but slow.
    # L-BFGS-B is faster but needs gradient.
    # We can use numerical gradient or just DE with low pop size.
    
    # Let's try a parameterized approach first to get a good start.
    # Family: (1 - (x/w)^2)^p clipped to 0.
    # Optimize w and p.
    
    def score_param(params):
        w, p = params
        if w <= 0 or p < 0:
            return 1e6
        f = (1 - (centers/w)**2)
        f = np.clip(f, 0, 1)
        f = f**p
        return compute_autoconv_peak(f)
    
    # Optimize w and p
    bounds = [(0.05, 0.25), (0.1, 5.0)]
    res = differential_evolution(score_param, bounds, seed=42, maxiter=20, popsize=5, polish=False)
    
    if time.time() > deadline - 20:
        # Fallback to current best
        return best_f.tolist()
        
    if res.fun < best_score:
        w, p = res.x
        f_opt = np.clip(1 - (centers/w)**2, 0, 1)**p
        best_f = f_opt
        best_score = res.fun
        
    # Refine with local search (L-BFGS-B)
    # We need a differentiable objective.
    # The objective is non-smooth (max).
    # We can use log-sum-exp approximation for max.
    # log(sum exp(r_k / T)) * T approx max(r_k).
    
    def smooth_score(f):
        r = np.correlate(f, f, mode='full')
        T = 1.0 / (100 * np.max(r) + 1)
        # Actually, just use max for now, L-BFGS-B might struggle.
        # Let's stick to DE or Nelder-Mead on the parameterized family or direct optimization with small steps.
        pass

    # Let's do a simple local search on the direct vector f.
    # Random perturbations.
    for i in range(5000):
        if time.time() > deadline:
            break
        if np.random.random() < 0.5:
            # Gaussian perturbation
            f_new = best_f + np.random.normal(0, 0.01 * np.max(best_f), N)
        else:
            # Coordinate ascent
            idx = np.random.randint(N)
            f_new = best_f.copy()
            f_new[idx] += np.random.uniform(-0.1, 0.1) * np.max(best_f)
        
        f_new = np.clip(f_new, 0, None)
        
        # Normalize to max 1? No, scaling f by c scales score by 1/c.
        # We can scale f to have max 1 to improve conditioning.
        max_f = np.max(f_new)
        if max_f > 0:
            f_new = f_new / max_f
            
        s = compute_autoconv_peak(f_new)
        if s < best_score:
            best_score = s
            best_f = f_new
    
    return best_f.tolist()

if __name__ == "__main__":
    # Simple test
    f = construct()
    print(f"Score: {compute_autoconv_peak(np.array(f))}")
