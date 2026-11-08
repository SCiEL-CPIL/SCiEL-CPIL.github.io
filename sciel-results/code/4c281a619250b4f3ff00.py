"""Optimize 50 nonnegative step heights on [-1/4,1/4] to maximize
   (sum h_i^2) / (sum h_i * sup_k (h*h)_k)
   where (h*h)_k = sum_i h_i h_{i+k} is the autoconvolution at lag k.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
# Grid points for the 50 intervals: x_i = -1/4 + (i + 0.5)/50 for i in 0..49
# Actually "50 nonnegative heights on equal intervals of [-1/4, 1/4]"
# This means we have 50 steps, each of width 1/50, covering [-1/4, 1/4].
# The autoconvolution of a step function: if h_i is the height on interval i,
# then the convolution at offset k (in units of intervals) is sum_i h_i * h_{i+k}
# But we need to be careful: the autoconvolution is a function, and we take its supremum.
# For a step function with N steps, the autoconvolution at continuous offset t is:
#   (f*f)(t) = integral f(x) f(x-t) dx
# If f is constant h_i on interval [x_i, x_{i+1}], then:
#   (f*f)(t) = sum over pairs of intervals of h_i h_j * (overlap length of interval i and interval j shifted by t)
# The supremum over t of this convolution.

# For simplicity and speed, let's compute the discrete autoconvolution:
# c[k] = sum_i h[i] * h[i+k] for k = -(N-1)..(N-1)
# But the true supremum of the continuous convolution might be higher than the max of c[k].
# However, for step functions, the convolution is piecewise linear (actually piecewise constant? no, piecewise linear with slope changes at interval boundaries).
# Actually, the convolution of two step functions is a piecewise linear function. Its maximum occurs either at a point where the derivative changes (i.e., at an interval boundary offset) or where the derivative is zero in between.

# For a practical approach: compute the convolution on a fine grid and take the max.
# But that's slow for optimization. Let's use the discrete approximation first, then refine.

# Actually, let's think more carefully. The step function f(x) = h_i for x in [x_i, x_{i+1}).
# The autoconvolution (f*f)(t) = integral f(x) f(x-t) dx.
# This is a piecewise linear function of t. The breakpoints are at t = x_j - x_i for all i,j.
# Since x_i are equally spaced with spacing dx = 1/50, the breakpoints are at multiples of dx.
# So (f*f)(t) is piecewise linear with breakpoints at t = k*dx for integer k.
# In each interval between consecutive breakpoints, (f*f)(t) is linear.
# Therefore, the maximum occurs at a breakpoint t = k*dx.
# At t = k*dx, (f*f)(k*dx) = sum_i h_i * h_{i-k} * dx ... wait, let me recompute.

# Let dx = 1/50. Interval i covers [i*dx - 1/4, (i+1)*dx - 11/4]... no.
# Let's say interval i covers [a_i, a_{i+1}] where a_i = -1/4 + i*dx.
# f(x) = h_i for x in [a_i, a_{i+1}).
# (f*f)(t) = integral f(x) f(x-t) dx.
# For t = k*dx, the shift aligns interval i with interval i-k (partially).
# Actually, the overlap between interval [a_i, a_{i+1}] and [a_j + t, a_{j+1} + t] is:
#   max(0, min(a_{i+1}, a_{j+1}+t) - max(a_i, a_j + t))
# For t = k*dx, a_{j+1}+t = a_j + (j+1-k)*dx... this is getting complicated.

# Simpler: since the grid is uniform, the autoconvolution at offset k*dx is:
#   c[k] = dx * sum_i h_i * h_{i-k}  (with h out of range = 0)
# And since the function is piecewise linear between these points, and linear on each segment,
# the max is at one of the k*dx points. So sup_t (f*f)(t) = max_k c[k] = dx * max_k sum_i h_i h_{i-k}.

# So the ratio is:
#   R = (sum h_i^2 * dx) / (sum h_i * dx * max_k sum_i h_i h_{i-k})
#   = (sum h_i^2) / (sum h_i * max_k sum_i h_i h_{i-k})
# The dx cancels! So we can just work with the discrete autoconvolution.

def _autoconv_max(h):
    """Compute max_k sum_i h[i] * h[i-k] for the given height array."""
    n = len(h)
    # Full convolution
    c = np.correlate(h, h, mode='full')
    return c.max()

def _ratio(h):
    """Compute the ratio (sum h^2) / (sum h * max autoconv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    ac_max = _autoconv_max(h)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _neg_ratio(h):
    return -_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try multiple initializations with L-BFGS-B
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Gaussian-like
    x = np.linspace(0, N-1, N)
    for sigma in [3, 5, 8, 12]:
        gauss = np.exp(-0.5 * ((x - N//2) / sigma)**2)
        initializations.append(gauss)
    
    # 3. Bimodal
    for gap in [5, 10, 15]:
        h = np.zeros(N)
        h[:N//2 - gap//2] = 1.0
        h[N//2 + gap//2:] = 1.0
        initializations.append(h)
    
    # 4. Random nonnegative
    for _ in range(10):
        initializations.append(rng.random(N))
    
    # 5. Sparse peaks
    for _ in range(5):
        h = np.zeros(N)
        peaks = rng.choice(N, size=rng.integers(2, 6), replace=False)
        h[peaks] = 1.0
        initializations.append(h)
    
    # 6. Triangular
    tri = np.array([min(i, N-1-i, N//2) for i in range(N)], dtype=float)
    if tri.max() > 0:
        initializations.append(tri / tri.max())
    
    for h0 in initializations:
        if time.time() > deadline - 5:
            break
        h0 = np.asarray(h0, dtype=float)
        # Normalize to avoid scale issues (ratio is scale-invariant)
        if h0.max() > 0:
            h0 = h0 / h0.max()
        
        try:
            res = minimize(_neg_ratio, h0, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < _neg_ratio(best_h):
                best_h = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    # Try differential evolution on a subset of promising regions
    if time.time() < deadline - 30:
        try:
            # Coarse DE search
            bounds = [(0, 1)] * N
            de_res = differential_evolution(_neg_ratio, bounds, seed=42,
                                            maxiter=30, popsize=10, tol=1e-6,
                                            polish=False, update_interval=1)
            if de_res.fun < _neg_ratio(best_h):
                best_h = de_res.x
                best_score = -de_res.fun
        except Exception:
            pass
    
    # Refine with L-BFGS-B from the best DE result
    if time.time() < deadline - 5:
        try:
            res = minimize(_neg_ratio, best_h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _neg_ratio(best_h):
                best_h = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
