"""Second autocorrelation inequality: maximize ratio ||f||_2^2 / (||f||_1 * sup|f*f|)
for nonnegative step heights on 50 equal intervals of [-1/4, 1/4].

Strategy:
- The seed (all ones) gives a baseline. We need to find a shape that concentrates
  mass to reduce the autoconvolution peak relative to the L1 norm, or shapes the
  L2 norm favorably.
- The autoconvolution of a nonnegative function is maximized at the origin (lag 0)
  for many symmetric shapes, but we can try to "spread" the mass to reduce the peak
  while keeping L2/L1 ratio reasonable.
- Key insight: For the ratio R = ||f||_2^2 / (||f||_1 * sup|f*f|), if we scale f by c,
  ||cf||_2^2 = c^2 ||f||_2^2, ||cf||_1 = c ||f||_1, sup|cf*cf| = c^2 sup|f*f|.
  So R is scale-invariant. We can fix ||f||_1 = 1 (or sum=1) and optimize.
  
- Let's use a parameterized family and local optimization.
- A promising family: f(x) = 1 + a * cos(2*pi*k*x) for small a, or more generally
  smooth bumps. But since we have 50 discrete points, we can directly optimize the 50
  variables with constraints f_i >= 0.
  
- Approach: Use scipy.optimize.minimize with SLSQP or trust-constr to minimize
  -R(f) = -(sum f^2) / (sum f * max(conv(f)))
  subject to f_i >= 0.
  
- The autoconvolution of discrete samples: if f is defined on 50 points with spacing h,
  the convolution at lag k is sum_i f_i * f_{i+k} (with appropriate boundaries).
  But the problem says "autoconvolution" in the continuous sense approximated by steps.
  For step functions on equal intervals, the autoconvolution is a piecewise linear function.
  The supremum of the autoconvolution for nonnegative f is at lag 0, which is sum_i f_i^2 * h
  (approximately, if we think of it as sum of f_i * f_j * h for overlapping intervals).
  
  Actually, for step functions: f(x) = f_i for x in [(i-25)*h, (i-24)*h] where h = 1/50.
  The autoconvolution (f*f)(t) = integral f(x) f(t-x) dx.
  At t=0: (f*f)(0) = integral f(x)^2 dx = h * sum f_i^2.
  The supremum over t is at t=0 for nonnegative f (by rearrangement inequality / 
  the fact that the maximum of convolution of a nonnegative function with itself
  is at the origin when the function is symmetric, but in general it's at the point
  where the overlap of mass is maximum).
  
  For nonnegative functions, the autoconvolution is maximized at the origin if the
  function is symmetric about 0. If not symmetric, the maximum might be elsewhere.
  But we can assume symmetry to simplify, or just compute the full autoconvolution.
  
  Let's compute the autoconvolution properly. With 50 steps on [-1/4, 1/4], h = 1/50 = 0.02.
  The autoconvolution has support on [-1/2, 1/2] with 99 intervals (or 100 if we count carefully).
  Actually, the convolution of two step functions with N steps each has 2N-1 steps.
  
  For simplicity and speed, let's compute the autoconvolution at all relevant lags.
  The lag k corresponds to shift by k*h. The value at lag k is:
  (f*f)(k*h) = h * sum_{i,j: i-j = k} f_i * f_j ... 
  Actually, for step functions, the autoconvolution at offset m*h (m from -(N-1) to N-1) is:
  h * sum_{i=max(0,-m)}^{min(N-1, N-1-m)} f_i * f_{i+m}  (with appropriate indexing)
  
  Let's just use numpy.convolve on the array of heights (which represents the step function
  values) and multiply by h. The convolution of two arrays of length N gives length 2N-1.
  The maximum of this convolution (times h) is sup|f*f|.
  
  So:
  - f: array of 50 nonnegative numbers
  - conv = np.convolve(f, f)  # length 99
  - sup_conv = max(conv) * h  where h = 1/50
  - R = sum(f^2) / (sum(f) * sup_conv)
  
  We want to maximize R.
  
  Let's use scipy.optimize to maximize R over f >= 0.
  Since R is scale-invariant, we can fix sum(f) = 1 to remove the scale degree of freedom.
  Then R = sum(f^2) / sup_conv (since sum(f)=1).
  
  We'll optimize over f >= 0, sum(f) = 1, to maximize sum(f^2) / sup_conv.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
h = 1.0 / N
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _compute_ratio(f):
    """Compute R = ||f||_2^2 / (||f||_1 * sup|f*f|) for step function f."""
    f = np.asarray(f, dtype=np.float64)
    norm1 = np.sum(f)
    if norm1 < 1e-15:
        return 0.0
    norm2_sq = np.sum(f ** 2)
    conv = np.convolve(f, f)
    sup_conv = np.max(conv) * h
    if sup_conv < 1e-15:
        return 0.0
    return norm2_sq / (norm1 * sup_conv)

def _objective(f):
    """Negative ratio for minimization."""
    r = _compute_ratio(f)
    return -r

def _gradient(f):
    """Numerical gradient of the negative ratio."""
    f = np.asarray(f, dtype=np.float64)
    grad = np.zeros_like(f)
    eps = 1e-8
    base = _objective(f)
    for i in range(N):
        fp = f.copy()
        fp[i] += eps
        fm = f.copy()
        fm[i] -= eps
        grad[i] = (_objective(fp) - _objective(fm)) / (2 * eps)
    return grad

def construct():
    global START_TIME
    start = time.time()
    
    # Start with the seed
    best_f = np.ones(N)
    best_score = _compute_ratio(best_f)
    
    # Try a few initial guesses and optimize from each
    initial_guesses = [
        np.ones(N),  # uniform
        np.array([1.0] * 25 + [0.0] * 25),  # half
        np.array([0.0] * 10 + [1.0] * 30 + [0.0] * 10),  # center
        np.linspace(0, 1, N),  # ramp
        np.cos(np.linspace(-np.pi/2, np.pi/2, N)) ** 2,  # cosine bump
    ]
    
    for guess in initial_guesses:
        if time.time() - start > TIME_LIMIT - 10:
            break
        
        # Normalize to sum=1
        guess = np.asarray(guess, dtype=np.float64)
        s = np.sum(guess)
        if s < 1e-15:
            continue
        guess = guess / s
        
        # Optimize
        try:
            res = minimize(
                _objective,
                guess,
                method='SLSQP',
                bounds=[(0, None)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                             'jac': lambda x: np.ones(N)}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success or res.fun < -best_score:
                f_opt = np.maximum(res.x, 0)
                s = np.sum(f_opt)
                if s > 1e-15:
                    f_opt = f_opt / s
                    score = _compute_ratio(f_opt)
                    if score > best_score:
                        best_score = score
                        best_f = f_opt.copy()
        except Exception:
            pass
    
    # Local refinement with more iterations
    if time.time() - start < TIME_LIMIT - 20:
        try:
            res = minimize(
                _objective,
                best_f,
                method='SLSQP',
                bounds=[(0, None)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                             'jac': lambda x: np.ones(N)}],
                options={'maxiter': 2000, 'ftol': 1e-14}
            )
            if res.fun < -best_score:
                f_opt = np.maximum(res.x, 0)
                s = np.sum(f_opt)
                if s > 1e-15:
                    f_opt = f_opt / s
                    score = _compute_ratio(f_opt)
                    if score > best_score:
                        best_score = score
                        best_f = f_opt.copy()
        except Exception:
            pass
    
    # Return the best heights found
    return list(best_f)
