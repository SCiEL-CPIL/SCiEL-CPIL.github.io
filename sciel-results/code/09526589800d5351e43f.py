import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 600
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _autoconv_peak(h):
    """Compute the normalized maximum of the autoconvolution of step function h.
    
    h: array of N nonnegative step heights on equal intervals of [-1/4, 1/4].
    The autoconvolution at lag k (integer) is sum_i h_i * h_{i+k} (with appropriate bounds).
    We normalize by the L1 norm squared or similar. The task says "normalized maximum".
    
    For a step function on [-1/4, 1/4] with N equal intervals, each interval has width w = (1/2)/N = 1/(2N).
    The autoconvolution (correlation) at discrete lag m is sum_i h_i * h_{i+m} * w^2 for continuous, 
    but since we're dealing with step functions, the standard autoconvolution at lag m (in units of intervals)
    is sum_i h_i * h_{i+m}.
    
    The "normalized maximum" likely means: max over lags of (autoconv at lag) / (total mass)^2 * (total mass)
    or simply max_lag (sum_i h_i h_{i+m}) / (sum h_i)^2 * something.
    
    Actually, for the first autocorrelation inequality (C1), the relevant quantity is:
    C1 = sup over nonnegative f of max_x |f*f(x)| / (||f||_1^2)
    
    For step functions with N intervals of width w=1/(2N) on [-1/4, 1/4]:
    f(x) = h_i for x in interval i
    ||f||_1 = w * sum_i h_i
    (f*f)(x) at the center of interval corresponding to lag m is approximately w^2 * sum_i h_i h_{i+m}
    
    So max |f*f(x)| / ||f||_1^2 = w^2 * max_m sum_i h_i h_{i+m} / (w^2 * (sum h_i)^2) = max_m sum_i h_i h_{i+m} / (sum h_i)^2
    
    So the score to minimize is: max_m (sum_i h_i h_{i+m}) / (sum_i h_i)^2
    where m ranges over valid lags.
    """
    h = np.asarray(h, dtype=np.float64)
    S = np.sum(h)
    if S <= 0:
        return 0.0
    # Compute all lags
    max_val = 0.0
    for m in range(-N + 1, N):
        if m >= 0:
            num = np.dot(h[:-m], h[m:]) if m < N else 0.0
        else:
            num = np.dot(h[-m:], h[:m + N]) if -m < N else 0.0
        val = num / (S * S)
        if val > max_val:
            max_val = val
    return max_val

def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # We want to minimize max_m (sum_i h_i h_{i+m}) / (sum h_i)^2
    # Subject to h_i >= 0
    
    # This is a non-convex problem in general, but let's try to use the structure.
    # Note that if we fix the normalization sum(h) = 1, we minimize max_m sum_i h_i h_{i+m}
    # subject to sum h_i = 1, h_i >= 0.
    
    # Let's use scipy.optimize.minimize with SLSQP.
    # The objective is the max of N-1+... quadratic forms, which is convex in h (max of convex functions).
    # Wait, sum_i h_i h_{i+m} is not convex in h. It's a quadratic form.
    # Actually, for fixed m, sum_i h_i h_{i+m} is a quadratic form h^T A_m h where A_m is a shift matrix.
    # This is not necessarily convex.
    
    # Alternative: use the fact that we can try to make the autoconvolution as flat as possible.
    # A known good candidate is the triangular function (B-spline of order 2), but that's for continuous.
    # For step functions, let's try a few structured initializations.
    
    candidates = [
        np.ones(N),  # uniform
        np.linspace(1, 0.5, N),  # linear decay
        np.array([1.0 if i < N//2 else 0.5 for i in range(N)]),  # step
    ]
    
    # Try to optimize each candidate
    best_h = np.ones(N)
    best_score = _autoconv_peak(best_h)
    
    for h_init in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        def objective(h):
            return _autoconv_peak(h)
        
        def objective_grad(h):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(h)
            for i in range(N):
                h_plus = h.copy()
                h_minus = h.copy()
                h_plus[i] += eps
                h_minus[i] -= eps
                if h_minus[i] < 0:
                    h_minus[i] = 0
                grad[i] = (objective(h_plus) - objective(h_minus)) / (2 * eps)
            return grad
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)}
        ]
        bounds = [(0, None)] * N
        
        try:
            res = minimize(
                objective, h_init,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            if res.success or res.fun < best_score:
                h_opt = np.maximum(res.x, 0)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt / s
                    score = _autoconv_peak(h_opt)
                    if score < best_score:
                        best_score = score
                        best_h = h_opt
        except Exception:
            pass
    
    # Return the best heights found
    return list(best_h)

