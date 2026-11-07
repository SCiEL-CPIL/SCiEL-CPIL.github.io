"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _evaluate(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    # Spacing w: points are np.linspace(-0.5, 0.5, m+2), so w = 1.0/(m+1)
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2_terms = y[:m+1]**2 + y[:m+1] * y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(l2_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def _negative_value(h):
    """Negative objective for minimization."""
    h = np.maximum(h, 0.0)  # Enforce nonnegativity
    return -_evaluate(h)

def _gradient(h):
    """Numerical gradient for the negative objective."""
    h = np.maximum(h, 0.0)
    eps = 1e-8
    grad = np.zeros_like(h)
    for i in range(N):
        h_plus = h.copy()
        h_minus = h.copy()
        h_plus[i] += eps
        h_minus[i] -= eps
        # Clip to nonnegative
        h_plus = np.maximum(h_plus, 0.0)
        h_minus = np.maximum(h_minus, 0.0)
        grad[i] = (_negative_value(h_plus) - _negative_value(h_minus)) / (2 * eps)
    return grad

def construct():
    """Find optimal nonnegative step heights to maximize the objective."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Strategy 1: L-BFGS-B from diverse starting points
    start_points = [
        np.ones(N),
        np.linspace(0.5, 2.0, N),
        np.linspace(2.0, 0.5, N),
        np.cos(np.linspace(0, np.pi, N)) ** 2,
        np.exp(-np.linspace(-2, 2, N) ** 2),
        np.abs(np.sin(np.linspace(0, np.pi, N))),
        np.ones(N) * 1.5,
        np.random.RandomState(42).rand(N) * 2,
        np.random.RandomState(123).rand(N) * 2,
        np.random.RandomState(456).rand(N) * 2,
    ]
    
    for start in start_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _negative_value,
                start,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0.0)
            val = _evaluate(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Strategy 2: High-precision polish from the best solution found
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _negative_value,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_polish = np.maximum(res.x, 0.0)
            val = _evaluate(h_polish)
            if val > best_val:
                best_val = val
                best_h = h_polish.copy()
        except Exception:
            pass
    
    # Strategy 3: Try a few more random perturbations around the best
    if time.time() < deadline - 5:
        rng = np.random.RandomState(789)
        for _ in range(20):
            if time.time() > deadline:
                break
            perturbation = rng.randn(N) * 0.1
            h_try = np.maximum(best_h + perturbation, 0.0)
            try:
                res = minimize(
                    _negative_value,
                    h_try,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = np.maximum(res.x, 0.0)
                val = _evaluate(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.copy()
            except Exception:
                continue
    
    return list(np.maximum(best_h, 0.0))
