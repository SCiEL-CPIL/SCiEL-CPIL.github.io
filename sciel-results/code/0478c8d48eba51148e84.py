"""
Uncertainty inequality: tune Hermite-Gaussian test function coefficients to minimize the upper bound on C4.

We parameterize g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is derived from the constraint g(0) = 0.

The score is the upper bound on A(f)*A(F(f)) (or C4-related constant).
We use scipy.optimize to minimize this bound over (a0, a4, a8).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.90

# Precompute grid for max finding
_OPT_GRID_X = np.linspace(-10, 10, 10000)
_OPT_GRID_H = {
    0: np.ones_like(_OPT_GRID_X),
    4: eval_hermite(4, _OPT_GRID_X),
    8: eval_hermite(8, _OPT_GRID_X),
    12: eval_hermite(12, _OPT_GRID_X)
}
_OPT_GRID_EXP = np.exp(-0.5 * _OPT_GRID_X**2)
_OPT_BASIS = np.column_stack([
    _OPT_GRID_H[0] * _OPT_GRID_EXP,
    _OPT_GRID_H[4] * _OPT_GRID_EXP,
    _OPT_GRID_H[8] * _OPT_GRID_EXP,
    _OPT_GRID_H[12] * _OPT_GRID_EXP
])

# Analytic L2 norms squared for H_k(x) exp(-x^2/2)
_SQRT_PI = np.sqrt(np.pi)
_L2_NORMS_SQ = np.array([
    _SQRT_PI * (2**0) * factorial(0, exact=True),
    _SQRT_PI * (2**4) * factorial(4, exact=True),
    _SQRT_PI * (2**8) * factorial(8, exact=True),
    _SQRT_PI * (2**12) * factorial(12, exact=True)
])

def _compute_bound_fast(a0, a4, a8):
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    coeffs = np.array([a0, a4, a8, a12])
    
    l2_sq = np.dot(coeffs**2, _L2_NORMS_SQ)
    if l2_sq <= 0:
        return 1e10
    l2 = np.sqrt(l2_sq)
    
    g_vals = _OPT_BASIS @ coeffs
    l_inf = np.max(np.abs(g_vals))
    
    return (l_inf / l2)**2

def _compute_bound_high(a0, a4, a8):
    x = np.linspace(-12, 12, 50000)
    exp_term = np.exp(-0.5 * x**2)
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    coeffs = np.array([a0, a4, a8, a12])
    
    l2_sq = np.dot(coeffs**2, _L2_NORMS_SQ)
    if l2_sq <= 0:
        return 1e10
    l2 = np.sqrt(l2_sq)
    
    g_vals = coeffs[0]*H0*exp_term + coeffs[1]*H4*exp_term + coeffs[2]*H8*exp_term + coeffs[3]*H12*exp_term
    l_inf = np.max(np.abs(g_vals))
    
    return (l_inf / l2)**2

def construct():
    start_time = time.time()
    deadline = 210.0 * DEADLINE_FRACTION
    
    best_val = float('inf')
    best_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    try:
        seed_val = _compute_bound_fast(best_x[0], best_x[1], best_x[2])
        if seed_val < best_val:
            best_val = seed_val
    except Exception:
        pass
    
    try:
        if time.time() - start_time < deadline - 60:
            res = minimize(
                lambda x: _compute_bound_fast(x[0], x[1], x[2]),
                best_x,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass
    
    try:
        if time.time() - start_time < deadline - 30:
            res = minimize(
                lambda x: _compute_bound_fast(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass
    
    try:
        if time.time() - start_time < deadline - 10:
            current_high_val = _compute_bound_high(best_x[0], best_x[1], best_x[2])
            res_high = minimize(
                lambda x: _compute_bound_high(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_high.fun < current_high_val:
                best_x = res_high.x
                best_val = res_high.fun
    except Exception:
        pass

    return best_x
# EVOLVE-BLOCK-END
