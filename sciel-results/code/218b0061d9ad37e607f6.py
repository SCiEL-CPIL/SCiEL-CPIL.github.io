"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_start_time = time.time()
_TIMEOUT = 85.0

def _Om_of_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    num = Om0 * np.power(a, 4.0 + w0)
    den = num + (1.0 - Om0) * np.exp(wa * (a - 1.0))
    return num / den

def _compute_gamma(a, c, Om0, w0, wa):
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    return gamma

def _f_of_a(a, c, Om0, w0, wa):
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, c, Om0, w0, wa)
    gamma = np.clip(gamma, 0.1, 1.2)
    return np.power(Om_a, gamma)

# Grid for integration
_x_grid = np.linspace(-2.5, 0.0, 200)  # ln(a) from ln(0.082) to 0

_validation_models = [
    (0.3, -1.0, 0.0),
    (0.3, -1.0, 0.5),
    (0.3, -1.0, -0.5),
    (0.25, -1.0, 0.3),
    (0.35, -1.0, -0.3),
    (0.3, -0.9, 0.2),
    (0.3, -1.1, 0.4),
]
_a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_ln_a_values = np.log(_a_values)

_ref_cache = {}

def _get_reference(Om0, w0, wa):
    key = (Om0, w0, wa)
    if key in _ref_cache:
        return _ref_cache[key]
    
    f0 = (-3.0 + np.sqrt(9.0 + 6.0 * Om0)) / 2.0
    f_ref = np.zeros(len(_a_values))
    D_ref = np.zeros(len(_a_values))
    
    for i, ai in enumerate(_a_values):
        if ai >= 1.0:
            f_ref[i] = f0
            D_ref[i] = 1.0
        else:
            def ode(x, y):
                a_x = np.exp(x)
                Om_x = _Om_of_a(np.array([a_x]), Om0, w0, wa)[0]
                coeff = 3.0 + 1.5 * wa * (1.0 - a_x)
                f_val = y[0]
                return [-(f_val**2 + coeff * f_val - 1.5 * Om_x), f_val]
            
            ln_a = np.log(ai)
            sol = solve_ivp(ode, [0.0, ln_a], [f0, 0.0], rtol=1e-10, atol=1e-12)
            if sol.success:
                f_ref[i] = sol.y[0, -1]
                D_ref[i] = np.exp(sol.y[1, -1])
            else:
                f_ref[i] = f0
                D_ref[i] = 1.0
    
    _ref_cache[key] = (f_ref, D_ref)
    return _ref_cache[key]

def _D_normalized_fast(a, c, Om0, w0, wa):
    """Fast D_normalized using precomputed grid."""
    a = np.asarray(a, dtype=float)
    result = np.ones_like(a)
    
    # Compute f on the grid
    a_grid = np.exp(_x_grid)
    f_grid = _f_of_a(a_grid, c, Om0, w0, wa)
    
    # Cumulative integral from 0 to each x
    # integral_0^x f(exp(t)) dt
    # Use cumulative trapezoidal integration
    dx = _x_grid[1] - _x_grid[0]
    cum_int = np.zeros_like(_x_grid)
    for i in range(1, len(_x_grid)):
        cum_int[i] = cum_int[i-1] + 0.5 * (f_grid[i-1] + f_grid[i]) * dx
    
    # For each a, find the integral from 0 to ln(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            result[i] = 1.0
        else:
            ln_ai = np.log(ai)
            # Interpolate cum_int at ln_ai
            # cum_int is defined on _x_grid, which goes from -2.5 to 0
            # We need integral from 0 to ln_ai (which is negative)
            # = -integral from ln_ai to 0
            # cum_int at x gives integral from _x_grid[0] to x
            # integral from ln_ai to 0 = cum_int at 0 - cum_int at ln_ai
            # But cum_int[0] = 0 (integral from _x_grid[0] to _x_grid[0])
            # So we need integral from _x_grid[0] to 0 minus integral from _x_grid[0] to ln_ai
            # = cum_int[-1] - cum_int at ln_ai
            # Actually, let me redefine: let I(x) = integral from 0 to x of f(exp(t)) dt
            # Then D(a)/D(1) = exp(I(ln(a)))
            # For ln(a) < 0, I(ln(a)) = -integral from ln(a) to 0
            # 
            # Let me compute J(x) = integral from x to 0 of f(exp(t)) dt for x < 0
            # Then D(a)/D(1) = exp(-J(ln(a))) for a < 1
            # 
            # J(x) = integral from x to 0 = -integral from 0 to x
            # 
            # Let me just compute K(x) = integral from 0 to x for x in [-2.5, 0]
            # Using cumulative from the left:
            # K(x_i) = sum_{j=1}^{i} 0.5*(f(x_{j-1})+f(x_j))*(x_j-x_{j-1})
            # This gives integral from x_0 to x_i
            # We want integral from 0 to x, which is -integral from x to 0
            # integral from x to 0 = integral from x_0 to 0 - integral from x_0 to x = K(x_0->0) - K(x_0->x)
            # K(x_0->0) = K at the last point = cum_int[-1]
            # So integral from 0 to x = -(cum_int[-1] - cum_int_at_x) = cum_int_at_x - cum_int[-1]
            
            # Interpolate cum_int at ln_ai
            idx = np.searchsorted(_x_grid, ln_ai)
            if idx == 0:
                ci = cum_int[0]
            elif idx >= len(_x_grid):
                ci = cum_int[-1]
            else:
                x0, x1 = _x_grid[idx-1], _x_grid[idx]
                c0, c1 = cum_int[idx-1], cum_int[idx]
                t = (ln_ai - x0) / (x1 - x0)
                ci = c0 + t * (c1 - c0)
            
            # integral from 0 to ln_ai = ci - cum_int[-1]
            I_val = ci - cum_int[-1]
            result[i] = np.exp(I_val)
    
    return result

def _objective(c):
    if time.time() - _start_time > _TIMEOUT:
        return 1e10
    
    max_err = 0.0
    for (Om0, w0, wa) in _validation_models:
        f_ref, D_ref = _get_reference(Om0, w0, wa)
        f_approx = _f_of_a(_a_values, c, Om0, w0, wa)
        D_approx = _D_normalized_fast(_a_values, c, Om0, w0, wa)
        
        err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
        err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    
    return max_err

def construct():
    global _start_time
    _start_time = time.time()
    
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0.copy()
    best_obj = _objective(c0)
    
    try:
        result = minimize(_objective, c0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() - _start_time < _TIMEOUT and result.fun < best_obj:
            best_c = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() - _start_time < _TIMEOUT - 5:
        try:
            result2 = minimize(_objective, best_c, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if time.time() - _start_time < _TIMEOUT and result2.fun < best_obj:
                best_c = result2.x
                best_obj = result2.fun
        except Exception:
            pass
    
    best_c = np.clip(best_c, -10.0, 10.0)
    best_c = np.nan_to_num(best_c, nan=0.0, posinf=10.0, neginf=-10.0)
    
    return list(best_c)
# EVOLVE-BLOCK-END

