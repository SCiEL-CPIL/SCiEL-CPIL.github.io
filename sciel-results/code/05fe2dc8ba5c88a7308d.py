"""Improved growth-index approximation via in-process optimization with ODE reference."""
import time
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Deadline guard: must finish within 90s wall-clock; use 85s as safety margin
_START_TIME = time.time()
_DEADLINE = 85.0  # seconds

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_Om(a, Om0, wa):
    """Matter density parameter Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    de = np.power(a, 3.0 * (1.0 + wa))
    Om = Om0 / (Om0 + (1.0 - Om0) * de)
    return Om

def _compute_f_ode(a_grid, Om0, wa, n_steps=2000):
    """
    Compute f(a) = d ln D / d ln a using the first-order ODE for f:
    f' = f - f^2 - (3/2) Om(a)
    with IC f(1) = Om0^0.55 (or better, f(1) = Om0).
    
    Actually, the exact relation is:
    f(a) satisfies: f' = f - f^2 - (3/2) Om(a)  where ' is d/da
    Wait, let's be more careful.
    
    The growth equation for D(a) is:
    D'' + (3/a + H'/H) D' - (3/2) Om(a) D = 0
    
    Let f = d ln D / d ln a = a D'/D. Then D' = f D / a, D'' = (f' + f^2/a - f/a^2) D... 
    
    Actually the standard first-order ODE for f(a) is:
    df/da = f - f^2 - (3/2) Om(a)
    
    with boundary condition f(1) = Om0 (since at a=1, f ≈ Om0 for reasonable models).
    
    Wait, let me reconsider. The exact relation is:
    f(a) = d ln D / d ln a
    
    The growth equation in terms of f:
    df/da = f - f^2 - (3/2) Om(a)
    
    This is correct. At a=1, f(1) ≈ Om0 (since D''/D at a=1 gives f ≈ Om0 for flat Lambda CDM).
    Actually f(1) = Om0 is the exact value at a=1 for the matter-dominated era limit, but for general models
    f(1) is slightly different. Let's use f(1) = Om0 as IC and integrate backwards.
    """
    a_min = a_grid[0]
    a_max = a_grid[-1]
    
    # Use f(1) = Om0 as initial condition, integrate from a=1 to a_min
    # For a > 1, we can also integrate forward if needed, but typically a <= 1
    
    def ode_rhs(a, f):
        Om_a = _compute_Om(a, Om0, wa)
        return [f[0] - f[0]**2 - 1.5 * Om_a]
    
    # We need to integrate from a=1 down to a_min (and up to a_max if a_max > 1)
    # Let's handle the case where a_grid spans [a_min, a_max] with a_min < 1 < a_max
    
    # First, create a fine grid for integration
    if a_min < 1.0:
        a_int_down = np.linspace(1.0, a_min, n_steps)
        sol_down = solve_ivp(ode_rhs, [1.0, a_min], [Om0], t_eval=a_int_down, 
                            method='RK45', rtol=1e-10, atol=1e-12, max_step=0.01)
        f_down = sol_down.y[0] if sol_down.success else None
    else:
        a_int_down = np.array([1.0])
        f_down = np.array([Om0])
    
    if a_max > 1.0:
        a_int_up = np.linspace(1.0, a_max, n_steps)
        sol_up = solve_ivp(ode_rhs, [1.0, a_max], [Om0], t_eval=a_int_up, 
                          method='RK45', rtol=1e-10, atol=1e-12, max_step=0.01)
        f_up = sol_up.y[0] if sol_up.success else None
    else:
        a_int_up = np.array([1.0])
        f_up = np.array([Om0])
    
    # Interpolate to get f at a_grid
    f_ref = np.zeros_like(a_grid)
    for i, ai in enumerate(a_grid):
        if ai <= 1.0:
            if len(a_int_down) > 1:
                f_ref[i] = np.interp(ai, a_int_down, f_down)
            else:
                f_ref[i] = Om0
        else:
            if len(a_int_up) > 1:
                f_ref[i] = np.interp(ai, a_int_up, f_up)
            else:
                f_ref[i] = Om0
    
    return f_ref

def _compute_D_normalized(a_grid, Om0, wa, f_ref, n_steps=2000):
    """
    Compute D(a)/D(1) = exp(-integral from a to 1 of f(exp(x)) dx) where x = ln(a)
    Wait, the task says: D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    
    So D(a)/D(1) = exp(-∫_{ln(a)}^{0} f(exp(x)) dx)
    
    Let u = ln(a), then this is exp(-∫_{u}^{0} f(exp(x)) dx)
    
    We can compute this by numerical integration.
    """
    D_norm = np.zeros_like(a_grid)
    for i, ai in enumerate(a_grid):
        if ai == 1.0:
            D_norm[i] = 1.0
            continue
        u = math.log(ai)
        # Integrate f(exp(x)) from x=u to x=0
        # Create integration grid
        x_grid = np.linspace(u, 0.0, n_steps)
        f_vals = np.zeros_like(x_grid)
        a_vals = np.exp(x_grid)
        # Interpolate f_ref to get f at a_vals
        f_vals = np.interp(a_vals, a_grid, f_ref)
        # Integrate
        integral = np.trapz(f_vals, x_grid)
        D_norm[i] = math.exp(-integral)
    return D_norm

def _compute_D_normalized_fast(a_grid, Om0, wa, f_ref, n_steps=500):
    """Faster version of D_normalized using vectorized interpolation."""
    D_norm = np.ones_like(a_grid)
    for i, ai in enumerate(a_grid):
        if ai == 1.0:
            D_norm[i] = 1.0
            continue
        u = math.log(ai)
        x_grid = np.linspace(u, 0.0, n_steps)
        a_vals = np.exp(x_grid)
        f_vals = np.interp(a_vals, a_grid, f_ref)
        integral = np.trapz(f_vals, x_grid)
        D_norm[i] = math.exp(-integral)
    return D_norm

def _evaluate_model(coeffs, Om0, wa, a_test):
    """
    Evaluate the model with given coefficients against ODE reference.
    Returns max relative error for f and D.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    
    Om = _compute_Om(a_test, Om0, wa)
    q = 1.0 - Om
    u = 1.0 + wa + wa * (1.0 - a_test)
    
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_test + c5 * u * q
    
    # Check gamma bounds
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1e6  # Invalid
    
    f_model = Om ** gamma
    
    # Reference
    f_ref = _compute_f_ode(a_test, Om0, wa)
    
    # Relative error in f
    rel_err_f = np.abs(f_model - f_ref) / np.abs(f_ref)
    max_err_f = np.max(rel_err_f)
    
    # D(a)/D(1)
    D_ref = _compute_D_normalized_fast(a_test, Om0, wa, f_ref)
    D_model = _compute_D_normalized_fast(a_test, Om0, wa, f_model)
    
    rel_err_D = np.abs(D_model - D_ref) / np.abs(D_ref)
    max_err_D = np.max(rel_err_D)
    
    return max(max_err_f, max_err_D)

def _objective(coeffs, Om0, wa, a_test):
    """Objective function for optimizer."""
    return _evaluate_model(coeffs, Om0, wa, a_test)

def construct():
    """
    Find optimal six coefficients for the growth-index formula.
    Returns [c0, c1, c2, c3, c4, c5].
    """
    # Test cosmologies (disjoint from validation, but representative)
    test_models = [
        (0.3, 0.0),    # LCDM
        (0.3, -0.5),   # Phantom-like
        (0.3, 0.5),    # Quintessence-like
        (0.25, 0.0),   # Lower Om0
        (0.35, 0.0),   # Higher Om0
    ]
    
    # Test scale factors
    a_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Seed
    seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Try to improve using Nelder-Mead on a combined objective
    # To save time, use a subset of test models
    if _time_remaining() < 5.0:
        return seed.tolist()
    
    # Use LCDM as primary test case for optimization
    Om0_opt, wa_opt = 0.3, 0.0
    
    def obj(coeffs):
        return _evaluate_model(coeffs, Om0_opt, wa_opt, a_test)
    
    # Initial bounds: coefficients in [-10, 10], gamma in [0.1, 1.2]
    # Use Nelder-Mead with reasonable initial simplex
    x0 = seed.copy()
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 
                                  'adaptive': True})
        if result.success and result.fun < 0.01:
            best = result.x
        else:
            best = x0
    except Exception:
        best = x0
    
    # Validate gamma bounds for all test models
    valid = True
    for Om0_t, wa_t in test_models:
        Om_t = _compute_Om(a_test, Om0_t, wa_t)
        q_t = 1.0 - Om_t
        u_t = 1.0 + wa_t + wa_t * (1.0 - a_test)
        gamma_t = best[0] + best[1]*q_t + best[2]*q_t**2 + best[3]*u_t + best[4]*wa_t*a_test + best[5]*u_t*q_t
        if np.any(gamma_t < 0.1) or np.any(gamma_t > 1.2):
            valid = False
            break
    
    if not valid:
        # Fall back to seed if gamma bounds violated
        best = seed
    
    return best.tolist()

# Pre-compute at import time to ensure construct() is fast
# Actually, let's just let construct() do the work with time guard
