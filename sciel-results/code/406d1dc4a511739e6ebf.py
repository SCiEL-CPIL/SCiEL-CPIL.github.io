"""Improved growth-index approximation via fast analytical reference and multi-start search."""
import time
import math
import numpy as np
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

def _fast_reference_f(a, Om0, wa, n_grid=400):
    """
    Fast computation of f(a) = d ln D / d ln a using a single ODE solve
    on a fine grid, then interpolation. This is called once per (Om0, wa) pair.
    """
    a = np.asarray(a, dtype=float)
    
    # Use a fine grid from min(a) to max(a), with special handling around a=1
    a_min = max(0.01, float(np.min(a)))
    a_max = min(10.0, float(np.max(a)))
    
    # Create fine integration grid
    if a_min < 1.0 and a_max > 1.0:
        # Split at a=1 for better resolution
        n_half = n_grid // 2
        a_grid_down = np.linspace(1.0, a_min, n_half)
        a_grid_up = np.linspace(1.0, a_max, n_half)
        a_grid = np.concatenate([a_grid_down[::-1], a_grid_up[1:]])
    elif a_min < 1.0:
        a_grid = np.linspace(a_min, 1.0, n_grid)
    else:
        a_grid = np.linspace(1.0, a_max, n_grid)
    
    # Use the ODE: df/da = f - f^2 - (3/2) Om(a)
    # Integrate from a=1 outward
    from scipy.integrate import solve_ivp
    
    def ode_rhs(a_val, f_val):
        Om_val = float(Om0 / (Om0 + (1.0 - Om0) * a_val**(3.0 * (1.0 + wa))))
        return [f_val[0] - f_val[0]**2 - 1.5 * Om_val]
    
    # Integrate from a=1 to a_min (backwards)
    if a_min < 1.0:
        a_int_down = np.linspace(1.0, a_min, n_grid)
        sol_down = solve_ivp(ode_rhs, [1.0, a_min], [Om0], t_eval=a_int_down,
                            method='RK45', rtol=1e-10, atol=1e-12, max_step=0.005)
        f_down = sol_down.y[0] if sol_down.success else np.full_like(a_int_down, Om0)
    else:
        a_int_down = np.array([1.0])
        f_down = np.array([Om0])
    
    # Integrate from a=1 to a_max (forwards)
    if a_max > 1.0:
        a_int_up = np.linspace(1.0, a_max, n_grid)
        sol_up = solve_ivp(ode_rhs, [1.0, a_max], [Om0], t_eval=a_int_up,
                          method='RK45', rtol=1e-10, atol=1e-12, max_step=0.005)
        f_up = sol_up.y[0] if sol_up.success else np.full_like(a_int_up, Om0)
    else:
        a_int_up = np.array([1.0])
        f_up = np.array([Om0])
    
    # Combine and interpolate
    all_a = np.concatenate([a_int_down, a_int_up[1:]])
    all_f = np.concatenate([f_down, f_up[1:]])
    
    # Sort by a for interpolation
    sort_idx = np.argsort(all_a)
    all_a_sorted = all_a[sort_idx]
    all_f_sorted = all_f[sort_idx]
    
    f_ref = np.interp(a, all_a_sorted, all_f_sorted)
    return f_ref

def _fast_D_normalized(a, f_vals):
    """
    Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    using cumulative trapezoidal integration on a log-spaced grid.
    """
    a = np.asarray(a, dtype=float)
    f_vals = np.asarray(f_vals, dtype=float)
    
    # Work in x = ln(a) space
    # Create a fine grid in x-space covering all a values
    x_min = math.log(max(0.01, float(np.min(a))))
    x_max = math.log(min(10.0, float(np.max(a))))
    
    n_grid = 800
    x_grid = np.linspace(x_min, x_max, n_grid)
    a_grid = np.exp(x_grid)
    
    # Interpolate f to the grid
    f_grid = np.interp(a_grid, a, f_vals)
    
    # Compute cumulative integral of f(exp(x)) from x to x_max=0 (i.e., a=1)
    # integral from x to 0 of f(exp(t)) dt = -integral from 0 to x of f(exp(t)) dt
    # So D(a)/D(1) = exp(-integral_{ln(a)}^{0} f(exp(x)) dx)
    
    # Use cumulative trapezoidal rule
    dx = x_grid[1] - x_grid[0]
    # Cumulative integral from x_min to each point
    cum_int = np.zeros(n_grid)
    for i in range(1, n_grid):
        cum_int[i] = cum_int[i-1] + 0.5 * (f_grid[i-1] + f_grid[i]) * dx
    
    # integral from x to 0 = cum_int[0] - cum_int at position where x_grid=0
    # Find index where x_grid is closest to 0
    idx_zero = np.argmin(np.abs(x_grid))
    integral_from_x_to_0 = cum_int[idx_zero] - cum_int
    
    # Interpolate to get values at actual a points
    x_actual = np.log(a)
    # We need integral from x_actual to 0
    # integral_from_x_to_0 is defined on x_grid
    # For x_actual < x_grid[0], extrapolate; for x_actual > x_grid[-1], extrapolate
    integral_actual = np.interp(x_actual, x_grid, integral_from_x_to_0,
                                left=integral_from_x_to_0[0], right=integral_from_x_to_0[-1])
    
    D_norm = np.exp(-integral_actual)
    return D_norm

def _evaluate_model_fast(coeffs, Om0, wa, a_test, f_ref_cache=None):
    """
    Evaluate the model with given coefficients. Returns max relative error.
    f_ref_cache: if provided, (f_ref, D_ref) precomputed for this (Om0, wa).
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
    
    # Get reference (use cache if available)
    if f_ref_cache is not None:
        f_ref, D_ref = f_ref_cache
    else:
        f_ref = _fast_reference_f(a_test, Om0, wa)
        D_ref = _fast_D_normalized(a_test, f_ref)
    
    # Relative error in f
    rel_err_f = np.abs(f_model - f_ref) / np.maximum(np.abs(f_ref), 1e-15)
    max_err_f = np.max(rel_err_f)
    
    # D(a)/D(1)
    D_model = _fast_D_normalized(a_test, f_model)
    
    rel_err_D = np.abs(D_model - D_ref) / np.maximum(np.abs(D_ref), 1e-15)
    max_err_D = np.max(rel_err_D)
    
    return max(max_err_f, max_err_D)

def _combined_objective(coeffs, test_models, a_test, ref_cache):
    """
    Combined objective: max error across all test models.
    ref_cache: dict mapping (Om0, wa) -> (f_ref, D_ref)
    """
    max_err = 0.0
    for Om0, wa in test_models:
        key = (round(Om0, 6), round(wa, 6))
        if key not in ref_cache:
            f_ref = _fast_reference_f(a_test, Om0, wa)
            D_ref = _fast_D_normalized(a_test, f_ref)
            ref_cache[key] = (f_ref, D_ref)
        
        err = _evaluate_model_fast(coeffs, Om0, wa, a_test, ref_cache[key])
        if err > max_err:
            max_err = err
        if err >= 1e6:
            return err
    return max_err

def construct():
    """
    Find optimal six coefficients for the growth-index formula.
    Returns [c0, c1, c2, c3, c4, c5].
    """
    # Test cosmologies representative of validation space
    test_models = [
        (0.30, 0.0),    # LCDM
        (0.30, -0.5),   # Phantom-like
        (0.30, 0.5),    # Quintessence-like
        (0.25, 0.0),    # Lower Om0
        (0.35, 0.0),    # Higher Om0
        (0.28, -0.3),   # Another combination
        (0.32, 0.3),    # Another combination
    ]
    
    # Test scale factors - include a range
    a_test = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Seed
    seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Pre-compute references for all test models
    ref_cache = {}
    for Om0, wa in test_models:
        key = (round(Om0, 6), round(wa, 6))
        if _time_remaining() < 5.0:
            break
        try:
            f_ref = _fast_reference_f(a_test, Om0, wa)
            D_ref = _fast_D_normalized(a_test, f_ref)
            ref_cache[key] = (f_ref, D_ref)
        except Exception:
            pass
    
    if _time_remaining() < 3.0:
        return seed.tolist()
    
    # Multi-start Nelder-Mead optimization
    best_coeffs = seed.copy()
    best_score = _combined_objective(seed, test_models, a_test, ref_cache)
    
    # Multiple starting points
    np.random.seed(42)
    starts = [seed.copy()]
    # Add perturbed starts around seed
    for _ in range(8):
        perturbation = np.random.uniform(-0.15, 0.15, size=6)
        perturbation[0] = np.random.uniform(-0.05, 0.15)  # c0 stays near 0.55
        starts.append(seed + perturbation)
    # Add some broader starts
    for _ in range(4):
        s = np.random.uniform(-0.5, 0.5, size=6)
        s[0] = np.random.uniform(0.4, 0.7)
        starts.append(s)
    
    for i, x0 in enumerate(starts):
        if _time_remaining() < 5.0:
            break
        
        def obj(coeffs):
            return _combined_objective(coeffs, test_models, a_test, ref_cache)
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9,
                                      'adaptive': True, 'initial_simplex': None})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final refinement on best
    if _time_remaining() > 10.0:
        def obj_final(coeffs):
            return _combined_objective(coeffs, test_models, a_test, ref_cache)
        
        try:
            result = minimize(obj_final, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Validate gamma bounds for a broader set of models
    validation_models = [
        (0.25, -0.5), (0.25, 0.0), (0.25, 0.5),
        (0.30, -0.5), (0.30, 0.0), (0.30, 0.5),
        (0.35, -0.5), (0.35, 0.0), (0.35, 0.5),
    ]
    a_validate = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    
    valid = True
    for Om0_v, wa_v in validation_models:
        Om_v = _compute_Om(a_validate, Om0_v, wa_v)
        q_v = 1.0 - Om_v
        u_v = 1.0 + wa_v + wa_v * (1.0 - a_validate)
        gamma_v = (best_coeffs[0] + best_coeffs[1]*q_v + best_coeffs[2]*q_v**2 + 
                   best_coeffs[3]*u_v + best_coeffs[4]*wa_v*a_validate + best_coeffs[5]*u_v*q_v)
        if np.any(gamma_v < 0.1) or np.any(gamma_v > 1.2):
            valid = False
            break
    
    if not valid:
        # Fall back to seed
        best_coeffs = seed
    
    # Clip coefficients to [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    return best_coeffs.tolist()
