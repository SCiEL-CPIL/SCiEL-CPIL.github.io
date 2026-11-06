"""Improved program for cosmology growth-index optimization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

# EVOLVE-BLOCK-START
def _H2_of_a(a, Om0, w0, wa):
    """Hubble parameter squared normalized to H0^2 for CPL dark energy."""
    Om_m = Om0
    Om_de = 1.0 - Om0
    if Om_de < 0:
        Om_de = 1e-10
    a_safe = np.maximum(a, 1e-10)
    term_m = Om_m * a_safe ** (-3)
    exponent = -3.0 * (1.0 + w0 + wa)
    exp_factor = 3.0 * wa * (1.0 / a_safe - 1.0)
    term_de = Om_de * (a_safe ** exponent) * np.exp(exp_factor)
    return term_m + term_de

def _Om_m_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    H2 = _H2_of_a(a, Om0, w0, wa)
    return Om0 * a ** (-3) / H2

def _precompute_references(models, a_values, num_steps=5000):
    """
    Pre-compute true f(a) and D(a) for all models and a values.
    Returns a list of dicts, each containing 'f_true' and 'D_true' arrays.
    """
    refs = []
    for (Om0, w0, wa) in models:
        a_min = np.min(a_values)
        x_grid = np.linspace(0, np.log(a_min), num_steps + 1)
        a_grid = np.exp(x_grid)
        
        f_vals = np.zeros_like(x_grid)
        f_vals[0] = 0.5
        ln_D_vals = np.zeros_like(x_grid)
        ln_D_vals[0] = 0.0
        
        dx = x_grid[1] - x_grid[0]
        
        for i in range(num_steps):
            a_curr = a_grid[i]
            Om_m_curr = _Om_m_of_a(a_curr, Om0, w0, wa)
            f_curr = f_vals[i]
            df_dx = -f_curr * (2.0 + f_curr) + 1.5 * Om_m_curr
            f_vals[i+1] = f_curr + df_dx * dx
            ln_D_vals[i+1] = ln_D_vals[i] + f_curr * dx
        
        x_targets = np.log(a_values)
        f_true = np.interp(x_targets, x_grid, f_vals)
        ln_D_true = np.interp(x_targets, x_grid, ln_D_vals)
        D_true = np.exp(ln_D_true)
        
        refs.append({
            'f_true': f_true,
            'D_true': D_true,
            'Om_m': _Om_m_of_a(a_values, Om0, w0, wa),
            'w0': w0,
            'wa': wa
        })
    return refs

def _compute_D_app(coeffs, a_values, Om_m, w0, wa):
    """
    Compute normalized growth factor D(a)/D(1) using the approximation.
    Uses cumulative trapezoidal integration in x = ln(a) space.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a_values)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_values + c5*u*q
    f_app = Om_m ** gamma
    
    x_vals = np.log(a_values)
    
    # We need to integrate from x=0 (a=1) to each x_i.
    # Prepend x=0 with f value at a=1.
    # At a=1: Om_m(1) = Om0, so q=1-Om0, u=1+w0
    Om0 = Om_m[-1]  # a_values is sorted descending, so last element is closest to 1
    q0 = 1.0 - Om0
    u0 = 1.0 + w0
    gamma0 = c0 + c1*q0 + c2*q0**2 + c3*u0 + c4*wa*1.0 + c5*u0*q0
    f0 = Om0 ** gamma0
    
    x_ext = np.concatenate([[0.0], x_vals])
    f_ext = np.concatenate([[f0], f_app])
    
    # Integrate using trapezoidal rule
    integrals = cumulative_trapezoid(f_ext, x_ext, initial=0.0)
    ln_D_app = integrals[1:]  # Skip the first point (x=0)
    D_app = np.exp(ln_D_app)
    
    return D_app

def _local_error(coeffs, refs, a_values):
    """
    Calculate the maximum relative error for both f and D.
    """
    max_err = 0.0
    c0, c1, c2, c3, c4, c5 = coeffs
    
    for ref in refs:
        f_true = ref['f_true']
        D_true = ref['D_true']
        Om_m = ref['Om_m']
        w0 = ref['w0']
        wa = ref['wa']
        
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a_values)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_values + c5*u*q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_app = Om_m ** gamma
        err_f = np.abs(f_app - f_true) / np.abs(f_true)
        
        D_app = _compute_D_app(coeffs, a_values, Om_m, w0, wa)
        err_D = np.abs(D_app - D_true) / np.abs(D_true)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
        
    return 100.0 * max_err

def _cosmology_setup():
    models = [
        (0.30, -1.0, 0.0),
        (0.28, -1.1, 0.4),
        (0.32, -0.9, 0.1),
        (0.25, -1.2, 0.5),
        (0.30, -0.8, -0.5),
        (0.27, -1.05, 0.3),
        (0.33, -0.95, -0.2),
        (0.31, -1.0, 0.2),
        (0.29, -1.05, -0.3),
        (0.30, -0.95, 0.35),
    ]
    return models

def _check_gamma_bounds(coeffs, refs, a_values):
    """Check if gamma stays within [0.1, 1.2] for all models and a values."""
    c0, c1, c2, c3, c4, c5 = coeffs
    for ref in refs:
        Om_m = ref['Om_m']
        w0 = ref['w0']
        wa = ref['wa']
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a_values)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_values + c5*u*q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
    return True

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    models = _cosmology_setup()
    a_values = np.geomspace(0.1, 0.99, 25)
    
    refs = _precompute_references(models, a_values, num_steps=5000)
    
    # Better initial guess based on known growth index behavior
    # For flat LCDM, gamma ≈ 0.55
    # Allow some variation with q and u
    x0 = np.array([0.55, 0.1, -0.05, 0.05, 0.0, 0.0])
    
    best_score = _local_error(x0, refs, a_values)
    best_params = x0.copy()
    
    # Multiple restarts with different initial guesses
    initial_guesses = [
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.1, -0.05, 0.05, 0.0, 0.0]),
        np.array([0.54, 0.15, -0.1, 0.1, 0.05, 0.0]),
        np.array([0.56, 0.05, -0.02, 0.03, -0.05, 0.02]),
        np.array([0.53, 0.2, -0.15, 0.12, 0.1, -0.05]),
    ]
    
    for x_init in initial_guesses:
        if time.time() - start_time > time_limit - 15.0:
            break
        try:
            res = minimize(
                _local_error, 
                x_init, 
                args=(refs, a_values), 
                method='Nelder-Mead', 
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 400, 'maxfev': 500}
            )
            if res.fun < best_score and _check_gamma_bounds(res.x, refs, a_values):
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - start_time < time_limit - 10.0 and best_score < 1e10:
        try:
            res2 = minimize(
                _local_error, 
                best_params, 
                args=(refs, a_values), 
                method='Nelder-Mead', 
                options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 600, 'maxfev': 800}
            )
            if res2.fun < best_score and _check_gamma_bounds(res2.x, refs, a_values):
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Ensure coefficients are within [-10, 10]
    best_params = np.clip(best_params, -10.0, 10.0)
    
    # Final validation
    if not _check_gamma_bounds(best_params, refs, a_values):
        best_params = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return [float(x) for x in best_params]

# EVOLVE-BLOCK-END
