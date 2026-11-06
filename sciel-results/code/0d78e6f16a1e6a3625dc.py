"""Improved program for cosmology growth-index optimization."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _H2_of_a(a, Om0, w0, wa):
    """Hubble parameter squared normalized to H0^2 for CPL dark energy."""
    Om_m = Om0
    Om_de = 1.0 - Om0
    if Om_de < 0:
        Om_de = 1e-10
    a_safe = np.maximum(a, 1e-10)
    term_m = Om_m * a_safe ** (-3)
    # Exact CPL dark energy density: rho_DE ~ a^{-3(1+w0+wa)} * exp(3*wa*(1/a - 1))
    exponent = -3.0 * (1.0 + w0 + wa)
    exp_factor = 3.0 * wa * (1.0 / a_safe - 1.0)
    term_de = Om_de * (a_safe ** exponent) * np.exp(exp_factor)
    return term_m + term_de

def _Om_m_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    H2 = _H2_of_a(a, Om0, w0, wa)
    return Om0 * a ** (-3) / H2

def _compute_true_f(a, Om0, w0, wa, f_start=0.5):
    """
    Compute the 'true' growth rate f(a) by integrating the ODE:
    df/dlna = -f(2+f) + 1.5 * Om_m(a)
    backwards from a=1 to a.
    """
    if a >= 1.0:
        return f_start
    
    # Use adaptive step size or fixed small steps for accuracy
    # We integrate from 1.0 down to a.
    # Let x = ln(a). We integrate from x=0 to x=ln(a).
    # dx = da/a.
    # df/dx = -f(2+f) + 1.5 * Om_m(a)
    
    # Simple Euler with small steps is usually sufficient for this smooth function.
    # Number of steps proportional to distance in ln(a).
    x_start = 0.0
    x_end = np.log(a)
    dx = (x_end - x_start) / 1000.0
    
    f_curr = f_start
    x_curr = x_start
    
    for i in range(1000):
        a_curr = np.exp(x_curr)
        Om_m_curr = _Om_m_of_a(a_curr, Om0, w0, wa)
        df_dx = -f_curr * (2.0 + f_curr) + 1.5 * Om_m_curr
        f_curr += df_dx * dx
        x_curr += dx
        
    return f_curr

def _compute_D_ratio(a, Om0, w0, wa, coeffs):
    """
    Compute D(a)/D(1) = exp(integral_1^a f_app(x) dlnx)
    and true D ratio for error calculation.
    """
    if a >= 1.0:
        return 1.0, 1.0
    
    c0, c1, c2, c3, c4, c5 = coeffs
    
    x_start = 0.0
    x_end = np.log(a)
    num_steps = 1000
    dx = (x_end - x_start) / num_steps
    
    integral_app = 0.0
    integral_true = 0.0
    f_true_curr = 0.5
    
    for i in range(num_steps):
        x_curr = x_start + i * dx
        a_curr = np.exp(x_curr)
        
        # True f update
        Om_m_curr = _Om_m_of_a(a_curr, Om0, w0, wa)
        df_dx_true = -f_true_curr * (2.0 + f_true_curr) + 1.5 * Om_m_curr
        f_true_curr_new = f_true_curr + df_dx_true * dx
        
        # Approx f
        Om_a_curr = Om_m_curr
        q = 1.0 - Om_a_curr
        u = 1.0 + w0 + wa * (1.0 - a_curr)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_curr + c5*u*q
        
        # Enforce gamma bounds softly for the integral? 
        # The task says gamma must lie in [0.1, 1.2]. If it doesn't, the program is invalid.
        # For the error calculation, we assume valid gamma.
        
        f_app_curr = Om_a_curr ** gamma
        
        # Trapezoidal rule or simple sum? Simple sum (Left Riemann) is fine with 1000 steps.
        integral_app += f_app_curr * dx
        integral_true += f_true_curr * dx
        
        f_true_curr = f_true_curr_new
    
    D_app = np.exp(integral_app)
    D_true = np.exp(integral_true)
    
    return D_app, D_true

def _local_error(coeffs, models, a_values):
    """
    Calculate the maximum relative error for f and D over the given models and a values.
    """
    max_err = 0.0
    c0, c1, c2, c3, c4, c5 = coeffs
    
    for (Om0, w0, wa) in models:
        for a in a_values:
            if a >= 1.0:
                continue
                
            # Compute true f(a)
            f_true = _compute_true_f(a, Om0, w0, wa)
            
            # Compute approximation f(a)
            Om_a = _Om_m_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            # Check gamma bounds
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
                
            f_app = Om_a ** gamma
            
            # Error in f
            if abs(f_true) > 1e-8:
                err_f = abs(f_app - f_true) / abs(f_true)
            else:
                err_f = abs(f_app - f_true)
                
            # Error in D
            D_app, D_true = _compute_D_ratio(a, Om0, w0, wa, coeffs)
            
            if abs(D_true) > 1e-8:
                err_D = abs(D_app - D_true) / abs(D_true)
            else:
                err_D = abs(D_app - D_true)
                
            err = max(err_f, err_D)
            if err > max_err:
                max_err = err
                
    return 100.0 * max_err

def _cosmology_setup():
    """
    Setup a set of validation cosmologies. 
    """
    models = [
        (0.30, -1.0, 0.0),
        (0.28, -1.1, 0.4),
        (0.32, -0.9, 0.1),
        (0.25, -1.2, 0.5),
        (0.30, -0.8, -0.5),
        (0.27, -1.05, 0.3),
        (0.33, -0.95, -0.2),
    ]
    return models

def construct():
    start_time = time.time()
    time_limit = 80.0 # Safety margin (90s total, leave 10s for safety)
    
    models = _cosmology_setup()
    # Sample a values: logarithmic spacing is better for growth
    a_values = np.geomspace(0.1, 0.99, 15)
    
    # Initial guess: The seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_score = _local_error(x0, models, a_values)
    best_params = x0
    
    # Optimization
    if time.time() - start_time < time_limit - 10.0:
        try:
            # Use Nelder-Mead
            res = minimize(
                _local_error, 
                x0, 
                args=(models, a_values), 
                method='Nelder-Mead', 
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Polish with tighter tolerance if time remains
    if time.time() - start_time < time_limit - 5.0 and best_score < 1e10:
        try:
            res2 = minimize(
                _local_error, 
                best_params, 
                args=(models, a_values), 
                method='Nelder-Mead', 
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Final check on gamma bounds for the best params
    c0, c1, c2, c3, c4, c5 = best_params
    valid = True
    for (Om0, w0, wa) in models:
        for a in a_values:
            Om_a = _Om_m_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            if gamma < 0.1 or gamma > 1.2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to seed if best params are invalid
        best_params = x0
    
    return [float(x) for x in best_params]

# EVOLVE-BLOCK-END
