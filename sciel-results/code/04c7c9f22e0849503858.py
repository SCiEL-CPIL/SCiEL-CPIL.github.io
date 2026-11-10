"""Improved growth-index fitting with Taylor-based initialization and smooth objective."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, cumulative_trapezoid

START_TIME = time.time()
DEADLINE = 82.0  # 8s margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    if a <= 0:
        return 1.0
    de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
    om_m = Om0 * a**(-3.0)
    om_de = (1.0 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _compute_reference_data():
    """
    Pre-compute reference f(a) and D(a) for a grid of cosmologies and a-values.
    """
    cosmos = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -0.8, 0.3),
        (0.35, -1.2, 0.2),
        (0.28, -0.95, -0.2),
        (0.32, -1.05, 0.4),
        (0.30, -0.95, 0.1),
        (0.27, -1.1, 0.3),
    ]
    
    a_vals = np.geomspace(0.05, 1.0, 60)
    ref_data = []
    
    for Om0, w0, wa in cosmos:
        f_refs = np.zeros_like(a_vals)
        D_refs = np.zeros_like(a_vals)
        
        def ode(t, y):
            a = np.exp(t)
            om_a = _Om_of_a(a, Om0, w0, wa)
            f_val = y[0]
            D_val = y[1]
            df_dlna = f_val - f_val**2 - 1.5 * om_a
            dD_dlna = -f_val * D_val
            return [df_dlna, dD_dlna]
        
        f_init = Om0 ** 0.55
        D_init = 1.0
        t_end = np.log(a_vals[0])
        
        sol = solve_ivp(
            ode, [0.0, t_end], [f_init, D_init],
            method='RK45',
            rtol=1e-10,
            atol=1e-13,
            dense_output=True
        )
        
        if not sol.success:
            continue
        
        for i, a in enumerate(a_vals):
            t = np.log(a)
            y = sol.sol(t)
            f_refs[i] = y[0]
            D_refs[i] = y[1]
        
        ref_data.append((Om0, w0, wa, a_vals, f_refs, D_refs))
    
    return ref_data

def _taylor_init():
    """
    Fit coefficients to first-order Taylor expansion of exact gamma around central cosmology.
    gamma_exact(a) = ln(f_exact(a)) / ln(Om(a))
    We approximate this with our ansatz and match derivatives.
    """
    Om0_c, w0_c, wa_c = 0.30, -1.0, 0.0
    a_vals = np.geomspace(0.1, 0.99, 30)
    
    def ode(t, y):
        a = np.exp(t)
        om_a = _Om_of_a(a, Om0_c, w0_c, wa_c)
        return [y[0] - y[0]**2 - 1.5*om_a, -y[0]*y[1]]
    
    sol = solve_ivp(ode, [0, np.log(0.1)], [Om0_c**0.55, 1.0],
                    method='RK45', rtol=1e-10, atol=1e-13, dense_output=True)
    
    if not sol.success:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    om_a = np.array([_Om_of_a(a, Om0_c, w0_c, wa_c) for a in a_vals])
    q = 1.0 - om_a
    u = 1.0 + w0_c + wa_c*(1.0 - a_vals)
    
    f_exact = np.array([sol.sol(np.log(a))[0] for a in a_vals])
    gamma_exact = np.log(f_exact) / np.log(om_a)
    
    # Build design matrix for linear fit: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    X = np.column_stack([
        np.ones_like(a_vals),
        q,
        q**2,
        u,
        wa_c * a_vals,
        u * q
    ])
    
    coeffs, _, _, _ = np.linalg.lstsq(X, gamma_exact, rcond=None)
    return np.clip(coeffs, -10, 10)

def _error_function(coeffs, ref_data, smooth=False):
    """
    Compute error. If smooth=True, use soft-max instead of hard max.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    all_errors = []
    
    for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
        om_a_vals = np.array([_Om_of_a(a, Om0, w0, wa) for a in a_vals])
        
        if np.any(om_a_vals <= 0) or np.any(om_a_vals >= 1):
            return 1e10
            
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_approx = om_a_vals ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs(f_approx - f_refs) / np.abs(f_refs)
            rel_err_f = np.nan_to_num(rel_err_f, nan=1e10, posinf=1e10, neginf=1e10)
            
        all_errors.extend(rel_err_f.tolist())
        
        x = np.log(a_vals)
        cum_int = cumulative_trapezoid(f_approx, x, initial=0.0)
        integrals = cum_int[-1] - cum_int
        D_approx = np.exp(-integrals)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
            rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
            
        all_errors.extend(rel_err_D.tolist())
    
    errors = np.array(all_errors)
    if smooth:
        # Soft-max via log-sum-exp
        max_e = np.max(errors)
        if max_e > 50:
            return max_e
        return max_e + np.log(np.sum(np.exp(errors - max_e)))
    else:
        return np.max(errors)

def _optimize():
    if time.time() > DEADLINE - 25:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    ref_data = _compute_reference_data()
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Taylor-based initial guess
    x0_taylor = _taylor_init()
    x0_const = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Evaluate both
    err_taylor = _error_function(x0_taylor, ref_data)
    err_const = _error_function(x0_const, ref_data)
    
    if err_taylor < err_const:
        best_x = x0_taylor
        best_err = err_taylor
    else:
        best_x = x0_const
        best_err = err_const
    
    if time.time() > DEADLINE - 18:
        return np.clip(best_x, -10, 10)
    
    bounds = [(-10, 10)] * 6
    
    # Differential evolution with smooth objective for better gradient info
    try:
        def de_callback(xk, convergence=None):
            return time.time() > DEADLINE - 12
        
        de_result = differential_evolution(
            _error_function, bounds, args=(ref_data, True),
            maxiter=40,
            popsize=15,
            tol=1e-6,
            seed=42,
            init='sobol',
            callback=de_callback,
            polish=False
        )
        if de_result.fun < best_err:
            # Re-evaluate with hard max
            hard_err = _error_function(de_result.x, ref_data, smooth=False)
            if hard_err < best_err:
                best_err = hard_err
                best_x = de_result.x
    except Exception:
        pass
    
    if time.time() > DEADLINE - 10:
        return np.clip(best_x, -10, 10)
    
    # Local polish with Nelder-Mead
    try:
        res = minimize(
            _error_function, best_x, args=(ref_data, False),
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 300, 'maxfev': 800}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() > DEADLINE - 5:
        return np.clip(best_x, -10, 10)
    
    # Second polish
    try:
        res2 = minimize(
            _error_function, best_x, args=(ref_data, False),
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 200, 'maxfev': 500}
        )
        if res2.fun < best_err:
            best_err = res2.fun
            best_x = res2.x
    except Exception:
        pass
    
    return np.clip(best_x, -10, 10)

# Run optimization at module load
try:
    _coeffs = _optimize().tolist()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

if not all(np.isfinite(c) for c in _coeffs):
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
