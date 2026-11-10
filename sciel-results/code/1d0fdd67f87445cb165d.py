"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = 85.0  # seconds, with safety margin before 90s limit

# Precomputed reference data cache
_ref_cache = {}

def _Om_of_a(a, Om0, w0, wa):
    """Compute Omega_matter(a) for CPL dark energy."""
    if a == 1.0:
        return Om0
    
    def integrand_de(x):
        return (1.0 + w0 + wa * (1.0 - x)) / x
    
    if a < 1.0:
        integral_de, _ = quad(integrand_de, a, 1.0, limit=100)
    else:
        integral_de, _ = quad(integrand_de, 1.0, a, limit=100)
    
    ratio_de = np.exp(-3.0 * integral_de)
    ratio_m = a ** (-3.0)
    
    Om_m_1 = Om0
    Om_de_1 = 1.0 - Om0
    
    rho_m_a = Om_m_1 * ratio_m
    rho_de_a = Om_de_1 * ratio_de
    
    return rho_m_a / (rho_m_a + rho_de_a)

def _f_reference(a, Om0, w0, wa):
    """Reference growth rate from ODE with caching."""
    key = (round(a, 12), round(Om0, 12), round(w0, 12), round(wa, 12))
    if key in _ref_cache:
        return _ref_cache[key]
    
    Om_a = _Om_of_a(a, Om0, w0, wa)
    
    def ode_rhs(lna, y):
        a_local = np.exp(lna)
        Om_local = _Om_of_a(a_local, Om0, w0, wa)
        u_local = 1.0 + w0 + wa * (1.0 - a_local)
        dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
        Hprime_over_H = 0.5 * dlnH2_dlna
        f_local = y[0]
        df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
        return [df_dlna]
    
    lna_target = np.log(a)
    lna_start = -5.0
    y0 = [2.0/3.0]
    
    sol = solve_ivp(ode_rhs, [lna_start, lna_target], y0, method='RK45', rtol=1e-8, atol=1e-10)
    if sol.success:
        result = sol.y[0, -1]
        _ref_cache[key] = result
        return result
    else:
        return 0.55

def _D_normalized(a, Om0, w0, wa, coeffs):
    """Compute D(a)/D(1) using the approximate gamma formula."""
    c0, c1, c2, c3, c4, c5 = coeffs
    if a == 1.0:
        return 1.0
    
    def integrand(x):
        a_local = np.exp(x)
        Om_local = _Om_of_a(a_local, Om0, w0, wa)
        q_local = 1.0 - Om_local
        u_local = 1.0 + w0 + wa * (1.0 - a_local)
        gamma_local = c0 + c1 * q_local + c2 * q_local**2 + c3 * u_local + c4 * wa * a_local + c5 * u_local * q_local
        if gamma_local < 0.1 or gamma_local > 1.2:
            return 0.55
        return Om_local ** gamma_local
    
    lna_target = np.log(a)
    if lna_target < 0:
        integral, _ = quad(integrand, lna_target, 0.0, limit=100)
        return np.exp(integral)
    else:
        integral, _ = quad(integrand, 0.0, lna_target, limit=100)
        return np.exp(-integral)

def _D_reference(a, Om0, w0, wa):
    """Reference D(a)/D(1) using constant gamma=0.55, with caching."""
    key = ('D', round(a, 12), round(Om0, 12), round(w0, 12), round(wa, 12))
    if key in _ref_cache:
        return _ref_cache[key]
    
    result = _D_normalized(a, Om0, w0, wa, [0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    _ref_cache[key] = result
    return result

def _compute_errors(coeffs, cosmologies, a_values):
    """Compute max relative error for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_rel_err = 0.0
    
    for Om0, w0, wa in cosmologies:
        for a in a_values:
            Om_a = _Om_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_ref = _f_reference(a, Om0, w0, wa)
            
            if abs(f_ref) > 1e-10:
                rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                max_rel_err = max(max_rel_err, rel_err_f)
            
            D_ref = _D_reference(a, Om0, w0, wa)
            D_approx = _D_normalized(a, Om0, w0, wa, coeffs)
            
            if abs(D_ref) > 1e-10:
                rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_rel_err = max(max_rel_err, rel_err_D)
    
    return 100.0 * max_rel_err

def _compute_errors_smooth(coeffs, cosmologies, a_values, beta=50.0):
    """Compute smooth surrogate using log-sum-exp for the max over errors."""
    c0, c1, c2, c3, c4, c5 = coeffs
    errors = []
    
    for Om0, w0, wa in cosmologies:
        for a in a_values:
            Om_a = _Om_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_ref = _f_reference(a, Om0, w0, wa)
            
            if abs(f_ref) > 1e-10:
                rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                errors.append(rel_err_f)
            
            D_ref = _D_reference(a, Om0, w0, wa)
            D_approx = _D_normalized(a, Om0, w0, wa, coeffs)
            
            if abs(D_ref) > 1e-10:
                rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                errors.append(rel_err_D)
    
    if not errors:
        return 0.0
    
    errors_arr = np.array(errors)
    # Log-sum-exp for smooth max
    max_err = np.max(errors_arr)
    if max_err < 1e-10:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (errors_arr - max_err))))
    smooth_max = max_err + np.log(np.sum(np.exp(beta * (errors_arr - max_err)))) / beta
    return 100.0 * smooth_max

def _objective(coeffs):
    """Objective function for optimization."""
    if time.time() > _DEADLINE:
        return 1e10
    
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
    ]
    
    a_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    
    return _compute_errors(coeffs, cosmologies, a_values)

def _objective_smooth(coeffs):
    """Smooth objective for optimization."""
    if time.time() > _DEADLINE:
        return 1e10
    
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
    ]
    
    a_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    
    return _compute_errors_smooth(coeffs, cosmologies, a_values)

def _validate_coeffs(coeffs):
    """Check if coefficients are valid (in range and gamma constraints)."""
    if any(c < -10 or c > 10 for c in coeffs):
        return False
    return True

def construct():
    """Construct the six coefficients."""
    global _START_TIME, _DEADLINE
    
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = _objective(best_coeffs)
    
    # Phase 1: Smooth optimization with Nelder-Mead
    try:
        result = minimize(_objective_smooth, best_coeffs, method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7,
                                 'adaptive': True})
        
        if result.fun < best_score and _validate_coeffs(result.x):
            c = list(result.x)
            # Verify with exact objective
            exact_score = _objective(c)
            if exact_score < best_score:
                best_coeffs = c
                best_score = exact_score
    except Exception:
        pass
    
    if time.time() > _DEADLINE - 15.0:
        return best_coeffs
    
    # Phase 2: Multi-start with smooth objective
    start_points = [
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
    ]
    
    for sp in start_points:
        if time.time() > _DEADLINE - 20.0:
            break
        
        try:
            result = minimize(_objective_smooth, sp, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7,
                                     'adaptive': True})
            
            if result.fun < best_score and _validate_coeffs(result.x):
                c = list(result.x)
                exact_score = _objective(c)
                if exact_score < best_score:
                    best_coeffs = c
                    best_score = exact_score
        except Exception:
            pass
    
    if time.time() > _DEADLINE - 10.0:
        return best_coeffs
    
    # Phase 3: High-precision polish with exact objective
    try:
        result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9,
                                 'adaptive': True})
        
        if result.fun < best_score and _validate_coeffs(result.x):
            c = list(result.x)
            best_coeffs = c
            best_score = result.fun
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
