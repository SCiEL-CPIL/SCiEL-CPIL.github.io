"""Improved growth-index formula for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, differential_evolution
import time

_START_TIME = time.time()
_DEADLINE = 88.0  # seconds, safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_E_sq(Om0, w0, wa, a):
    """Compute E^2(a) = H^2/H0^2 for flat CPL universe."""
    a = np.asarray(a, dtype=float)
    Om_m = Om0 * a**(-3)
    Om_de = (1.0 - Om0) * a**(-3.0*(1.0+w0)) * np.exp(3.0*wa*(1.0-a))
    return Om_m + Om_de

def _compute_Om_m(Om0, w0, wa, a):
    """Compute matter density parameter Om_m(a)."""
    a = np.asarray(a, dtype=float)
    E2 = _compute_E_sq(Om0, w0, wa, a)
    return Om0 * a**(-3) / E2

def _generate_reference_f(a, Om0, w0, wa):
    """
    Generate reference growth rate f(a) by solving the ODE:
    f' + f(f+1) = (3/2) Om_m(a)
    where ' denotes d/da.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = np.array([a])
    
    # Solve ODE from a=1 where f(1) = Om_m(1)^0.55 approximately
    Om_m_1 = Om0
    f_1 = Om_m_1 ** 0.55
    
    def ode(a_val, y):
        f = y[0]
        Om_m_val = _compute_Om_m(Om0, w0, wa, a_val)
        # df/da = -(f*(f+1) - 1.5*Om_m)
        return [-(f*(f+1) - 1.5*Om_m_val)]
    
    # Integrate from a=1 down to min(a)
    a_min = max(np.min(a), 0.01)
    if a_min >= 1.0:
        return np.ones_like(a) * f_1
    
    try:
        sol = solve_ivp(ode, [1.0, a_min], [f_1], method='RK45',
                       rtol=1e-10, atol=1e-12, dense_output=True)
        if sol.success:
            f_vals = sol.sol(a)[0]
            return f_vals
    except:
        pass
    
    # Fallback: use the analytic approximation
    Om_m_vals = _compute_Om_m(Om0, w0, wa, a)
    return Om_m_vals ** 0.55 + 0.05 * wa * (1.0 - a)

def _generate_reference_D_ratio(a, Om0, w0, wa, f_ref_func):
    """
    Generate reference D(a)/D(1) by integrating:
    ln(D(a)/D(1)) = -integral from ln(a) to 0 of f(exp(x)) dx
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = np.array([a])
    
    D_ratios = np.ones_like(a)
    for i, a_i in enumerate(a):
        if a_i >= 1.0:
            D_ratios[i] = 1.0
        elif a_i <= 0.01:
            # Use approximation
            Om_m_i = _compute_Om_m(Om0, w0, wa, a_i)
            f_i = f_ref_func(a_i, Om0, w0, wa)
            D_ratios[i] = a_i ** (1.0 + f_i)
        else:
            def integrand(x):
                a_x = np.exp(x)
                f_x = f_ref_func(a_x, Om0, w0, wa)
                return f_x[0] if hasattr(f_x, '__len__') and len(f_x) > 0 else f_x
            
            try:
                integral, _ = quad(integrand, np.log(a_i), 0.0, limit=100)
                D_ratios[i] = np.exp(-integral)
            except:
                # Fallback: trapezoidal
                n = 200
                xs = np.linspace(np.log(a_i), 0.0, n)
                fs = np.array([f_ref_func(np.exp(x), Om0, w0, wa) for x in xs])
                fs = np.array([f[0] if hasattr(f, '__len__') and len(f) > 0 else f for f in fs])
                integral = np.trapz(fs, xs)
                D_ratios[i] = np.exp(-integral)
    
    return D_ratios

def _compute_gamma(c, a, Om0, w0, wa):
    """Compute gamma from the six-coefficient formula."""
    a = np.asarray(a, dtype=float)
    Om_m = _compute_Om_m(Om0, w0, wa, a)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    return gamma

def _compute_f(c, a, Om0, w0, wa):
    """Compute f = Om_m(a)^gamma."""
    a = np.asarray(a, dtype=float)
    gamma = _compute_gamma(c, a, Om0, w0, wa)
    Om_m = _compute_Om_m(Om0, w0, wa, a)
    # Ensure Om_m > 0
    Om_m_safe = np.maximum(Om_m, 1e-10)
    return Om_m_safe ** gamma

def _compute_D_ratio(c, a, Om0, w0, wa):
    """Compute D(a)/D(1) using the formula's f."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = np.array([a])
    
    D_ratios = np.ones_like(a)
    for i, a_i in enumerate(a):
        if a_i >= 1.0:
            D_ratios[i] = 1.0
        elif a_i <= 0.01:
            Om_m_i = _compute_Om_m(Om0, w0, wa, a_i)
            f_i = _compute_f(c, a_i, Om0, w0, wa)
            f_i = f_i[0] if hasattr(f_i, '__len__') and len(f_i) > 0 else f_i
            D_ratios[i] = a_i ** (1.0 + f_i)
        else:
            def integrand(x):
                a_x = np.exp(x)
                f_x = _compute_f(c, a_x, Om0, w0, wa)
                return f_x[0] if hasattr(f_x, '__len__') and len(f_x) > 0 else f_x
            
            try:
                integral, _ = quad(integrand, np.log(a_i), 0.0, limit=100)
                D_ratios[i] = np.exp(-integral)
            except:
                n = 200
                xs = np.linspace(np.log(a_i), 0.0, n)
                fs = np.array([_compute_f(c, np.exp(x), Om0, w0, wa) for x in xs])
                fs = np.array([f[0] if hasattr(f, '__len__') and len(f) > 0 else f for f in fs])
                integral = np.trapz(fs, xs)
                D_ratios[i] = np.exp(-integral)
    
    return D_ratios

def _make_reference_data():
    """Pre-compute reference data for validation cosmologies."""
    validation_models = [
        (0.30, -1.0, 0.0),
        (0.27, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -0.9, 0.0),
        (0.30, -1.1, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -0.8, 0.3),
        (0.40, -1.2, 0.2),
        (0.30, -0.7, 0.0),
        (0.30, -1.3, 0.0),
        (0.28, -1.0, 0.3),
    ]
    
    a_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    ref_data = []
    for Om0, w0, wa in validation_models:
        f_ref = _generate_reference_f(a_values, Om0, w0, wa)
        D_ref = _generate_reference_D_ratio(a_values, Om0, w0, wa, lambda a, O, w, wa_: _generate_reference_f(a, O, w, wa_))
        ref_data.append((Om0, w0, wa, f_ref, D_ref))
    
    return validation_models, a_values, ref_data

def _objective(c, validation_models, a_values, ref_data):
    """Compute 100 * max relative error across all validation points."""
    if _time_left() < 1.0:
        return 1e10
    
    max_err = 0.0
    
    for idx, (Om0, w0, wa, f_ref, D_ref) in enumerate(ref_data):
        # Compute approximate f and D
        f_approx = _compute_f(c, a_values, Om0, w0, wa)
        D_approx = _compute_D_ratio(c, a_values, Om0, w0, wa)
        
        # Check gamma constraint
        gamma_vals = _compute_gamma(c, a_values, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1e10
        
        # Relative error in f
        for i in range(len(a_values)):
            if abs(f_ref[i]) > 1e-10:
                err_f = abs(f_approx[i] - f_ref[i]) / abs(f_ref[i])
                max_err = max(max_err, err_f)
        
        # Relative error in D
        for i in range(len(a_values)):
            if abs(D_ref[i]) > 1e-10:
                err_D = abs(D_approx[i] - D_ref[i]) / abs(D_ref[i])
                max_err = max(max_err, err_D)
    
    return 100.0 * max_err

def construct():
    """Return six finite coefficients in [-10,10] for the growth-index formula."""
    # Pre-compute reference data
    validation_models, a_values, ref_data = _make_reference_data()
    
    # Initial guess: constant gamma ~ 0.55
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = c0.copy()
    best_obj = _objective(c0, validation_models, a_values, ref_data)
    
    # First, try Nelder-Mead from the initial guess
    if _time_left() > 10.0:
        try:
            result = minimize(_objective, c0, args=(validation_models, a_values, ref_data),
                            method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_obj:
                best_c = result.x
                best_obj = result.fun
        except:
            pass
    
    # Try differential evolution for global search
    if _time_left() > 20.0:
        try:
            bounds = [(-10, 10)] * 6
            de_result = differential_evolution(
                _objective, bounds,
                args=(validation_models, a_values, ref_data),
                maxiter=50, popsize=15,
                tol=1e-6, mutation=(0.3, 1.0),
                recombination=0.7, seed=42,
                polish=True
            )
            if de_result.fun < best_obj:
                best_c = de_result.x
                best_obj = de_result.fun
        except:
            pass
    
    # Refine with Nelder-Mead from the best found so far
    if _time_left() > 5.0:
        try:
            result = minimize(_objective, best_c, args=(validation_models, a_values, ref_data),
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_obj:
                best_c = result.x
                best_obj = result.fun
        except:
            pass
    
    # Try a few different starting points with Nelder-Mead
    if _time_left() > 10.0:
        starts = [
            np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.6, -0.1, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.55, 0.05, -0.05, 0.0, 0.0, 0.0]),
            np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]),
            np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]),
            np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.1]),
        ]
        for start in starts:
            if _time_left() < 5.0:
                break
            try:
                result = minimize(_objective, start, args=(validation_models, a_values, ref_data),
                                method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
                if result.fun < best_obj:
                    best_c = result.x
                    best_obj = result.fun
            except:
                pass
    
    # Clip to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Final validation: ensure gamma is in [0.1, 1.2]
    valid = True
    for Om0, w0, wa, _, _ in ref_data:
        gamma_vals = _compute_gamma(best_c, a_values, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            valid = False
            break
    
    if not valid:
        # Fall back to constant gamma which is always valid
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return list(best_c)
