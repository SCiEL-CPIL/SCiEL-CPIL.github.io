"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START

def _H2(a, Om0, w0, wa):
    """H(a)^2 / H0^2 for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    Ode0 = 1.0 - Om0
    # Ode(a) = Ode0 * a^(-3*(1+w0+wa))
    Ode = Ode0 * np.power(a, -3.0 * (1.0 + w0 + wa))
    Om_m = Om0 * np.power(a, -3.0)
    return Om_m + Ode

def _Om_m(a, Om0):
    """Matter density parameter."""
    return Om0 * np.power(np.asarray(a, dtype=float), -3.0)

def _w_eff(a, w0, wa):
    """Effective w at scale factor a."""
    return w0 + wa * (1.0 - np.asarray(a, dtype=float))

def _growth_ode(t, y, Om0, w0, wa):
    """ODE for growth: t = ln(a), y = [D, f] where f = d ln D / d ln a."""
    a = np.exp(t)
    Om = Om0 * a**(-3)
    w = w0 + wa * (1 - a)
    D, f = y
    # f' = d f / d ln a = (3/2) Om - f*(2 + f + (1+3w)/2)
    fp = 0.5 * Om - f * (2.0 + f + 0.5 * (1.0 + 3.0 * w))
    # D' = d D / d ln a = f * D
    Dp = f * D
    return [Dp, fp]

def _compute_reference(Om0, w0, wa, a_vals):
    """Compute true f(a) and normalized D(a)/D(1) for given cosmology."""
    # Integrate from a=1 (t=0) backwards to a_min
    a_min = min(a_vals)
    t_end = np.log(a_min)
    t0 = 0.0  # a=1
    
    # Initial conditions at a=1: D(1)=1, f(1) ~ Om(1)^0.55 (use good initial guess)
    f_init = Om0 ** 0.55
    D_init = 1.0
    
    t_eval = np.log(a_vals)
    # We need to integrate from t0=0 to t_end (negative)
    sol = solve_ivp(_growth_ode, [t0, t_end], [D_init, f_init], 
                    args=(Om0, w0, wa), t_eval=t_eval, 
                    rtol=1e-10, atol=1e-12, method='RK45')
    
    D_ref = sol.y[0]
    f_ref = sol.y[1]
    
    # Normalize D so that D(1)=1 (already is by construction)
    # D(a)/D(1) = D(a) since D(1)=1
    D_norm = D_ref  # already normalized
    
    return f_ref, D_norm

def _compute_model(coeffs, a_vals, Om0, w0, wa):
    """Compute model f(a) and D(a)/D(1) from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a_vals, dtype=float)
    Om = Om0 * a**(-3)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    
    # Clip gamma to [0.1, 1.2] for safety in computation
    gamma_clipped = np.clip(gamma, 0.1, 1.2)
    
    f_model = np.power(Om, gamma_clipped)
    
    # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # = exp(-integral from 0 to ln(a) of -f(exp(x)) dx) 
    # Actually: D(a)/D(1) = exp(-int_{ln(a)}^0 f(exp(x)) dx)
    # Let's compute: for each a, integral from ln(a) to 0 of f(exp(x)) dx
    # Use numerical integration
    
    D_model = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D_model[i] = 1.0
        else:
            t_a = np.log(ai)
            # Integrate f(exp(x)) from t_a to 0
            def integrand(x):
                ax = np.exp(x)
                Om_x = Om0 * ax**(-3)
                q_x = 1.0 - Om_x
                u_x = 1.0 + w0 + wa * (1.0 - ax)
                gamma_x = c0 + c1*q_x + c2*q_x**2 + c3*u_x + c4*wa*ax + c5*u_x*q_x
                gamma_x = np.clip(gamma_x, 0.1, 1.2)
                return Om_x ** gamma_x
            from scipy.integrate import quad
            val, _ = quad(integrand, t_a, 0.0, limit=100)
            D_model[i] = np.exp(-val)
    
    return f_model, D_model, gamma

def _objective(coeffs, cosmologies, a_vals_list):
    """Compute 100 * max relative error over all cosmologies and a values."""
    max_err = 0.0
    for (Om0, w0, wa), a_vals in zip(cosmologies, a_vals_list):
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_vals)
        f_model, D_model, gamma = _compute_model(coeffs, a_vals, Om0, w0, wa)
        
        # Check gamma constraint
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6  # penalty
        
        # Relative errors
        mask_f = np.abs(f_ref) > 1e-10
        err_f = np.abs((f_model[mask_f] - f_ref[mask_f]) / f_ref[mask_f])
        mask_D = np.abs(D_ref) > 1e-10
        err_D = np.abs((D_model[mask_D] - D_ref[mask_D]) / D_ref[mask_D])
        
        local_max = max(np.max(err_f) if len(err_f) > 0 else 0, 
                       np.max(err_D) if len(err_D) > 0 else 0)
        max_err = max(max_err, local_max)
    
    return 100.0 * max_err

def _is_valid(coeffs):
    """Check if coefficients are valid."""
    if len(coeffs) != 6:
        return False
    for c in coeffs:
        if not np.isfinite(c) or c < -10 or c > 10:
            return False
    return True

def construct():
    """Construct the six coefficients."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Representative validation cosmologies (standard set)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -0.9, 0.2),    # CPL
        (0.3, -1.1, 0.5),    # CPL
        (0.25, -1.0, 0.0),   # Lambda CDM, different Om
        (0.35, -1.0, 0.0),   # Lambda CDM, different Om
        (0.3, -0.8, 0.3),    # CPL
        (0.3, -1.2, -0.3),   # CPL
    ]
    
    # a values to check (log-spaced from ~0.01 to 1)
    a_vals = np.logspace(-2, 0, 50)
    a_vals_list = [a_vals] * len(cosmologies)
    
    # Pre-compute references
    refs = []
    for Om0, w0, wa in cosmologies:
        if time.time() > deadline:
            break
        f_ref, D_ref = _compute_reference(Om0, w0, wa, a_vals)
        refs.append((f_ref, D_ref))
    
    if not refs:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Objective using pre-computed references
    def obj_fast(coeffs):
        if not _is_valid(coeffs):
            return 1e6
        max_err = 0.0
        for i, ((Om0, w0, wa), (f_ref, D_ref)) in enumerate(zip(cosmologies, refs)):
            f_model, D_model, gamma = _compute_model(coeffs, a_vals, Om0, w0, wa)
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            mask_f = np.abs(f_ref) > 1e-10
            err_f = np.abs((f_model[mask_f] - f_ref[mask_f]) / f_ref[mask_f])
            mask_D = np.abs(D_ref) > 1e-10
            err_D = np.abs((D_model[mask_D] - D_ref[mask_D]) / D_ref[mask_D])
            local_max = max(np.max(err_f) if len(err_f) > 0 else 0, 
                           np.max(err_D) if len(err_D) > 0 else 0)
            max_err = max(max_err, local_max)
        return 100.0 * max_err
    
    # Start from seed
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = obj_fast(best_coeffs)
    
    # Try local optimization from seed
    if time.time() < deadline:
        try:
            res = minimize(obj_fast, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if _is_valid(res.x) and res.fun < best_score:
                best_coeffs = res.x
                best_score = res.fun
        except:
            pass
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    n_restarts = 5
    for _ in range(n_restarts):
        if time.time() > deadline - 5:
            break
        x0 = rng.uniform(-1, 1, 6)
        x0[0] = 0.55  # keep c0 near seed
        try:
            res = minimize(obj_fast, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if _is_valid(res.x) and res.fun < best_score:
                best_coeffs = res.x
                best_score = res.fun
        except:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 2:
        try:
            res = minimize(obj_fast, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if _is_valid(res.x) and res.fun < best_score:
                best_coeffs = res.x
                best_score = res.fun
        except:
            pass
    
    # Ensure coefficients are in range
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)

# EVOLVE-BLOCK-END
