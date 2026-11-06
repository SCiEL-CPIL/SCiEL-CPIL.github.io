"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp, quad

# EVOLVE-BLOCK-START

COSMOLOGIES = [
    (0.3, -1.0, 0.0),
    (0.3, -0.9, 0.1),
    (0.3, -1.1, 0.5),
    (0.3, -0.8, -0.3),
    (0.3, -0.95, 0.2),
    (0.315, -1.0, 0.0),
    (0.315, -0.9, 0.1),
]

A_VALUES = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])


def _compute_om_a_vec(a, Om_m, w0, wa):
    """Vectorized matter density parameter at scale factor a for CPL dark energy."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    w_eff = 1.0 + w0 + wa
    factor_de = np.power(a, -3.0 * w_eff) * np.exp(-3.0 * wa * (1.0 - a))
    factor_m = np.power(a, -3.0)
    num = Om_m * factor_m
    den = Om_m * factor_m + (1.0 - Om_m) * factor_de
    return num / den


def _compute_gamma_vec(coeffs, a, Om_m, w0, wa):
    """Vectorized gamma computation."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _compute_om_a_vec(a, Om_m, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma, Om, q, u


def _solve_gr_ode(Om_m, w0, wa, a_grid):
    """Solve the GR growth ODE for f and D on a grid of a values."""
    a_grid = np.atleast_1d(np.asarray(a_grid, dtype=float))
    
    def E2(a_val):
        w_eff = 1.0 + w0 + wa
        term_m = Om_m * a_val**(-3.0)
        term_de = (1.0 - Om_m) * a_val**(-3.0 * w_eff) * np.exp(-3.0 * wa * (1.0 - a_val))
        return term_m + term_de
    
    def dlnH_dx(x):
        a_val = np.exp(x)
        dx = 1e-6
        E2_plus = E2(np.exp(x + dx))
        E2_minus = E2(np.exp(x - dx))
        E2_here = E2(a_val)
        dE2_dx = (E2_plus - E2_minus) / (2.0 * dx)
        return 0.5 * dE2_dx / E2_here
    
    def ode_func(x, y):
        f_val, lnD_val = y
        a_val = np.exp(x)
        Om_val = _compute_om_a_vec(a_val, Om_m, w0, wa)[0]
        dlH = dlnH_dx(x)
        df_dx = -(1.5 + dlH) * f_val + 1.5 * Om_val
        dlnD_dx = f_val
        return [df_dx, dlnD_dx]
    
    # Integrate from x=0 (a=1) backward to each a in a_grid
    x_min = np.log(np.min(a_grid))
    x_max = 0.0
    
    # Initial conditions at a=1
    Om_1 = _compute_om_a_vec(1.0, Om_m, w0, wa)[0]
    f_1 = Om_1 ** 0.55
    lnD_1 = 0.0
    
    n_points = 200
    x_eval = np.linspace(x_min, x_max, n_points)
    
    sol = solve_ivp(ode_func, [x_min, x_max], [f_1, lnD_1], 
                    method='RK45', rtol=1e-9, atol=1e-11,
                    t_eval=x_eval, dense_output=True)
    
    if not sol.success:
        f_ref = np.array([_compute_om_a_vec(a, Om_m, w0, wa)[0] ** 0.55 for a in a_grid])
        D_ref = np.array([np.exp(-np.log(1.0/a) * 0.55) for a in a_grid])
        return f_ref, D_ref
    
    f_ref = np.zeros_like(a_grid)
    lnD_ref = np.zeros_like(a_grid)
    
    for i, a_val in enumerate(a_grid):
        x_val = np.log(a_val)
        if x_val >= x_max - 1e-10:
            f_ref[i] = f_1
            lnD_ref[i] = 0.0
        else:
            y_interp = sol.sol(x_val)
            f_ref[i] = y_interp[0]
            lnD_ref[i] = y_interp[1]
    
    D_ref = np.exp(lnD_ref)
    return f_ref, D_ref


def _precompute_references():
    """Precompute reference f and D values for all cosmologies."""
    refs = []
    for (Om_m, w0, wa) in COSMOLOGIES:
        f_ref, D_ref = _solve_gr_ode(Om_m, w0, wa, A_VALUES)
        refs.append((Om_m, w0, wa, f_ref, D_ref))
    return refs


def _evaluate_score(coeffs, refs):
    """Compute worst-case relative error across f and D for all cosmologies and a values."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for (Om_m, w0, wa, f_ref, D_ref) in refs:
        gamma, Om, q, u = _compute_gamma_vec(coeffs, A_VALUES, Om_m, w0, wa)
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            viol = np.maximum(0.1 - gamma, 0) + np.maximum(gamma - 1.2, 0)
            return 100.0 + 10.0 * np.max(viol)
        
        f_ansatz = np.power(Om, gamma)
        
        valid_f = np.abs(f_ref) > 1e-10
        if np.any(valid_f):
            err_f = np.abs(f_ansatz[valid_f] - f_ref[valid_f]) / np.abs(f_ref[valid_f])
            max_err = max(max_err, np.max(err_f))
        
        D_ansatz = np.zeros_like(A_VALUES)
        for i, a_val in enumerate(A_VALUES):
            if a_val >= 1.0 - 1e-10:
                D_ansatz[i] = 1.0
            else:
                def integrand(x):
                    a_x = np.exp(x)
                    g_x, Om_x, _, _ = _compute_gamma_vec(coeffs, np.array([a_x]), Om_m, w0, wa)
                    return Om_x[0] ** g_x[0]
                
                try:
                    val, _ = quad(integrand, np.log(a_val), 0.0, limit=100)
                    D_ansatz[i] = np.exp(-val)
                except Exception:
                    D_ansatz[i] = np.exp(-np.log(1.0/a_val) * gamma[i])
        
        valid_D = np.abs(D_ref) > 1e-10
        if np.any(valid_D):
            err_D = np.abs(D_ansatz[valid_D] - D_ref[valid_D]) / np.abs(D_ref[valid_D])
            max_err = max(max_err, np.max(err_D))
    
    return 100.0 * max_err


def _evaluate_score_smooth(coeffs, refs, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max error."""
    c0, c1, c2, c3, c4, c5 = coeffs
    errors = []
    
    for (Om_m, w0, wa, f_ref, D_ref) in refs:
        gamma, Om, q, u = _compute_gamma_vec(coeffs, A_VALUES, Om_m, w0, wa)
        
        gamma_pen = np.maximum(0.1 - gamma, 0) + np.maximum(gamma - 1.2, 0)
        errors.extend(gamma_pen * 10.0)
        
        f_ansatz = np.power(Om, gamma)
        
        valid_f = np.abs(f_ref) > 1e-10
        if np.any(valid_f):
            err_f = np.abs(f_ansatz[valid_f] - f_ref[valid_f]) / np.abs(f_ref[valid_f])
            errors.extend(err_f)
        
        D_ansatz = np.zeros_like(A_VALUES)
        for i, a_val in enumerate(A_VALUES):
            if a_val >= 1.0 - 1e-10:
                D_ansatz[i] = 1.0
            else:
                def integrand(x):
                    a_x = np.exp(x)
                    g_x, Om_x, _, _ = _compute_gamma_vec(coeffs, np.array([a_x]), Om_m, w0, wa)
                    return Om_x[0] ** g_x[0]
                
                try:
                    val, _ = quad(integrand, np.log(a_val), 0.0, limit=100)
                    D_ansatz[i] = np.exp(-val)
                except Exception:
                    D_ansatz[i] = np.exp(-np.log(1.0/a_val) * gamma[i])
        
        valid_D = np.abs(D_ref) > 1e-10
        if np.any(valid_D):
            err_D = np.abs(D_ansatz[valid_D] - D_ref[valid_D]) / np.abs(D_ref[valid_D])
            errors.extend(err_D)
    
    if not errors:
        return 0.0
    
    errors = np.array(errors)
    max_e = np.max(errors)
    if max_e < 1e-10:
        return 100.0 * max_e
    lse = max_e + np.log(np.sum(np.exp((errors - max_e) * beta)) / beta)
    return 100.0 * lse / beta


def _optimize_coeffs(refs):
    """Optimize coefficients using DE + smooth Nelder-Mead polish."""
    start_time = time.time()
    deadline = start_time + 70.0
    
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bounds = [(-2, 2)] * 6
    
    best_result = None
    best_obj = 1e10
    
    try:
        def de_obj(x):
            if time.time() > deadline - 10:
                return 1e10
            return _evaluate_score_smooth(x, refs, beta=50.0)
        
        result = differential_evolution(
            de_obj,
            bounds,
            seed=42,
            maxiter=80,
            tol=1e-5,
            polish=False,
            popsize=15
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        if best_result is not None:
            exact_obj = _evaluate_score(best_result, refs)
            if exact_obj < 100.0:
                return [float(c) for c in best_result]
        return x0
    
    if best_result is not None:
        try:
            def nm_obj(x):
                if time.time() > deadline - 5:
                    return 1e10
                return _evaluate_score(x, refs)
            
            res2 = minimize(nm_obj, best_result, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_result = res2.x
        except Exception:
            pass
    
    if best_result is not None and time.time() < deadline - 3:
        try:
            def nm_obj2(x):
                if time.time() > deadline - 1:
                    return 1e10
                return _evaluate_score(x, refs)
            
            res3 = minimize(nm_obj2, best_result, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            if res3.fun < best_obj:
                best_obj = res3.fun
                best_result = res3.x
        except Exception:
            pass
    
    if best_result is not None:
        final_score = _evaluate_score(best_result, refs)
        if final_score < 100.0:
            return [float(c) for c in best_result]
    
    return x0


def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    refs = _precompute_references()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    result = _optimize_coeffs(refs)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    result = [float(c) for c in result]
    result = [max(-10.0, min(10.0, c)) for c in result]
    result = [c if np.isfinite(c) else 0.0 for c in result]
    
    return result

# EVOLVE-BLOCK-END
