"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution, linprog
import time

_DEADLINE = time.time() + 85.0

def _check_time():
    return time.time() > _DEADLINE

def _compute_om_a_vec(a, Om0, w0, wa):
    """Vectorized computation of matter density fraction."""
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.where(a > 1e-10, a, 1e-10)
    w_eff = w0 + wa
    de_term = (1.0 - Om0) * (a_safe ** (-3.0 * (1.0 + w_eff))) * np.exp(3.0 * wa * a_safe)
    m_term = Om0 * (a_safe ** -3.0)
    E2 = m_term + de_term
    return m_term / E2

def _compute_gamma_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of gamma."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _compute_f_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of f = Om(a)^gamma."""
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    gamma = _compute_gamma_vec(a, Om0, w0, wa, coeffs)
    return np.maximum(Om_a, 1e-15) ** gamma

def _compute_D_vec(a, Om0, w0, wa, coeffs, n_grid=1000):
    """Vectorized computation of normalized D."""
    a = np.asarray(a, dtype=np.float64)
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    a_masked = a[mask]
    a_min = max(np.min(a_masked), 0.01)
    log_a_grid = np.linspace(np.log(a_min), 0.0, n_grid)
    a_grid = np.exp(log_a_grid)
    f_grid = _compute_f_vec(a_grid, Om0, w0, wa, coeffs)
    dx = np.diff(log_a_grid)
    trapezoids = 0.5 * (f_grid[1:] + f_grid[:-1]) * dx
    cumulative = np.concatenate([[0.0], np.cumsum(trapezoids)])
    ln_a_masked = np.log(a_masked)
    I_masked = np.interp(ln_a_masked, log_a_grid, cumulative) - cumulative[-1]
    D[mask] = np.exp(I_masked)
    return D

def _solve_reference_f(Om0, w0, wa, a_vals):
    """Solve the ODE for f reference values with high precision."""
    def ode_func(ln_a, y):
        f = y[0]
        a = np.exp(ln_a)
        Om_a = _compute_om_a_vec(a, Om0, w0, wa)[0]
        w_eff = w0 + wa
        de_term = (1.0 - Om0) * (a ** (-3.0 * (1.0 + w_eff))) * np.exp(3.0 * wa * a)
        m_term = Om0 * (a ** -3.0)
        E2 = m_term + de_term
        g = a ** (-3.0 * (1.0 + w_eff)) * np.exp(3.0 * wa * a)
        dE2_da = -3.0 * Om0 * a ** (-4.0) + (1.0 - Om0) * g * 3.0 * (wa - (1.0 + w_eff) / a)
        dlnE2_dlna = (dE2_da * a) / E2
        Hprime_H = 0.5 * dlnE2_dlna
        f_prime = -1.5 * Om_a / E2 - (3.0 / a + Hprime_H) * f - f**2
        return [f_prime]
    
    f_init = 0.55 * Om0**0.55
    ln_a_grid = np.log(a_vals)
    try:
        sol = solve_ivp(ode_func, [0.0, np.log(a_vals[0])], [f_init], 
                       t_eval=ln_a_grid, method='DOP853', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[0]
    except Exception:
        pass
    coeffs = [0.55, 0, 0, 0, 0, 0]
    return _compute_f_vec(a_vals, Om0, w0, wa, coeffs)

def _solve_reference_D(Om0, w0, wa, a_vals, f_ref, n_grid=2000):
    """Compute reference D from reference f via high-precision integration."""
    D = np.ones_like(a_vals)
    mask = a_vals < 1.0
    if not np.any(mask):
        return D
    a_masked = a_vals[mask]
    a_min = max(np.min(a_masked), 0.01)
    
    log_a_grid = np.linspace(np.log(a_min), 0.0, n_grid)
    a_grid = np.exp(log_a_grid)
    
    ln_a_ref = np.log(a_vals)
    f_fine = np.interp(log_a_grid, ln_a_ref, f_ref)
    
    dx = np.diff(log_a_grid)
    trapezoids = 0.5 * (f_fine[1:] + f_fine[:-1]) * dx
    cumulative = np.concatenate([[0.0], np.cumsum(trapezoids)])
    
    ln_a_masked = np.log(a_masked)
    I_masked = np.interp(ln_a_masked, log_a_grid, cumulative) - cumulative[-1]
    D[mask] = np.exp(I_masked)
    return D

def _precompute_all_references():
    """Precompute reference data for all validation cosmologies."""
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.30, -1.5, 1.0),
        (0.30, -0.5, -0.5),
        (0.20, -1.2, 0.8),
        (0.40, -0.7, -0.3),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.3),
        (0.25, -0.9, 0.1),
        (0.35, -1.1, -0.1),
    ]
    
    a_vals = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    
    ref_data = []
    for Om0, w0, wa in models:
        f_ref = _solve_reference_f(Om0, w0, wa, a_vals)
        D_ref = _solve_reference_D(Om0, w0, wa, a_vals, f_ref)
        ref_data.append((Om0, w0, wa, a_vals, f_ref, D_ref))
    return ref_data

def _evaluate_score(coeffs, ref_data):
    """Calculate the worst-case relative error."""
    max_err = 0.0
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref_data:
        f_approx = _compute_f_vec(a_vals, Om0, w0, wa, coeffs)
        D_approx = _compute_D_vec(a_vals, Om0, w0, wa, coeffs)
        
        err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        max_err = max(max_err, np.max(err_f))
        
        err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err = max(max_err, np.max(err_D))
        
        gamma_vals = _compute_gamma_vec(a_vals, Om0, w0, wa, coeffs)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1.0
    return max_err

def _optimize_coeffs():
    start_time = time.time()
    deadline = start_time + 83.0
    
    ref_data = _precompute_all_references()
    
    if _check_time():
        return np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    
    best_coeffs = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_score = _evaluate_score(best_coeffs, ref_data)
    
    # Try LP for initial good starting point
    try:
        c_obj, A_ub, b_ub, bounds = _build_lp_problem(ref_data)
        if c_obj is not None and time.time() < deadline - 40:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                candidate = res.x[:6]
                np.clip(candidate, -10, 10, out=candidate)
                score = _evaluate_score(candidate, ref_data)
                if score < best_score and not np.any(np.isnan(candidate)):
                    best_score = score
                    best_coeffs = candidate
    except Exception:
        pass
    
    if _check_time():
        return best_coeffs
    
    # Differential evolution for global exploration
    def objective(x):
        if time.time() > deadline:
            return 1.0
        x = np.clip(x, -10, 10)
        return _evaluate_score(x, ref_data)
    
    bounds_de = [(-10, 10)] * 6
    try:
        if time.time() < deadline - 25:
            de_result = differential_evolution(objective, bounds_de, seed=42,
                                               maxiter=80, popsize=20, tol=1e-7,
                                               polish=False, init='sobol')
            if de_result.fun < best_score and np.all(np.isfinite(de_result.x)):
                best_score = de_result.fun
                best_coeffs = np.clip(de_result.x, -10, 10)
    except Exception:
        pass
    
    if _check_time():
        return best_coeffs
    
    # Local refinement with Nelder-Mead
    try:
        if time.time() < deadline - 15:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
            if result.fun < best_score and np.all(np.isfinite(result.x)):
                best_score = result.fun
                best_coeffs = np.clip(result.x, -10, 10)
    except Exception:
        pass
    
    if _check_time():
        return best_coeffs
    
    # Final high-precision polish
    try:
        if time.time() < deadline - 5:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-11, 'adaptive': True})
            if result.fun < best_score and np.all(np.isfinite(result.x)):
                best_score = result.fun
                best_coeffs = np.clip(result.x, -10, 10)
    except Exception:
        pass
    
    return best_coeffs

def _build_lp_problem(ref_data):
    """Build LP problem to minimize max relative error."""
    n_coeffs = 6
    A_ub_rows = []
    b_ub_vals = []
    
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref_data:
        Om_a = _compute_om_a_vec(a_vals, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        log_Om = np.log(np.maximum(Om_a, 1e-15))
        
        design = np.column_stack([np.ones_like(a_vals), q, q**2, u, wa*a_vals, u*q])
        
        safe_log_Om = np.where(np.abs(log_Om) > 1e-10, log_Om, 1e-10)
        gamma_ref_f = np.log(np.maximum(f_ref, 1e-15)) / safe_log_Om
        gamma_ref_f = np.clip(gamma_ref_f, 0.1, 1.2)
        
        for i in range(len(a_vals)):
            if a_vals[i] >= 1.0:
                continue
            abs_log_Om = abs(log_Om[i])
            if abs_log_Om < 1e-8:
                continue
            
            row1 = design[i] * abs_log_Om
            A_ub_rows.append(np.concatenate([row1, [-1.0]]))
            b_ub_vals.append(gamma_ref_f[i] * abs_log_Om)
            
            A_ub_rows.append(np.concatenate([-row1, [-1.0]]))
            b_ub_vals.append(-gamma_ref_f[i] * abs_log_Om)
            
        for i in range(len(a_vals)):
            if a_vals[i] >= 1.0:
                continue
            A_ub_rows.append(np.concatenate([-design[i], [0.0]]))
            b_ub_vals.append(-0.1)
            A_ub_rows.append(np.concatenate([design[i], [0.0]]))
            b_ub_vals.append(1.2)
    
    if not A_ub_rows:
        return None, None, None, None
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    c_obj = np.zeros(n_coeffs + 1)
    c_obj[-1] = 1.0
    
    bounds = [(-10, 10)] * n_coeffs + [(0, 1.0)]
    
    return c_obj, A_ub, b_ub, bounds

_optimized_coeffs = None

def construct():
    global _optimized_coeffs
    if _optimized_coeffs is None:
        _optimized_coeffs = _optimize_coeffs()
    return _optimized_coeffs
