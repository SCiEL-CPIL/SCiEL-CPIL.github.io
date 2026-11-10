"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import time
import math
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize

# Deadline guard
_START = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _elapsed_ok():
    return (time.time() - _START) < _DEADLINE

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter as function of scale factor for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    de_exp = 3.0 * wa * (1.0/a_safe - 1.0)
    E2 = Om0 * a_safe**3 + (1.0 - Om0) * np.exp(de_exp)
    return Om0 * a_safe**3 / E2

def _gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _f_approx(coeffs, a, Om0, w0, wa):
    """f = Om(a)^gamma"""
    Om = _Om_of_a(a, Om0, w0, wa)
    g = _gamma_from_coeffs(coeffs, a, Om0, w0, wa)
    g = np.clip(g, 0.1, 1.2)
    return Om**g

def _D_normalized(coeffs, a, Om0, w0, wa):
    """D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)"""
    if a >= 1.0:
        return 1.0
    # Integrate f(t)/t from a to 1
    n_pts = 300
    t = np.linspace(a, 1.0, n_pts)
    f_vals = _f_approx(coeffs, t, Om0, w0, wa)
    integrand = f_vals / t
    h = (1.0 - a) / (n_pts - 1)
    integral = h/3.0 * (integrand[0] + integrand[-1] + 
                       4.0*np.sum(integrand[1:-1:2]) + 
                       2.0*np.sum(integrand[2:-1:2]))
    return math.exp(-integral)

def _f_reference_ode_vectorized(a_vals, Om0, w0, wa):
    """Solve the GR ODE for f(a) = d ln D / d ln a, vectorized over a_vals."""
    a_vals = np.asarray(a_vals, dtype=float)
    a_start = 0.01
    a_end = 1.0
    n_steps = 3000
    h = (a_end - a_start) / n_steps
    
    def ode_rhs(a_val, f_val):
        Om = _Om_of_a(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        return -f_val**2 - 1.5*(1.0+w)*f_val + 1.5*Om

    a_curr = a_start
    # Initial condition: f ~ 0.55 at early times
    f_curr = 0.55
    a_grid = np.linspace(a_start, a_end, n_steps+1)
    f_grid = np.zeros(n_steps+1)
    f_grid[0] = f_curr
    
    for i in range(n_steps):
        a_curr = a_grid[i]
        f_curr = f_grid[i]
        def df_da(av, fv):
            return ode_rhs(av, fv) / av
        
        k1 = df_da(a_curr, f_curr)
        k2 = df_da(a_curr + h/2, f_curr + h/2*k1)
        k3 = df_da(a_curr + h/2, f_curr + h/2*k2)
        k4 = df_da(a_curr + h, f_curr + h*k3)
        f_curr = f_curr + h/6.0*(k1 + 2*k2 + 2*k3 + k4)
        f_grid[i+1] = f_curr
    
    # Interpolate to get f at desired a values
    result = np.empty_like(a_vals)
    for i, a in enumerate(a_vals):
        if a <= a_start:
            result[i] = f_grid[0]
        elif a >= a_end:
            result[i] = f_grid[-1]
        else:
            idx = int((a - a_start) / h)
            if idx >= n_steps:
                idx = n_steps - 1
            frac = (a - a_grid[idx]) / h
            result[i] = f_grid[idx] + frac * (f_grid[idx+1] - f_grid[idx])
    return result

def _D_reference(a_val, Om0, w0, wa, f_ref_grid=None, a_grid=None):
    """Compute reference D(a)/D(1) using the reference f."""
    if a_val >= 1.0:
        return 1.0
    
    # Integrate f_ref(t)/t from a_val to 1
    n_pts = 200
    t = np.linspace(a_val, 1.0, n_pts)
    f_ref_vals = _f_reference_ode_vectorized(t, Om0, w0, wa)
    integrand = f_ref_vals / t
    h = (1.0 - a_val) / (n_pts - 1)
    integral = h/3.0 * (integrand[0] + integrand[-1] + 
                       4.0*np.sum(integrand[1:-1:2]) + 
                       2.0*np.sum(integrand[2:-1:2]))
    return math.exp(-integral)

def _objective_smooth(coeffs, cosmologies, a_values, beta=50.0):
    """Compute smooth surrogate of max relative error using log-sum-exp."""
    errors = []
    for Om0, w0, wa in cosmologies:
        valid_a = [a for a in a_values if 0 < a < 1.0]
        if not valid_a:
            continue
        a_arr = np.array(valid_a)
        
        # Check gamma constraint
        g_arr = _gamma_from_coeffs(coeffs, a_arr, Om0, w0, wa)
        if np.any(g_arr < 0.1) or np.any(g_arr > 1.2):
            return 1e10
        
        # f approximation
        f_approx_arr = _f_approx(coeffs, a_arr, Om0, w0, wa)
        f_ref_arr = _f_reference_ode_vectorized(a_arr, Om0, w0, wa)
        
        for i in range(len(a_arr)):
            if abs(f_ref_arr[i]) > 1e-8:
                err_f = abs(f_approx_arr[i] - f_ref_arr[i]) / abs(f_ref_arr[i])
            else:
                err_f = abs(f_approx_arr[i] - f_ref_arr[i])
            errors.append(err_f)
            
            # D normalized error
            a_val = a_arr[i]
            D_approx = _D_normalized(coeffs, np.array([a_val]), Om0, w0, wa)[0]
            D_ref = _D_reference(a_val, Om0, w0, wa)
            
            if abs(D_ref) > 1e-8:
                err_D = abs(D_approx - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_approx - D_ref)
            errors.append(err_D)
    
    if not errors:
        return 0.0
    
    errors_arr = np.array(errors)
    # Log-sum-exp for smooth max
    max_e = np.max(errors_arr)
    if max_e < 1e-15:
        return max_e
    smooth_max = max_e + np.log(np.sum(np.exp(beta * (errors_arr - max_e)))) / beta
    return smooth_max

def _objective(coeffs, cosmologies, a_values):
    """Compute max relative error over all cosmologies and a values."""
    max_err = 0.0
    for Om0, w0, wa in cosmologies:
        valid_a = [a for a in a_values if 0 < a < 1.0]
        if not valid_a:
            continue
        a_arr = np.array(valid_a)
        
        # Check gamma constraint
        g_arr = _gamma_from_coeffs(coeffs, a_arr, Om0, w0, wa)
        if np.any(g_arr < 0.1) or np.any(g_arr > 1.2):
            return 1e10
        
        f_approx_arr = _f_approx(coeffs, a_arr, Om0, w0, wa)
        f_ref_arr = _f_reference_ode_vectorized(a_arr, Om0, w0, wa)
        
        for i in range(len(a_arr)):
            if abs(f_ref_arr[i]) > 1e-8:
                err_f = abs(f_approx_arr[i] - f_ref_arr[i]) / abs(f_ref_arr[i])
            else:
                err_f = abs(f_approx_arr[i] - f_ref_arr[i])
            max_err = max(max_err, err_f)
            
            a_val = a_arr[i]
            D_approx = _D_normalized(coeffs, np.array([a_val]), Om0, w0, wa)[0]
            D_ref = _D_reference(a_val, Om0, w0, wa)
            
            if abs(D_ref) > 1e-8:
                err_D = abs(D_approx - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_approx - D_ref)
            max_err = max(max_err, err_D)
    
    return max_err

def _chebyshev_init(cosmologies, a_values):
    """
    Use Chebyshev spectral projection to find optimal coefficients.
    
    Strategy:
    1. For a dense grid of CPL parameters, compute the exact gamma(a) = ln(f_ref)/ln(Om)
    2. Build the design matrix for the 6-term ansatz
    3. Solve least-squares to find coefficients that best fit gamma across all models
    """
    # Dense grid of representative CPL parameters
    Om0_grid = np.linspace(0.25, 0.35, 5)
    w0_grid = np.linspace(-1.2, -0.8, 5)
    wa_grid = np.linspace(-0.5, 0.5, 5)
    
    # a values for fitting (dense grid)
    a_fit = np.linspace(0.1, 0.99, 50)
    
    all_q = []
    all_q2 = []
    all_u = []
    all_wa_a = []
    all_uq = []
    all_gamma = []
    
    for Om0 in Om0_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                # Compute reference f for this cosmology
                f_ref = _f_reference_ode_vectorized(a_fit, Om0, w0, wa)
                Om_arr = _Om_of_a(a_fit, Om0, w0, wa)
                
                # Compute gamma_ref = ln(f_ref) / ln(Om)
                for i in range(len(a_fit)):
                    if Om_arr[i] > 0.01 and Om_arr[i] < 0.99 and f_ref[i] > 0.01:
                        gamma_val = np.log(f_ref[i]) / np.log(Om_arr[i])
                        # Only use if gamma is in reasonable range
                        if 0.1 <= gamma_val <= 1.2:
                            q_val = 1.0 - Om_arr[i]
                            u_val = 1.0 + w0 + wa * (1.0 - a_fit[i])
                            
                            all_q.append(q_val)
                            all_q2.append(q_val**2)
                            all_u.append(u_val)
                            all_wa_a.append(wa * a_fit[i])
                            all_uq.append(u_val * q_val)
                            all_gamma.append(gamma_val)
    
    if len(all_gamma) < 10:
        # Fallback to simple initialization
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Build design matrix
    n_samples = len(all_gamma)
    A = np.ones((n_samples, 6))
    A[:, 1] = all_q
    A[:, 2] = all_q2
    A[:, 3] = all_u
    A[:, 4] = all_wa_a
    A[:, 5] = all_uq
    
    b = np.array(all_gamma)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Clamp to reasonable range
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    return list(coeffs)

def _optimize(coeffs_init, cosmologies, a_values, max_time=70.0):
    """Nelder-Mead style optimization with time guard."""
    start_time = time.time()
    best_coeffs = list(coeffs_init)
    best_err = _objective(best_coeffs, cosmologies, a_values)
    
    # Try Chebyshev initialization
    try:
        cheb_coeffs = _chebyshev_init(cosmologies, a_values)
        cheb_err = _objective(cheb_coeffs, cosmologies, a_values)
        if cheb_err < best_err:
            best_coeffs = cheb_coeffs
            best_err = cheb_err
    except Exception:
        pass
    
    # Nelder-Mead simplex
    n_dim = len(coeffs_init)
    step_sizes = [0.05] * n_dim
    
    simplex = [list(best_coeffs)]
    for i in range(n_dim):
        pt = list(best_coeffs)
        pt[i] += step_sizes[i]
        simplex.append(pt)
    
    simplex_errs = [_objective(s, cosmologies, a_values) for s in simplex]
    
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() - start_time > max_time:
            break
        if not _elapsed_ok():
            break
        
        # Sort simplex by error
        order = np.argsort(simplex_errs)
        simplex = [simplex[i] for i in order]
        simplex_errs = [simplex_errs[i] for i in order]
        
        # Check convergence
        if simplex_errs[0] < 1e-8:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        worst_err = simplex_errs[-1]
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - np.array(worst))
        x_r = list(x_r)
        err_r = _objective(x_r, cosmologies, a_values)
        
        if err_r < simplex_errs[0]:
            # Expansion
            gamma_exp = 2.0
            x_e = centroid + gamma_exp * (np.array(x_r) - centroid)
            x_e = list(x_e)
            err_e = _objective(x_e, cosmologies, a_values)
            if err_e < err_r:
                simplex[-1] = x_e
                simplex_errs[-1] = err_e
            else:
                simplex[-1] = x_r
                simplex_errs[-1] = err_r
        elif err_r < simplex_errs[-2]:
            simplex[-1] = x_r
            simplex_errs[-1] = err_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (np.array(worst) - centroid)
            x_c = list(x_c)
            err_c = _objective(x_c, cosmologies, a_values)
            if err_c < worst_err:
                simplex[-1] = x_c
                simplex_errs[-1] = err_c
            else:
                # Shrink
                for i in range(1, n_dim + 1):
                    simplex[i] = [simplex[0][j] + 0.5*(simplex[i][j] - simplex[0][j]) for j in range(n_dim)]
                    simplex_errs[i] = _objective(simplex[i], cosmologies, a_values)
        
        # Update best
        min_idx = np.argmin(simplex_errs)
        if simplex_errs[min_idx] < best_err:
            best_err = simplex_errs[min_idx]
            best_coeffs = list(simplex[min_idx])
    
    # Polish with finer steps
    if _elapsed_ok() and (time.time() - start_time) < max_time - 5:
        for i in range(50):
            if not _elapsed_ok():
                break
            improved = False
            for j in range(n_dim):
                for delta in [0.005, -0.005, 0.001, -0.001, 0.0005, -0.0005]:
                    trial = list(best_coeffs)
                    trial[j] += delta
                    err = _objective(trial, cosmologies, a_values)
                    if err < best_err:
                        best_err = err
                        best_coeffs = trial
                        improved = True
            if not improved:
                break
    
    return best_coeffs, best_err

def construct():
    """Return 6 coefficients for the growth-index formula."""
    # Validation cosmologies (typical CPL models)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.3),    # CPL with wa=0.3
        (0.3, -1.0, -0.5),   # CPL with wa=-0.5
        (0.27, -1.0, 0.0),   # Lower Om0
        (0.3, -0.9, 0.0),    # w0 != -1
        (0.3, -1.1, 0.2),    # w0 < -1
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.3, -1.0, 0.5),    # Strong wa
    ]
    
    # a values for evaluation
    a_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Start from constant gamma
    coeffs_init = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        best_coeffs, best_err = _optimize(coeffs_init, cosmologies, a_values, max_time=70.0)
        
        # Ensure coefficients are in [-10, 10] and finite
        for i in range(6):
            if not math.isfinite(best_coeffs[i]):
                best_coeffs[i] = 0.0
            best_coeffs[i] = max(-10.0, min(10.0, best_coeffs[i]))
        
        return best_coeffs
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
