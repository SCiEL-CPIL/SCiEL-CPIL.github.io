"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Pre-computed reference data cache
_ref_data_cache = None

def _compute_om_a_vec(a, Om0, w0, wa):
    """Vectorized computation of matter density fraction."""
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.where(a > 0, a, 1e-10)
    
    # Exact CPL evolution for DE density
    w_eff = 1 + w0 + wa
    de_term = (1 - Om0) * (a_safe ** (-3 * w_eff)) * np.exp(3 * wa * a_safe)
    m_term = Om0 * (a_safe ** -3)
    E2 = m_term + de_term
    Om_a = m_term / E2
    return Om_a

def _compute_gamma_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of gamma."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _compute_f_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of f = Om(a)^gamma."""
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    gamma = _compute_gamma_vec(a, Om0, w0, wa, coeffs)
    Om_a_safe = np.maximum(Om_a, 1e-10)
    return Om_a_safe ** gamma

def _compute_D_vec(a, Om0, w0, wa, coeffs):
    """
    Vectorized computation of normalized D.
    D(a)/D(1) = exp(-int_ln(a)^0 f(exp(x)) dx)
    """
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    
    a_masked = a[mask]
    a_min = np.min(a_masked)
    
    # Use a finer grid for better accuracy
    n_points = 300
    log_a_grid = np.linspace(np.log(a_min), 0.0, n_points)
    a_grid = np.exp(log_a_grid)
    
    f_grid = _compute_f_vec(a_grid, Om0, w0, wa, coeffs)
    
    log_a_rev = log_a_grid[::-1]
    f_rev = f_grid[::-1]
    
    # Cumulative trapezoid
    I_rev = np.zeros_like(log_a_rev)
    for i in range(1, len(log_a_rev)):
        I_rev[i] = I_rev[i-1] + 0.5 * (f_rev[i] + f_rev[i-1]) * (log_a_rev[i] - log_a_rev[i-1])
    
    ln_a_masked = np.log(a_masked)
    
    xp = log_a_rev[::-1]
    fp = I_rev[::-1]
    
    I_masked = np.interp(ln_a_masked, xp, fp)
    
    D[mask] = np.exp(I_masked)
    return D

def _precompute_references():
    """Pre-compute reference f and D for a set of validation models with high accuracy."""
    global _ref_data_cache
    if _ref_data_cache is not None:
        return _ref_data_cache
        
    # Validation models
    models = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -0.9, 0.2),
        (0.3, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.3, -1.5, 1.0),
        (0.3, -0.5, -0.5),
    ]
    
    # Use more a values for better resolution
    a_vals = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    ref_data = []
    for Om0, w0, wa in models:
        def ode_func(ln_a, f):
            a = np.exp(ln_a)
            Om_a = _compute_om_a_vec(a, Om0, w0, wa)[0]
            w_eff = 1 + w0 + wa
            g = a**(-3*w_eff) * np.exp(3*wa*a)
            E2 = Om0 * a**(-3) + (1-Om0) * g
            
            dE2_da = -3*Om0*a**(-4) + (1-Om0)*g*3*(wa - w_eff/a)
            dlnE2_dlna = (dE2_da * a) / E2
            Hprime_H = 0.5 * dlnE2_dlna
            
            f_prime = -(1.5) * Om_a / E2 - (3.0 / a + Hprime_H) * f - f**2
            return [f_prime]
        
        ln_a_grid = np.log(a_vals)
        f_init = 0.55 * Om0**0.55
        
        try:
            # Use higher accuracy
            sol = solve_ivp(ode_func, [0, np.log(0.3)], [f_init], 
                           t_eval=ln_a_grid, method='RK45', rtol=1e-10, atol=1e-12)
            if sol.success:
                f_ref = sol.y[0]
            else:
                f_ref = _compute_f_vec(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
        except Exception:
            f_ref = _compute_f_vec(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
            
        # Compute D_ref using finer integration
        ln_a_rev = ln_a_grid[::-1]
        f_rev = f_ref[::-1]
        
        # Use finer grid for D integration
        n_fine = 500
        ln_a_fine = np.linspace(0, np.log(0.3), n_fine)
        
        # Interpolate f_ref to fine grid
        f_fine = np.interp(ln_a_fine, ln_a_grid, f_ref)
        
        integral_vals = np.zeros_like(ln_a_fine)
        for i in range(1, len(ln_a_fine)):
            integral_vals[i] = integral_vals[i-1] + 0.5 * (f_fine[i] + f_fine[i-1]) * (ln_a_fine[i] - ln_a_fine[i-1])
        
        D_rev = np.exp(integral_vals)
        D_ref = np.interp(ln_a_grid, ln_a_fine[::-1], D_rev[::-1])
        
        ref_data.append((Om0, w0, wa, a_vals, f_ref, D_ref))
        
    _ref_data_cache = ref_data
    return ref_data

def _evaluate_score(coeffs):
    """Calculate the worst-case relative error for a set of coefficients."""
    ref_data = _precompute_references()
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

def _evaluate_score_smooth(coeffs, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for the max-error objective.
    This enables gradient-based optimization.
    """
    ref_data = _precompute_references()
    
    # Collect all relative errors
    all_errors = []
    
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref_data:
        f_approx = _compute_f_vec(a_vals, Om0, w0, wa, coeffs)
        D_approx = _compute_D_vec(a_vals, Om0, w0, wa, coeffs)
        
        err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        all_errors.append(err_f)
        
        err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        all_errors.append(err_D)
        
        gamma_vals = _compute_gamma_vec(a_vals, Om0, w0, wa, coeffs)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1.0
    
    all_errors = np.concatenate(all_errors)
    
    # Log-sum-exp for smooth max approximation
    # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    max_err = np.max(all_errors)
    if max_err < 1e-15:
        return max_err
    
    # Use log-sum-exp with shift for numerical stability
    shifted = beta * (all_errors - max_err)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_max = max_err + (1.0 / beta) * log_sum
    
    return smooth_max

def _optimize_coeffs():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial guess
    best_coeffs = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_score = _evaluate_score(best_coeffs)
    
    # Constraints: coeffs in [-10, 10]
    bounds = [(-10, 10)] * 6
    
    # Phase 1: Use smooth surrogate with L-BFGS-B for gradient-based optimization
    try:
        def smooth_objective(x):
            return _evaluate_score_smooth(x, beta=50.0)
        
        result1 = minimize(smooth_objective, best_coeffs, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result1.fun < best_score:
            best_score = _evaluate_score(result1.x)
            best_coeffs = result1.x
            
    except Exception as e:
        print(f"L-BFGS-B error: {e}")
    
    # Check time
    if time.time() > deadline:
        return best_coeffs
    
    # Phase 2: Nelder-Mead on the exact (nonsmooth) objective for local refinement
    try:
        def exact_objective(x):
            return _evaluate_score(x)
        
        result2 = minimize(exact_objective, best_coeffs, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if result2.fun < best_score:
            best_score = result2.fun
            best_coeffs = result2.x
            
    except Exception as e:
        print(f"Nelder-Mead error: {e}")
    
    # Check time
    if time.time() > deadline:
        return best_coeffs
    
    # Phase 3: Final high-precision polish with tighter tolerances
    try:
        def smooth_objective_fine(x):
            return _evaluate_score_smooth(x, beta=100.0)
        
        result3 = minimize(smooth_objective_fine, best_coeffs, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result3.fun < best_score:
            best_score = _evaluate_score(result3.x)
            best_coeffs = result3.x
            
    except Exception as e:
        print(f"Polish error: {e}")
    
    # Check time
    if time.time() > deadline:
        return best_coeffs
    
    # Phase 4: A few random restarts with smooth objective if time permits
    np.random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 5.0:
            break
        
        x0 = np.random.uniform(-1, 1, 6)
        x0 = np.clip(x0, -5, 5)
        
        try:
            res = minimize(lambda x: _evaluate_score_smooth(x, beta=50.0), x0, 
                          method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.fun < best_score:
                best_score = _evaluate_score(res.x)
                best_coeffs = res.x
                
        except Exception:
            pass
    
    return best_coeffs

# Cache the optimized coefficients
_optimized_coeffs = None

def construct():
    global _optimized_coeffs
    if _optimized_coeffs is None:
        _optimized_coeffs = _optimize_coeffs()
    return _optimized_coeffs
# EVOLVE-BLOCK-END
