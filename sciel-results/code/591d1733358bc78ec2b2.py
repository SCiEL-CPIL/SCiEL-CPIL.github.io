import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

# Global variables
_REF_DATA = []
_A_GRID = None
_TIME_LIMIT = 85.0  # Safety margin before 90s kill

def _compute_E_sq(Om0, w0, wa, a):
    """Compute E^2(a) = H^2(a)/H0^2 for flat CPL universe."""
    Om_m = Om0 * a**(-3)
    Om_de0 = 1.0 - Om0
    w_eff = w0 + wa * (1.0 - a)
    Om_de = Om_de0 * a**(-3.0 * (1.0 + w_eff))
    return Om_m + Om_de

def _compute_Om_m(Om0, w0, wa, a):
    """Compute matter density parameter Om_m(a)."""
    E2 = _compute_E_sq(Om0, w0, wa, a)
    return Om0 * a**(-3) / E2

def _generate_reference(Om0, w0, wa, a_grid):
    """
    Generate reference f(a) and D(a)/D(1) for a given cosmology.
    Uses the GR growth equation: a*df/da = 1.5*Om_m(a) - f*(f+1)
    """
    def ode_f(a, y):
        f = y[0]
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        return [(1.5 * Om_m - f * (f + 1)) / a]
    
    a_min = min(a_grid)
    a_max = max(a_grid)
    
    # Integrate from a small value to a_max to get f throughout
    a_start = 0.01
    a_eval = np.sort(np.unique(np.concatenate([a_grid, [a_start, 1.0]])))
    
    try:
        sol = solve_ivp(ode_f, [a_start, a_max], [1.0], method='RK45', 
                       t_eval=a_eval, rtol=1e-10, atol=1e-12, max_step=0.01)
        if not sol.success:
            raise RuntimeError("Integration failed")
        f_at_eval = sol.y[0]
    except Exception:
        # Fallback: use approximate solution
        f_at_eval = _compute_Om_m(Om0, w0, wa, a_eval) ** 0.55
    
    # Interpolate to a_grid
    f_ref = np.interp(a_grid, a_eval, f_at_eval)
    
    # Compute D(a)/D(1) = exp(∫_1^a f(a')/a' da')
    # Create dense grid for integration
    a_dense = np.linspace(a_min, a_max, 2000)
    f_dense = np.interp(a_dense, a_eval, f_at_eval)
    integrand = f_dense / a_dense
    
    # Cumulative integral
    C = cumulative_trapezoid(integrand, a_dense, initial=0)
    
    # Find index where a_dense ≈ 1.0
    idx_1 = np.argmin(np.abs(a_dense - 1.0))
    C1 = C[idx_1]
    
    # Interpolate to a_grid
    I_a = np.interp(a_grid, a_dense, C - C1)
    D_ref = np.exp(I_a)
    
    return f_ref, D_ref

def _make_reference_data():
    """Generate reference data for a set of cosmologies."""
    global _REF_DATA, _A_GRID
    
    # Representative cosmologies covering the parameter space
    cosmo_params = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.3),   # CPL
        (0.30, -0.9, 0.0),
        (0.30, -0.9, 0.5),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -1.2, 0.5),   # Phantom
        (0.30, -0.5, 0.0),   # w0 > -1
        (0.40, -1.0, 0.3),
        (0.20, -1.0, 0.3),
        (0.30, -1.0, -0.3),  # Negative wa
        (0.30, -0.8, 0.2),
        (0.28, -1.1, 0.4),
        (0.32, -0.95, -0.2),
    ]
    
    # Scale factor grid
    _A_GRID = np.linspace(0.1, 1.0, 100)
    
    _REF_DATA = []
    for Om0, w0, wa in cosmo_params:
        try:
            f_ref, D_ref = _generate_reference(Om0, w0, wa, _A_GRID)
            _REF_DATA.append((Om0, w0, wa, f_ref, D_ref))
        except Exception as e:
            pass

def _objective(c, max_err_only=True):
    """
    Objective function: Max relative error of f and D.
    Returns the max relative error (lower is better).
    """
    c0, c1, c2, c3, c4, c5 = c
    
    max_err = 0.0
    
    for Om0, w0, wa, f_ref, D_ref in _REF_DATA:
        a = _A_GRID
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        
        # Check gamma constraints
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_model = np.power(Om_m, gamma)
        
        # Compute D_model
        integrand = f_model / a
        idx_1 = np.argmin(np.abs(a - 1.0))
        C = cumulative_trapezoid(integrand, a, initial=0)
        C1 = C[idx_1]
        I_a = C - C1
        D_model = np.exp(I_a)
        
        # Relative errors
        err_f = np.abs(f_model - f_ref) / np.maximum(f_ref, 1e-10)
        err_D = np.abs(D_model - D_ref) / np.maximum(D_ref, 1e-10)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
        
    return max_err

def _soft_objective(c, temperature=0.01):
    """
    Soft-max objective using log-sum-exp for smoother optimization.
    """
    c0, c1, c2, c3, c4, c5 = c
    
    errors = []
    
    for Om0, w0, wa, f_ref, D_ref in _REF_DATA:
        a = _A_GRID
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_model = np.power(Om_m, gamma)
        
        integrand = f_model / a
        idx_1 = np.argmin(np.abs(a - 1.0))
        C = cumulative_trapezoid(integrand, a, initial=0)
        C1 = C[idx_1]
        I_a = C - C1
        D_model = np.exp(I_a)
        
        err_f = np.abs(f_model - f_ref) / np.maximum(f_ref, 1e-10)
        err_D = np.abs(D_model - D_ref) / np.maximum(D_ref, 1e-10)
        
        errors.extend(err_f.tolist())
        errors.extend(err_D.tolist())
    
    if len(errors) == 0:
        return 1e10
    
    errors = np.array(errors)
    # Log-sum-exp for soft max
    max_e = np.max(errors)
    log_sum = max_e + np.log(np.sum(np.exp(errors - max_e)))
    return log_sum / len(errors) + 0.5 * max_e  # Approximate max with smoothing

def construct():
    """
    Construct the optimal coefficients.
    """
    global _REF_DATA
    start_time = time.time()
    
    if not _REF_DATA:
        _make_reference_data()
    
    elapsed = time.time() - start_time
    if elapsed > _TIME_LIMIT - 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Initial guess: Taylor expansion in z around z=0
    # z = 1/a - 1, so at z=0, a=1
    # gamma(z) = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # For LCDM, gamma ≈ 0.55 at z=0, with small slope
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_c = np.array(x0)
    best_err = _objective(best_c)
    
    # Phase 1: Differential evolution for global search
    elapsed = time.time() - start_time
    de_time = min(40.0, _TIME_LIMIT - elapsed - 15.0)
    
    if de_time > 5.0:
        try:
            def callback(xk, convergence):
                nonlocal best_c, best_err
                err_k = _objective(xk)
                if err_k < best_err:
                    best_err = err_k
                    best_c = xk.copy()
            
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=30, 
                popsize=12, 
                tol=1e-6, 
                polish=False,
                callback=callback,
                update_interval=1,
                workers=1
            )
            
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
                
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish
    elapsed = time.time() - start_time
    nm_time = min(15.0, _TIME_LIMIT - elapsed - 5.0)
    
    if nm_time > 2.0:
        try:
            res_nm = minimize(
                _objective, 
                best_c, 
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 500}
            )
            if res_nm.fun < best_err:
                best_err = res_nm.fun
                best_c = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with tighter tolerances
    elapsed = time.time() - start_time
    final_time = min(10.0, _TIME_LIMIT - elapsed - 3.0)
    
    if final_time > 1.0:
        try:
            res_final = minimize(
                _objective, 
                best_c, 
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 300}
            )
            if res_final.fun < best_err:
                best_err = res_final.fun
                best_c = res_final.x.copy()
        except Exception:
            pass
    
    # Clip to valid range
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Verify constraints on a wider grid
    a_check = np.linspace(0.05, 1.0, 200)
    valid = True
    for Om0, w0, wa, _, _ in _REF_DATA:
        Om_m = _compute_Om_m(Om0, w0, wa, a_check)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a_check)
        gamma = best_c[0] + best_c[1]*q + best_c[2]*q**2 + best_c[3]*u + best_c[4]*wa*a_check + best_c[5]*u*q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            valid = False
            break
            
    if not valid:
        # Fallback to safe constant
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    return [float(x) for x in best_c]

if __name__ == "__main__":
    _make_reference_data()
    c = construct()
    print(c)
