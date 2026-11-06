"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_start_time = time.time()
_DEADLINE = 85.0

_cosmologies = [
    (0.3, -1.0, 0.0),
    (0.3, -0.9, 0.1),
    (0.3, -1.1, 0.2),
    (0.3, -0.8, -0.2),
    (0.3, -1.2, 0.3),
    (0.2, -1.0, 0.0),
    (0.4, -1.0, 0.0),
    (0.3, -0.5, 0.0),
]
_a_grid = np.linspace(0.05, 1.0, 30)
_ref_data = {}

def _precompute_references():
    for Om0, w0, wa in _cosmologies:
        Ode0 = 1.0 - Om0
        a_vals = _a_grid
        
        def dlnHdx(a):
            Om_a = Om0 * a**(-3)
            if Ode0 == 0:
                Ode_a = 0.0
                dOde_da = 0.0
            else:
                log_Ode_a = np.log(Ode0) - 3*(1+w0)*np.log(a) - 3*wa*(1-a)
                Ode_a = np.exp(log_Ode_a)
                dOde_da = Ode_a * (-3*(1+w0)/a + 3*wa)
            H2 = Om_a + Ode_a
            dH2_da = -3*Om0*a**(-4) + dOde_da
            dH2_dx = dH2_da * a
            return 0.5 * dH2_dx / H2
        
        def ode_system(x, y):
            a = np.exp(x)
            Om = Om0 * a**(-3)
            dlnHdx_val = dlnHdx(a)
            dDdx = y[1]
            d2Ddx2 = -3 * Om * y[0] - (3 + dlnHdx_val) * y[1]
            return [dDdx, d2Ddx2]
        
        f1 = Om0**0.6
        D1 = 1.0
        D1_prime = f1 * D1
        x_vals = np.log(a_vals)
        x_min = min(x_vals)
        
        sol = solve_ivp(ode_system, [0.0, x_min], [D1, D1_prime], 
                        rtol=1e-10, atol=1e-12, method='DOP853',
                        dense_output=True)
        
        D_vals = np.zeros(len(a_vals))
        f_vals = np.zeros(len(a_vals))
        for i, x in enumerate(x_vals):
            y = sol.sol(x)
            D_vals[i] = y[0]
            f_vals[i] = y[1] / y[0]
        
        D_norm = D_vals / D_vals[-1]
        _ref_data[(Om0, w0, wa)] = (f_vals, D_norm)

def _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a_vals):
    a_vals = np.array(a_vals)
    Om_a = Om0 * a_vals**(-3)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_vals)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    f_vals = Om_a**gamma
    
    x_vals = np.log(a_vals)
    x = x_vals
    f = f_vals
    n = len(x)
    
    # Vectorized cumulative trapezoidal integration from right to left
    dx = np.diff(x)
    avg_f = 0.5 * (f[:-1] + f[1:])
    trapezoids = avg_f * dx
    # Cumulative sum from the right
    integral_to_0 = np.zeros(n)
    for i in range(n-2, -1, -1):
        integral_to_0[i] = integral_to_0[i+1] + trapezoids[i]
    integral_to_0[n-1] = 0.0
    
    D_norm = np.exp(-integral_to_0)
    
    return f_vals, D_norm

def _evaluate_error(coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    all_errs = []
    for Om0, w0, wa in _cosmologies:
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod = _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, _a_grid)
        
        err_f = np.max(np.abs((f_mod - f_ref) / f_ref))
        err_D = np.max(np.abs((D_mod - D_ref) / D_ref))
        
        max_err = max(max_err, err_f, err_D)
        all_errs.extend([err_f, err_D])
    
    return max_err, np.array(all_errs)

def _smooth_objective(coeffs):
    """Smooth surrogate using log-sum-exp of squared relative errors."""
    c0, c1, c2, c3, c4, c5 = coeffs
    all_sq_errs = []
    for Om0, w0, wa in _cosmologies:
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod = _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, _a_grid)
        
        rel_err_f = np.abs((f_mod - f_ref) / f_ref)
        rel_err_D = np.abs((D_mod - D_ref) / D_ref)
        
        all_sq_errs.extend(np.concatenate([rel_err_f**2, rel_err_D**2]).tolist())
    
    all_sq_errs = np.array(all_sq_errs)
    # Log-sum-exp for numerical stability
    max_val = np.max(all_sq_errs)
    if max_val == 0:
        return 0.0
    lse = max_val + np.log(np.sum(np.exp(all_sq_errs - max_val)))
    # Return sqrt of LSE to approximate max (in log space)
    return np.exp(0.5 * lse)

def _is_valid(coeffs, cosmologies, a_grid):
    c0, c1, c2, c3, c4, c5 = coeffs
    for Om0, w0, wa in cosmologies:
        Om_a = Om0 * a_grid**(-3)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa*(1.0 - a_grid)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
    return True

def construct():
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    _precompute_references()
    
    best_result = None
    best_score = float('inf')
    
    def _update_best(result, score):
        nonlocal best_result, best_score
        if score < best_score and _is_valid(result, _cosmologies, _a_grid):
            best_score = score
            best_result = result
    
    # Phase 1: Optimize smooth surrogate with L-BFGS-B
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        result = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        # Evaluate on original max objective
        actual_score, _ = _evaluate_error(result.x)
        _update_best(result.x, actual_score)
    except:
        pass
    
    if time.time() - _start_time > _DEADLINE - 20:
        if best_result is not None:
            return np.clip(best_result, -10, 10).tolist()
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Phase 2: Nelder-Mead on original max objective from best point
    x0_nm = best_result if best_result is not None else x0
    try:
        result = minimize(lambda c: _evaluate_error(c)[0], x0_nm, method='Nelder-Mead', 
                         options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
        actual_score, _ = _evaluate_error(result.x)
        _update_best(result.x, actual_score)
    except:
        pass
    
    if time.time() - _start_time > _DEADLINE - 15:
        if best_result is not None:
            return np.clip(best_result, -10, 10).tolist()
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Phase 3: Try multiple starting points with Nelder-Mead
    for x0_try in [
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.6, -0.1, 0.05, 0.0, 0.0, 0.0],
        [0.5, 0.15, -0.05, 0.0, 0.0, 0.0],
    ]:
        if time.time() - _start_time > _DEADLINE - 15:
            break
        try:
            result = minimize(lambda c: _evaluate_error(c)[0], x0_try, method='Nelder-Mead', 
                             options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            actual_score, _ = _evaluate_error(result.x)
            _update_best(result.x, actual_score)
        except:
            pass
    
    # Phase 4: Final high-precision polish with very tight tolerances
    if best_result is not None and time.time() - _start_time < _DEADLINE - 10:
        try:
            result = minimize(lambda c: _evaluate_error(c)[0], best_result, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            actual_score, _ = _evaluate_error(result.x)
            _update_best(result.x, actual_score)
        except:
            pass
    
    if best_result is not None:
        return np.clip(best_result, -10, 10).tolist()
    
    # Fallback to safe constant gamma
    return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
