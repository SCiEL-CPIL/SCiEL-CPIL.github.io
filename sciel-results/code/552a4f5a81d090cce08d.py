"""Compact dark-energy growth approximation: fit 6 coefficients to a GR reference."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

_start_time = time.time()
_DEADLINE = 85.0

# Precompute reference data using ODE integration
_cosmologies = [
    (0.3, -1.0, 0.0),
    (0.3, -0.9, 0.1),
    (0.3, -1.1, 0.2),
    (0.3, -0.8, -0.2),
    (0.3, -1.2, 0.3),
    (0.2, -1.0, 0.0),
    (0.4, -1.0, 0.0),
    (0.3, -0.5, 0.0),
    # Additional diverse models for better generalization
    (0.3, -1.5, 1.0),
    (0.3, -0.5, -0.5),
    (0.2, -1.2, 0.8),
    (0.4, -0.7, -0.3),
    (0.28, -1.0, 0.1),
    (0.32, -0.95, -0.15),
    (0.3, -1.0, 0.3),
    (0.3, -0.85, 0.25),
    (0.27, -1.1, 0.4),
    (0.33, -0.75, -0.25),
]

_a_grid = np.linspace(0.05, 1.0, 50)
_x_grid = np.log(_a_grid)
_ref_data = {}

def _precompute_references():
    """Precompute reference f and D values using ODE integration."""
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
        x_vals = _x_grid
        x_min = min(x_vals)
        
        try:
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
        except:
            # Fallback: use analytic approximation
            Om_a = Om0 * a_vals**(-3)
            f_vals = Om_a**0.55
            integral = cumulative_trapezoid(f_vals, _x_grid, initial=0.0)
            D_norm = np.exp(-(integral[-1] - integral))
            _ref_data[(Om0, w0, wa)] = (f_vals, D_norm)

def _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a_vals):
    """Fast computation of f and D for given coefficients."""
    a_vals = np.asarray(a_vals)
    Om_a = Om0 * a_vals**(-3)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_vals)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    f_vals = Om_a**gamma
    
    x_vals = _x_grid
    
    # Compute integral from a to 1 using cumulative trapezoid
    integral_forward = cumulative_trapezoid(f_vals, x_vals, initial=0.0)
    integral_to_0 = integral_forward[-1] - integral_forward
    
    D_norm = np.exp(-integral_to_0)
    
    return f_vals, D_norm

def _evaluate_error(coeffs):
    """Compute maximum relative error across all models."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    for Om0, w0, wa in _cosmologies:
        if (Om0, w0, wa) not in _ref_data:
            continue
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod = _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, _a_grid)
        
        err_f = np.max(np.abs((f_mod - f_ref) / f_ref))
        err_D = np.max(np.abs((D_mod - D_ref) / D_ref))
        
        max_err = max(max_err, err_f, err_D)
    
    return max_err

def _evaluate_error_smooth(coeffs, temperature=0.03):
    """Smooth surrogate using log-sum-exp for the max operation."""
    c0, c1, c2, c3, c4, c5 = coeffs
    all_errors = []
    for Om0, w0, wa in _cosmologies:
        if (Om0, w0, wa) not in _ref_data:
            continue
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod = _compute_model_fast(c0, c1, c2, c3, c4, c5, Om0, w0, wa, _a_grid)
        
        err_f = np.abs((f_mod - f_ref) / f_ref)
        err_D = np.abs((D_mod - D_ref) / D_ref)
        all_errors.append(err_f)
        all_errors.append(err_D)
    
    if not all_errors:
        return 1e10
    
    # Stack all errors
    all_errs = np.concatenate(all_errors)
    
    # Soft max via log-sum-exp
    max_val = np.max(all_errs)
    if max_val < 1e-15:
        return max_val
    log_sum_exp = np.log(np.sum(np.exp((all_errs - max_val) / temperature))) * temperature + max_val
    return log_sum_exp

def _is_valid(coeffs, cosmologies, a_grid):
    """Check if gamma stays within [0.1, 1.2] for all models."""
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
    
    def objective(coeffs):
        if time.time() - _start_time > _DEADLINE - 5:
            return 1e10
        return _evaluate_error(coeffs)
    
    def objective_smooth(coeffs, temp=0.03):
        if time.time() - _start_time > _DEADLINE - 5:
            return 1e10
        return _evaluate_error_smooth(coeffs, temperature=temp)
    
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Differential evolution for global search
    if time.time() - _start_time < _DEADLINE - 30:
        try:
            bounds = [(-1.0, 1.5), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
            de_result = differential_evolution(
                objective_smooth, bounds, 
                maxiter=100, popsize=30, tol=1e-8,
                seed=42, polish=False, 
                options={'init': 'sobol', 'polish': False}
            )
            if de_result.fun < best_score and _is_valid(de_result.x, _cosmologies, _a_grid):
                best_score = de_result.fun
                best_result = de_result.x
        except Exception as e:
            pass
    
    # Phase 2: Multi-start L-BFGS-B on smooth objective
    starting_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.05, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
        [0.55, 0.0, 0.05, -0.05, 0.0, 0.03],
        [0.5, 0.1, -0.05, 0.1, 0.0, 0.0],
        [0.6, -0.1, 0.1, 0.0, 0.0, 0.05],
        [0.55, 0.05, 0.0, 0.05, -0.05, 0.0],
    ]
    
    if best_result is not None:
        starting_points.append(list(best_result))
    
    for x0_try in starting_points:
        if time.time() - _start_time > _DEADLINE - 20:
            break
        try:
            result = minimize(objective_smooth, x0_try, method='L-BFGS-B', 
                             options={'maxiter': 200, 'gtol': 1e-9, 'ftol': 1e-11})
            if result.fun < best_score and _is_valid(result.x, _cosmologies, _a_grid):
                best_score = result.fun
                best_result = result.x
        except:
            pass
    
    # Phase 3: Polish with exact (nonsmooth) objective using Nelder-Mead
    if best_result is not None:
        if time.time() - _start_time < _DEADLINE - 10:
            try:
                result = minimize(objective, best_result, method='Nelder-Mead', 
                                 options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if result.fun < best_score and _is_valid(result.x, _cosmologies, _a_grid):
                    best_score = result.fun
                    best_result = result.x
            except:
                pass
        
        # Additional polish with L-BFGS-B on exact objective
        if time.time() - _start_time < _DEADLINE - 8:
            try:
                result = minimize(objective, best_result, method='L-BFGS-B', 
                                 options={'maxiter': 300, 'gtol': 1e-11, 'ftol': 1e-13})
                if result.fun < best_score and _is_valid(result.x, _cosmologies, _a_grid):
                    best_score = result.fun
                    best_result = result.x
            except:
                pass
    
    # Phase 4: Random restarts around best solution
    if best_result is not None and time.time() - _start_time < _DEADLINE - 5:
        rng = np.random.default_rng(12345)
        for _ in range(100):
            if time.time() - _start_time > _DEADLINE - 5:
                break
            c = rng.normal(
                loc=best_result,
                scale=[0.05, 0.2, 0.1, 0.2, 0.2, 0.15]
            )
            c = np.clip(c, -10, 10)
            s = _evaluate_error(c)
            if s < best_score and _is_valid(c, _cosmologies, _a_grid):
                best_score = s
                best_result = c.copy()
                # Quick local polish
                if time.time() - _start_time < _DEADLINE - 5:
                    try:
                        res = minimize(objective, best_result, method='Nelder-Mead',
                                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                        if res.fun < best_score and _is_valid(res.x, _cosmologies, _a_grid):
                            best_score = res.fun
                            best_result = res.x
                    except:
                        pass
    
    # Phase 5: Final high-precision polish
    if best_result is not None and time.time() - _start_time < _DEADLINE - 3:
        try:
            result = minimize(objective, best_result, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if result.fun < best_score and _is_valid(result.x, _cosmologies, _a_grid):
                best_score = result.fun
                best_result = result.x
        except:
            pass
    
    if best_result is not None:
        return np.clip(best_result, -10, 10).tolist()
    
    # Fallback to safe constant gamma
    return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
