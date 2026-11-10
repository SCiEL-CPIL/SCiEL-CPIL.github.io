"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, simpson
import time

# EVOLVE-BLOCK-START

_T0 = time.time()
_DEADLINE = 82.0  # seconds, safety margin before 90s kill

def _elapsed():
    return time.time() - _T0

def _Om_of_a(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a_safe = np.where(a > 1e-12, a, 1e-12)
    E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a_safe))
    return Om0 * a_safe**(-3) / E2

def _gamma_of_a(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1 - Om
    u = 1 + w0 + wa * (1 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_of_a(a, coeffs, Om0, w0, wa)
    gamma = np.clip(gamma, 0.1, 1.2)
    return Om ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa, npts=150):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    a = np.asarray(a, dtype=float)
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0 - 1e-12:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_arr = np.linspace(np.log(ai), 0.0, npts)
            a_arr = np.exp(x_arr)
            f_arr = _f_of_a(a_arr, coeffs, Om0, w0, wa)
            integral = simpson(f_arr, x=x_arr)
            D[i] = np.exp(-integral)
    return D

def _ode_solve_f(Om0, w0, wa, a_grid, npts=300):
    """Solve the GR ODE for f(a) on a fixed grid, return f at a_grid."""
    a_grid = np.asarray(a_grid, dtype=float)
    a_min = max(np.min(a_grid), 0.01)
    x_min = np.log(a_min)
    
    Om_at_1 = _Om_of_a(1.0, Om0, w0, wa)
    f0 = 0.5*Om_at_1 + 0.5*(1-Om_at_1)*(1+w0+wa)
    
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        dfdx = -f**2 - 0.5*(3*Om - 2)*f + 0.5*3*Om
        return [dfdx]
    
    x_grid = np.linspace(x_min, 0.0, npts)
    sol = solve_ivp(ode, [x_min, 0.0], [f0], t_eval=x_grid, method='RK45', rtol=1e-9, atol=1e-11)
    
    # Interpolate to a_grid
    f_ref = np.zeros(len(a_grid))
    for i, ai in enumerate(a_grid):
        if ai >= 1.0:
            f_ref[i] = f0
        elif ai <= 0:
            f_ref[i] = 0.0
        else:
            x_val = np.log(ai)
            idx = np.searchsorted(x_grid, x_val)
            if idx == 0:
                f_ref[i] = sol.y[0, 0]
            elif idx >= len(x_grid):
                f_ref[i] = sol.y[0, -1]
            else:
                x0, x1 = x_grid[idx-1], x_grid[idx]
                f0v, f1v = sol.y[0, idx-1], sol.y[0, idx]
                t = (x_val - x0) / (x1 - x0)
                f_ref[i] = f0v + t * (f1v - f0v)
    return f_ref

def _ode_reference_f(a, Om0, w0, wa, npts=300):
    """Solve the GR ODE for f(a) and return f at given a values."""
    return _ode_solve_f(Om0, w0, wa, a, npts)

def _ode_reference_D(a, Om0, w0, wa, npts=300):
    """Solve the GR ODE for D(a) and return normalized D(a)/D(1)."""
    a = np.asarray(a, dtype=float)
    f_ref = _ode_reference_f(a, Om0, w0, wa, npts)
    
    D = np.ones_like(a)
    a_min = max(np.min(a), 0.01)
    x_min = np.log(a_min)
    x_grid = np.linspace(x_min, 0.0, npts)
    
    a_grid = np.exp(x_grid)
    f_grid = _ode_solve_f(Om0, w0, wa, a_grid, npts)
    
    from scipy.integrate import cumulative_trapezoid
    integral_from_min = cumulative_trapezoid(f_grid, x_grid, initial=0.0)
    total = integral_from_min[-1]
    
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_val = np.log(ai)
            idx = np.searchsorted(x_grid, x_val)
            if idx == 0:
                int_val = total - integral_from_min[0]
            elif idx >= len(x_grid):
                int_val = 0.0
            else:
                x0, x1 = x_grid[idx-1], x_grid[idx]
                i0, i1 = integral_from_min[idx-1], integral_from_min[idx]
                t = (x_val - x0) / (x1 - x0)
                int_at_x = i0 + t * (i1 - i0)
                int_val = total - int_at_x
            D[i] = np.exp(int_val)
    return D

# Validation cosmologies
_VALIDATION_MODELS = [
    (0.30, 0.0, 0.0),
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -0.9, 0.3),
    (0.35, -1.1, 0.2),
]

_A_VALUES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

def _compute_error(coeffs, Om0, w0, wa, a_vals, npts=200):
    """Compute max relative error for f and D."""
    f_approx = _f_of_a(a_vals, coeffs, Om0, w0, wa)
    f_ref = _ode_reference_f(a_vals, Om0, w0, wa, npts)
    
    D_approx = _D_normalized(a_vals, coeffs, Om0, w0, wa)
    D_ref = _ode_reference_D(a_vals, Om0, w0, wa, npts)
    
    mask_f = np.abs(f_ref) > 1e-6
    mask_D = np.abs(D_ref) > 1e-6
    
    err_f = np.max(np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f])) if np.any(mask_f) else 0.0
    err_D = np.max(np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D])) if np.any(mask_D) else 0.0
    
    return max(err_f, err_D)

def _objective(coeffs):
    """Objective function: max error across all validation models."""
    max_err = 0.0
    for Om0, w0, wa in _VALIDATION_MODELS:
        err = _compute_error(coeffs, Om0, w0, wa, _A_VALUES)
        max_err = max(max_err, err)
    return max_err

def _objective_soft(coeffs, temperature=0.02):
    """Smooth surrogate for max error using log-sum-exp."""
    errs = []
    for Om0, w0, wa in _VALIDATION_MODELS:
        err = _compute_error(coeffs, Om0, w0, wa, _A_VALUES)
        errs.append(err)
    errs = np.array(errs)
    m = np.max(errs)
    if m < 1e-12:
        return m
    log_sum = m + np.log(np.sum(np.exp((errs - m) / temperature)))
    return log_sum * temperature

def _multi_model_initial_guess():
    """Get initial coefficients by fitting to multiple models simultaneously."""
    all_A = []
    all_gamma = []
    
    for Om0, w0, wa in _VALIDATION_MODELS:
        a_grid = np.linspace(0.1, 1.0, 30)
        f_ref = _ode_reference_f(a_grid, Om0, w0, wa, npts=200)
        Om = _Om_of_a(a_grid, Om0, w0, wa)
        gamma_exact = np.log(np.maximum(f_ref / Om, 1e-300))
        
        q = 1 - Om
        u = 1 + w0 + wa * (1 - a_grid)
        
        A = np.column_stack([
            np.ones_like(a_grid),
            q,
            q**2,
            u,
            wa * a_grid,
            u * q
        ])
        
        all_A.append(A)
        all_gamma.append(gamma_exact)
    
    A_all = np.vstack(all_A)
    gamma_all = np.concatenate(all_gamma)
    
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_all, gamma_all, rcond=None)
    return coeffs

def _optimize():
    """Run optimization with time budget."""
    t0 = time.time()
    deadline = t0 + 75.0
    
    # Use multi-model fit as initial guess
    x0 = _multi_model_initial_guess()
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Phase 1: Nelder-Mead with smooth surrogate
    if time.time() < deadline - 30:
        result = minimize(_objective_soft, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
        if time.time() < deadline - 30 and result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    
    # Phase 2: L-BFGS-B with bounds on exact objective
    if time.time() < deadline - 20:
        bounds = [(-5, 5)] * 6
        result2 = minimize(_objective, best_x, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
        if result2.fun < best_val:
            best_val = result2.fun
            best_x = result2.x
    
    # Phase 3: Multiple local refinement passes with Nelder-Mead
    for iteration in range(8):
        if time.time() >= deadline - 5:
            break
        try:
            result3 = minimize(_objective, best_x, method='Nelder-Mead',
                              options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'initial_simplex': 0.003 * np.ones(6)})
            if result3.fun < best_val:
                best_val = result3.fun
                best_x = result3.x
            else:
                np.random.seed(iteration)
                perturbed = best_x + 0.005 * np.random.randn(6)
                perturbed = np.clip(perturbed, -5, 5)
                result4 = minimize(_objective, perturbed, method='Nelder-Mead',
                                  options={'maxiter': 40, 'xatol': 1e-10, 'fatol': 1e-12})
                if result4.fun < best_val:
                    best_val = result4.fun
                    best_x = result4.x
                else:
                    break
        except Exception:
            break
    
    # Final polish
    if time.time() < deadline - 2:
        try:
            result5 = minimize(_objective, best_x, method='Nelder-Mead',
                              options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-14,
                                       'initial_simplex': 0.001 * np.ones(6)})
            if result5.fun < best_val:
                best_val = result5.fun
                best_x = result5.x
        except Exception:
            pass
    
    return list(best_x)

_BEST_COEFFS = None

def construct():
    global _BEST_COEFFS
    if _BEST_COEFFS is not None:
        return _BEST_COEFFS
    try:
        _BEST_COEFFS = _optimize()
    except Exception:
        _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    return _BEST_COEFFS

# Run optimization at import time
try:
    _BEST_COEFFS = _optimize()
except Exception:
    _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# EVOLVE-BLOCK-END
