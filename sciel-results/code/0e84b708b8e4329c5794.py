"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, simpson
import time

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 80.0  # 80s safety margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    a_safe = np.where(a > 1e-10, a, 1e-10)
    # DE density ratio: rho_de(a)/rho_de(1) = a^(3(1+w0+wa)) * exp(3*wa*(1-a))
    de_factor = np.power(a_safe, 3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (1.0 - a_safe))
    Om = Om0 / (Om0 / np.power(a_safe, 3.0) + (1.0 - Om0) * de_factor)
    return Om

def _gamma_of_a(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_of_a(a, coeffs, Om0, w0, wa)
    gamma = np.clip(gamma, 0.1, 1.2)
    Om_safe = np.maximum(Om, 1e-300)
    return np.power(Om_safe, gamma)

def _D_normalized(a, coeffs, Om0, w0, wa, npts=100):
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

def _ode_reference_f(a, Om0, w0, wa, npts=200):
    """Solve the GR ODE for f(a) = d ln D / d ln a."""
    # ODE in x = ln(a): df/dx + f^2 + 0.5*(3*Om - 2)*f - 1.5*Om = 0
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a[a > 0]), 0.01)
    x_min = np.log(a_min)
    
    # Initial condition at a=1
    Om_at_1 = _Om_of_a(1.0, Om0, w0, wa)
    f0 = 0.5 * Om_at_1 + 0.5 * (1.0 - Om_at_1) * (1.0 + w0 + wa)
    
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        dfdx = -f**2 - 0.5*(3.0*Om - 2.0)*f + 1.5*Om
        return [dfdx]
    
    x_grid = np.linspace(x_min, 0.0, npts)
    sol = solve_ivp(ode, [x_min, 0.0], [f0], t_eval=x_grid, method='RK45', rtol=1e-8, atol=1e-10)
    
    f_ref = np.zeros(len(a))
    for i, ai in enumerate(a):
        if ai >= 1.0:
            f_ref[i] = f0
        elif ai <= 0:
            f_ref[i] = 0.0
        else:
            x_val = np.log(ai)
            f_ref[i] = np.interp(x_val, x_grid, sol.y[0])
    return f_ref

def _ode_reference_D(a, Om0, w0, wa, npts=200):
    """Compute normalized growth factor from ODE reference f."""
    a = np.asarray(a, dtype=float)
    a_min = max(np.min(a[a > 0]), 0.01)
    x_min = np.log(a_min)
    x_grid = np.linspace(x_min, 0.0, npts)
    a_grid = np.exp(x_grid)
    f_grid = _ode_reference_f(a_grid, Om0, w0, wa, npts)
    
    # Cumulative integral from x_min to x
    cum_int = np.zeros_like(x_grid)
    for i in range(1, len(x_grid)):
        dx = x_grid[i] - x_grid[i-1]
        cum_int[i] = cum_int[i-1] + 0.5 * (f_grid[i] + f_grid[i-1]) * dx
    
    total = cum_int[-1]  # integral from x_min to 0
    
    D = np.ones_like(a)
    for i, ai in enumerate(a):
        if ai >= 1.0:
            D[i] = 1.0
        elif ai <= 0:
            D[i] = 0.0
        else:
            x_val = np.log(ai)
            int_at_x = np.interp(x_val, x_grid, cum_int)
            int_val = total - int_at_x  # integral from x_val to 0
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

# Pre-compute reference values for speed
_REF_CACHE = {}

def _get_reference(Om0, w0, wa):
    key = (Om0, w0, wa)
    if key not in _REF_CACHE:
        f_ref = _ode_reference_f(_A_VALUES, Om0, w0, wa)
        D_ref = _ode_reference_D(_A_VALUES, Om0, w0, wa)
        _REF_CACHE[key] = (f_ref, D_ref)
    return _REF_CACHE[key]

def _objective(coeffs):
    """Max relative error across all validation models."""
    if time.time() > _DEADLINE:
        return 1e6
    
    max_err = 0.0
    for Om0, w0, wa in _VALIDATION_MODELS:
        f_ref, D_ref = _get_reference(Om0, w0, wa)
        
        f_approx = _f_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        D_approx = _D_normalized(_A_VALUES, coeffs, Om0, w0, wa)
        
        # Check gamma constraint
        gamma_vals = _gamma_of_a(_A_VALUES, coeffs, Om0, w0, wa)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1e6
        
        mask_f = np.abs(f_ref) > 1e-6
        mask_D = np.abs(D_ref) > 1e-6
        
        if np.any(mask_f):
            err_f = np.max(np.abs(f_approx[mask_f] - f_ref[mask_f]) / np.abs(f_ref[mask_f]))
        else:
            err_f = 0.0
        
        if np.any(mask_D):
            err_D = np.max(np.abs(D_approx[mask_D] - D_ref[mask_D]) / np.abs(D_ref[mask_D]))
        else:
            err_D = 0.0
        
        max_err = max(max_err, err_f, err_D)
    
    return max_err

_BEST_COEFFS = None

def _optimize():
    """Run optimization with time budget."""
    global _BEST_COEFFS
    t0 = time.time()
    deadline = t0 + 70.0  # 70s for optimization, 10s margin
    
    # Pre-compute references
    for Om0, w0, wa in _VALIDATION_MODELS:
        _get_reference(Om0, w0, wa)
    
    # Initial guess: constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_val = _objective(x0)
    
    # Nelder-Mead
    if time.time() < deadline - 20:
        result = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    
    # L-BFGS-B with bounds
    if time.time() < deadline - 10:
        bounds = [(-10, 10)] * 6
        result2 = minimize(_objective, best_x, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
        if result2.fun < best_val:
            best_val = result2.fun
            best_x = result2.x
    
    # Local refinement
    for _ in range(5):
        if time.time() >= deadline - 5:
            break
        result3 = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'initial_simplex': 0.01 * np.ones(6)})
        if result3.fun < best_val:
            best_val = result3.fun
            best_x = result3.x
        else:
            break
    
    _BEST_COEFFS = list(np.clip(best_x, -10, 10))
    return _BEST_COEFFS

def construct():
    """Return optimized 6 coefficients."""
    if _BEST_COEFFS is not None:
        return _BEST_COEFFS
    else:
        return _optimize()

# Run optimization at import time
try:
    _optimize()
except Exception:
    _BEST_COEFFS = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
