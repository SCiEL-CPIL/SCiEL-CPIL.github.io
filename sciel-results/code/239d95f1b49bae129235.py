"""Growth-index approximation with combined optimization strategy."""
import time
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings("ignore")

_DEADLINE_SECONDS = 90.0
_DEADLINE_FRACTION = 0.82

# EVOLVE-BLOCK-START

_models = None
_a_grid = None
_ode_cache = {}

def _get_models():
    global _models
    if _models is None:
        _models = [
            (0.3, -0.5, 0.5),
            (0.3, -1.0, 0.0),
            (0.3, -0.9, 0.3),
            (0.25, -0.8, 0.2),
            (0.35, -1.0, 0.5),
            (0.3, -0.7, -0.2),
            (0.28, -1.0, 0.8),
            (0.32, -0.6, 0.1),
        ]
    return _models

def _get_a_grid():
    global _a_grid
    if _a_grid is None:
        _a_grid = np.logspace(-1, 0, 50)
    return _a_grid

def _h2_and_dh2dx(a, Om0, w0, wa):
    """Compute H^2/H0^2 and d(H^2)/d(ln a) for CPL dark energy."""
    Om_m = Om0 * a**3
    Om_de = 1.0 - Om_m
    if Om_de < 1e-15:
        Om_de = 1e-15
    # DE density factor: exp(3*(1+w0+wa)*ln(a) + 3*wa*(1-a))
    # = a^(3*(1+w0+wa)) * exp(3*wa*(1-a))
    # But actually: rho_de(a)/rho_de(1) = a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
    # H^2/H0^2 = Om_m + Om_de * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
    exponent = -3.0*(1.0 + w0 + wa) * math.log(a) - 3.0*wa*(1.0 - a)
    de_factor = math.exp(exponent)
    h2 = Om_m + Om_de * de_factor
    
    # d(h2)/d(ln a):
    # d(Om_m)/dx = 3*Om_m
    # d(Om_de * de_factor)/dx = -3*Om_m * de_factor + Om_de * de_factor * (-3*(1+w0+wa) - 3*wa*(-1))
    # = -3*Om_m * de_factor + Om_de * de_factor * (-3*(1+w0+wa) + 3*wa)
    # Wait, let me redo: de_factor = exp(-3(1+w0+wa)*x - 3*wa*(1-exp(x)))
    # d(de_factor)/dx = de_factor * (-3(1+w0+wa) - 3*wa*exp(x))
    # = de_factor * (-3*(1+w0+wa) - 3*wa*a)
    d_de_factor_dx = de_factor * (-3.0*(1.0 + w0 + wa) - 3.0*wa*a)
    dh2_dx = 3.0*Om_m + (-3.0*Om_m * de_factor + Om_de * d_de_factor_dx)
    
    return h2, dh2_dx

def _solve_ode_for_model(Om0, w0, wa, a_grid):
    """Solve the full GR ODE for f and D for a given model."""
    key = (round(Om0, 10), round(w0, 10), round(wa, 10), tuple(a_grid[:3]))
    if key in _ode_cache:
        return _ode_cache[key]
    
    def rhs(x, y):
        a_val = math.exp(x)
        f_val = y[0]
        D_val = y[1]
        
        h2, dh2_dx = _h2_and_dh2dx(a_val, Om0, w0, wa)
        if h2 <= 0:
            return [0.0, 0.0]
        
        # dlnH/dx = 0.5 * dh2/dx / h2
        dlnH_dx = 0.5 * dh2_dx / h2
        
        # 4*pi*G*rho_m/H^2 = 3*Om_m/h2
        Om_m = Om0 * a_val**3
        source = 3.0 * Om_m / h2
        
        # f' = -f^2 - (3 + dlnH/dx)*f + source
        df_dx = -f_val**2 - (3.0 + dlnH_dx) * f_val + source
        dD_dx = f_val * D_val
        
        return [df_dx, dD_dx]
    
    a_min = np.min(a_grid)
    a_max = np.max(a_grid)
    x_min = math.log(a_min)
    x_max = math.log(a_max)
    
    # Initial conditions at a=1 (x=0): f(1) ≈ 0.4, D(1) = 1
    # We need to integrate both forward and backward from x=0
    # For x > 0 (a > 1): integrate forward
    # For x < 0 (a < 1): integrate backward
    
    all_xs = np.sort(np.concatenate([np.linspace(x_min, 0, 50), np.linspace(0, x_max, 50)]))
    all_xs = np.unique(all_xs)
    
    # Integrate backward from 0 to x_min
    sol_back = solve_ivp(rhs, [0.0, x_min], [0.4, 1.0], method='DOP853',
                         rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.005)
    
    # Integrate forward from 0 to x_max
    sol_fwd = solve_ivp(rhs, [0.0, x_max], [0.4, 1.0], method='DOP853',
                        rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.005)
    
    f_ref = np.zeros(len(a_grid))
    D_ref = np.zeros(len(a_grid))
    
    for i, a in enumerate(a_grid):
        x = math.log(a)
        if x <= 0:
            if sol_back.success:
                f_ref[i] = sol_back.sol(x)[0]
                D_ref[i] = sol_back.sol(x)[1]
            else:
                f_ref[i] = 0.4
                D_ref[i] = 1.0
        else:
            if sol_fwd.success:
                f_ref[i] = sol_fwd.sol(x)[0]
                D_ref[i] = sol_fwd.sol(x)[1]
            else:
                f_ref[i] = 0.4
                D_ref[i] = 1.0
    
    _ode_cache[key] = (f_ref, D_ref)
    return f_ref, D_ref

def _compute_gamma_vec(c, a_grid, Om0, w0, wa):
    """Vectorized gamma computation."""
    Om_a = Om0 * a_grid**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a_grid)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_grid + c[5]*u*q

def _f_approx_vec(c, a_grid, Om0, w0, wa):
    """Vectorized f approximation."""
    g = _compute_gamma_vec(c, a_grid, Om0, w0, wa)
    Om_a = Om0 * a_grid**3
    return np.power(Om_a, g)

def _D_approx_vec(c, a_grid, Om0, w0, wa, n_int=500):
    """Compute D(a)/D(1) using the approximation."""
    ln_a = np.log(a_grid)
    D_ratio = np.ones_like(a_grid)
    for i, la in enumerate(ln_a):
        if la >= 0:
            continue
        xs = np.linspace(la, 0.0, n_int)
        a_vals = np.exp(xs)
        f_vals = _f_approx_vec(c, a_vals, Om0, w0, wa)
        if hasattr(np, 'trapezoid'):
            integral = np.trapezoid(f_vals, xs)
        else:
            integral = np.trapz(f_vals, xs)
        D_ratio[i] = np.exp(-integral)
    return D_ratio

def _validate_gamma(c, a_grid, Om0, w0, wa):
    """Check gamma in [0.1, 1.2]."""
    g = _compute_gamma_vec(c, a_grid, Om0, w0, wa)
    return bool(np.all(g >= 0.1 - 1e-9) and np.all(g <= 1.2 + 1e-9))

def _objective(c, models, a_grid):
    """Compute worst relative error across all models."""
    max_err = 0.0
    for Om0, w0, wa in models:
        if not _validate_gamma(c, a_grid, Om0, w0, wa):
            return 1e10
        
        f_ref, D_ref = _solve_ode_for_model(Om0, w0, wa, a_grid)
        
        f_approx = _f_approx_vec(c, a_grid, Om0, w0, wa)
        D_approx = _D_approx_vec(c, a_grid, Om0, w0, wa, n_int=500)
        
        # Relative errors for f
        denom_f = np.maximum(np.abs(f_ref), 1e-12)
        rel_err_f = np.abs(f_approx - f_ref) / denom_f
        
        # Relative errors for D
        denom_D = np.maximum(np.abs(D_ref), 1e-12)
        rel_err_D = np.abs(D_approx - D_ref) / denom_D
        
        model_err = max(np.max(rel_err_f), np.max(rel_err_D))
        if model_err > max_err:
            max_err = model_err
        if max_err >= 0.5:  # Early termination if error is already large
            return 1e10
    
    return 100.0 * max_err

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS * _DEADLINE_FRACTION
    
    models = _get_models()
    a_grid = _get_a_grid()
    
    # Start with improved seed: use a better initial gamma
    # For LambdaCDM, gamma ≈ 0.55 is good; for other models, add small corrections
    best_c = np.array([0.55, 0.05, 0.0, 0.0, 0.0, 0.0])
    best_obj = _objective(best_c, models, a_grid)
    
    # Also try pure constant
    c_const = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    obj_const = _objective(c_const, models, a_grid)
    if obj_const < best_obj:
        best_c = c_const
        best_obj = obj_const
    
    def obj_func(c):
        if time.time() > deadline:
            return 1e10
        c = np.clip(c, -10.0, 10.0)
        return _objective(c, models, a_grid)
    
    bounds = [(-10, 10)] * 6
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 25:
        try:
            result_de = differential_evolution(
                obj_func, bounds, seed=42, maxiter=30, tol=1e-5,
                polish=True, init='sobol', popsize=15,
                mutation=(0.3, 0.9), recombination=0.7
            )
            if result_de.fun < best_obj:
                best_c = np.clip(result_de.x, -10, 10)
                best_obj = result_de.fun
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement from multiple starts
    np.random.seed(123)
    start_points = [best_c.copy()]
    for _ in range(10):
        c0 = best_c + np.random.randn(6) * 0.15
        c0 = np.clip(c0, -10, 10)
        start_points.append(c0)
    
    for c0 in start_points:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(obj_func, c0, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-9,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_c = np.clip(res.x, -10, 10)
                best_obj = res.fun
        except Exception:
            pass
    
    # Phase 3: Fine L-BFGS-B polish
    if time.time() < deadline - 5:
        try:
            res2 = minimize(obj_func, best_c, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.fun < best_obj:
                best_c = np.clip(res2.x, -10, 10)
                best_obj = res2.fun
        except Exception:
            pass
    
    # Phase 4: Final fine Nelder-Mead
    if time.time() < deadline - 2:
        try:
            res3 = minimize(obj_func, best_c, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-11,
                                    'adaptive': True})
            if res3.fun < best_obj:
                best_c = np.clip(res3.x, -10, 10)
                best_obj = res3.fun
        except Exception:
            pass
    
    best_c = np.clip(best_c, -10.0, 10.0)
    if not np.all(np.isfinite(best_c)):
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
