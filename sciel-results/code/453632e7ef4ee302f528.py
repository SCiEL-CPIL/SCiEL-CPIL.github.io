"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

_DEADLINE_FRACTION = 0.85
_DEADLINE_SECONDS = 90.0

def _compute_gamma(c, a, Om, w0, wa):
    """Compute gamma(a) for given coefficients and scale factor."""
    Om_a = Om * a**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_from_gamma(c, a, Om, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    g = _compute_gamma(c, a, Om, w0, wa)
    Om_a = Om * a**3
    # Clip Om_a to avoid issues
    Om_a = np.maximum(Om_a, 1e-15)
    return np.power(Om_a, g)

def _normalized_growth(c, a_grid, Om, w0, wa, n_int=500):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    ln_a = np.log(a_grid)
    D_ratio = np.ones_like(a_grid)
    for i, la in enumerate(ln_a):
        if la >= 0:
            D_ratio[i] = 1.0
            continue
        xs = np.linspace(la, 0.0, n_int)
        a_vals = np.exp(xs)
        f_vals = _f_from_gamma(c, a_vals, Om, w0, wa)
        integral = np.trapz(f_vals, xs)
        D_ratio[i] = np.exp(-integral)
    return D_ratio

def _validate_gamma(c, a_grid, Om, w0, wa):
    """Check that gamma is in [0.1, 1.2] for all points."""
    g = _compute_gamma(c, a_grid, Om, w0, wa)
    return np.all(g >= 0.1 - 1e-9) and np.all(g <= 1.2 + 1e-9)

def _solve_ode_f_ref(Om, w0, wa, a_grid):
    """Solve the ODE for f and compute reference D ratio."""
    # f' = -(1/2)f^2 - (3/2)(1+w)f
    # Integrate from a=1 (ln a = 0) backwards to a_min
    
    a_min = np.min(a_grid)
    ln_a_min = np.log(a_min)
    
    def rhs(lna, y):
        a_val = np.exp(lna)
        w_val = w0 + wa * (1.0 - a_val)
        f_val = y[0]
        return [-(0.5)*f_val**2 - 1.5*(1.0+w_val)*f_val]
    
    # Initial condition: at a=1, f ≈ Om (for small a, f≈Om; at a=1, f≈0.5*Om+...)
    # Actually at a=1, f(1) ≈ 0.5*Om + small corrections
    # Let's use f(1) = Om * 0.5 + 0.5*Om*(1-Om) as a reasonable starting point
    # Better: solve forward from small a to a=1 to get accurate f(1)
    
    # Solve from a=0.01 to a=1
    a_start = max(0.01, a_min)
    ln_a_start = np.log(a_start)
    
    sol = solve_ivp(rhs, [ln_a_start, 0.0], [Om * 0.5], method='DOP853',
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.05)
    
    if not sol.success:
        return None, None
    
    # Get f at grid points
    ln_a_grid = np.log(a_grid)
    f_ref = np.array([sol.sol(la)[0] for la in ln_a_grid])
    
    # Compute D ratio using reference f
    D_ref_ratio = np.ones_like(a_grid)
    n_int = 300
    for i, la in enumerate(ln_a_grid):
        if la >= 0:
            D_ref_ratio[i] = 1.0
            continue
        xs = np.linspace(la, 0.0, n_int)
        f_vals = np.array([sol.sol(x)[0] for x in xs])
        integral = np.trapz(f_vals, xs)
        D_ref_ratio[i] = np.exp(-integral)
    
    return f_ref, D_ref_ratio

# EVOLVE-BLOCK-END

# Global cache
_models = None
_a_grid = None
_ref_cache = {}

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
        _a_grid = np.logspace(-1.5, 0, 60)
    return _a_grid

def _get_ref(model_idx, Om, w0, wa, a_grid):
    global _ref_cache
    key = model_idx
    if key not in _ref_cache:
        f_ref, D_ref = _solve_ode_f_ref(Om, w0, wa, a_grid)
        _ref_cache[key] = (f_ref, D_ref)
    return _ref_cache[key]

def _objective(c, Om, w0, wa, a_grid, model_idx):
    """Compute the worst relative error for f and D."""
    # Check gamma bounds on a dense grid
    a_dense = np.linspace(0.01, 1.0, 200)
    if not _validate_gamma(c, a_dense, Om, w0, wa):
        return 1e10
    
    f_approx = _f_from_gamma(c, a_grid, Om, w0, wa)
    
    f_ref, D_ref_ratio = _get_ref(model_idx, Om, w0, wa, a_grid)
    if f_ref is None:
        return 1e10
    
    # Relative error for f
    denom_f = np.maximum(np.abs(f_ref), 1e-12)
    rel_err_f = np.abs(f_approx - f_ref) / denom_f
    
    # For D
    D_approx_ratio = _normalized_growth(c, a_grid, Om, w0, wa, n_int=300)
    
    denom_D = np.maximum(np.abs(D_ref_ratio), 1e-12)
    rel_err_D = np.abs(D_approx_ratio - D_ref_ratio) / denom_D
    
    worst_err = max(np.max(rel_err_f), np.max(rel_err_D))
    return 100.0 * worst_err

def _multi_model_objective(c, models, a_grid):
    """Evaluate worst error across all models."""
    max_err = 0.0
    for i, (Om, w0, wa) in enumerate(models):
        err = _objective(c, Om, w0, wa, a_grid, i)
        if err > max_err:
            max_err = err
        if err >= 1e10:
            return 1e10
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS * _DEADLINE_FRACTION
    
    models = _get_models()
    a_grid = _get_a_grid()
    
    # Start with the seed
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_obj = _multi_model_objective(best_c, models, a_grid)
    
    # Define the objective function for the optimizer
    def obj(c):
        if time.time() > deadline:
            return 1e10
        c = np.clip(c, -10.0, 10.0)
        return _multi_model_objective(c, models, a_grid)
    
    # Try multiple starting points
    np.random.seed(42)
    start_points = [best_c.copy()]
    # Add some random perturbations
    for _ in range(15):
        c0 = best_c + np.random.randn(6) * 0.15
        c0 = np.clip(c0, -10.0, 10.0)
        start_points.append(c0)
    
    # Use Nelder-Mead
    for i, c0 in enumerate(start_points):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(obj, c0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_obj:
                best_c = np.clip(res.x, -10.0, 10.0)
                best_obj = res.fun
        except Exception:
            pass
    
    # Refine with smaller steps if time permits
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline:
                break
            try:
                res = minimize(obj, best_c, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_obj:
                    best_c = np.clip(res.x, -10.0, 10.0)
                    best_obj = res.fun
            except Exception:
                pass
    
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
