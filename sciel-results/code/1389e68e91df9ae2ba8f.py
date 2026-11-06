"""Cosmology: compact dark-energy growth approximation.
Vectorized differential evolution + local refinement against pre-computed references.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = 82.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START

def _check_deadline():
    if _elapsed() > _DEADLINE:
        raise TimeoutError()

def _om_matter_vec(a, om0):
    """Matter density parameter at scale factor a (vectorized)."""
    return om0 / (a ** 3)

def _gamma_of_a_vec(a, om0, w0, wa, c):
    """Compute gamma from the six-coefficient ansatz (vectorized)."""
    c0, c1, c2, c3, c4, c5 = c
    om_a = _om_matter_vec(a, om0)
    q = 1.0 - om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a_vec(a, om0, w0, wa, c):
    """Compute f = Om(a)^gamma (vectorized)."""
    om_a = _om_matter_vec(a, om0)
    gamma = _gamma_of_a_vec(a, om0, w0, wa, c)
    # Clip gamma to avoid numerical issues
    gamma = np.clip(gamma, 0.1, 1.2)
    return om_a ** gamma

def _solve_growth_ode(om0, w0, wa, a_min=0.01):
    """Solve the GR growth ODE to get reference f and D(a)/D(1)."""
    def ode(x, y):
        a = np.exp(x)
        E2 = om0 / (a**3) + (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        om_m = om0 / (a**3)
        om_de = (1.0 - om0) * np.exp(3.0 * (1.0 + w0 + wa) * (1.0 - a))
        w = w0 + wa * (1.0 - a)
        E = np.sqrt(E2)
        A = 3.0 / E * (1.0 + 0.5 * om_de * (1.0 + 3.0 * w) / om_m) + 2.0
        B = -3.0 * om_m / (2.0 * E2)
        return [y[1], -A * y[1] + B * y[0]]

    x_min = np.log(a_min)
    x_max = np.log(1.0)
    a_init = a_min
    D_init = a_init ** (6.0 / 5.0)
    Dp_init = (6.0 / 5.0) * a_init ** (1.0 / 5.0)
    
    sol = solve_ivp(ode, [x_min, x_max], [D_init, Dp_init], 
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.01)
    
    if not sol.success:
        raise RuntimeError("ODE solve failed")
    
    return sol

def _get_reference(om0, w0, wa, a_grid):
    """Get reference f and D(a)/D(1) on a_grid for a given cosmology."""
    sol = _solve_growth_ode(om0, w0, wa)
    x_grid = np.log(a_grid)
    y = sol.sol(x_grid)
    D_raw = y[0]
    Dp_raw = y[1]
    D_ref = D_raw / D_raw[-1]
    f_ref = Dp_raw / D_raw
    
    D_ref = np.maximum(D_ref, 1e-10)
    f_ref = np.maximum(f_ref, 1e-10)
    
    return f_ref, D_ref

def _build_validation_set():
    """Build a diverse set of validation cosmologies and a-grid."""
    om0_vals = [0.2, 0.25, 0.27, 0.3, 0.35, 0.4]
    w0_vals = [-1.0, -0.9, -0.8, -1.1]
    wa_vals = [0.0, 0.3, 0.5, -0.3, -0.5, 1.0]
    
    cosmologies = []
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                if 0 < om0 < 1:
                    cosmologies.append((om0, w0, wa))
    
    if len(cosmologies) > 30:
        np.random.seed(42)
        idx = np.random.choice(len(cosmologies), 30, replace=False)
        cosmologies = [cosmologies[i] for i in idx]
    
    cosmologies.append((0.27, -1.0, 0.0))
    cosmologies.append((0.3, -0.9, 0.3))
    cosmologies.append((0.25, -1.1, 0.5))
    
    a_grid = np.logspace(np.log10(0.05), 0.0, 50)
    a_grid = np.unique(a_grid)
    
    return cosmologies, a_grid

def _compute_D_vec(a, om0, w0, wa, c, x_grid, f_model_grid):
    """Compute D(a)/D(1) using pre-computed f_model on x_grid via cumulative integration."""
    # f_model_grid is f(exp(x_grid)) for all x in x_grid
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # We need integral from x to 0 where x = ln(a)
    # Use cumulative sum from the right
    dx = x_grid[1] - x_grid[0]
    # integral from x[i] to x[-1] (which is 0) of f dx
    # Cumulative trapezoidal from right
    cum_int = np.zeros_like(f_model_grid)
    for i in range(len(f_model_grid) - 2, -1, -1):
        cum_int[i] = cum_int[i + 1] + 0.5 * (f_model_grid[i] + f_model_grid[i + 1]) * dx
    # D = exp(-cum_int)
    return np.exp(-cum_int)

def _evaluate_score_vec(c, cosmologies, a_grid, f_refs, D_refs, x_grid):
    """Vectorized score evaluation: 100 * max relative error."""
    max_err = 0.0
    n_a = len(a_grid)
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        # f relative error
        err_f = np.abs(f_model - f_ref) / f_ref
        max_err = max(max_err, np.max(err_f))
        
        # D model: compute f on x_grid, then integrate
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        D_model = _compute_D_vec(a_grid, om0, w0, wa, c, x_grid, f_on_x)
        
        err_D = np.abs(D_model - D_ref) / D_ref
        max_err = max(max_err, np.max(err_D))
    
    return 100.0 * max_err

def _objective_smooth(c, cosmologies, a_grid, f_refs, D_refs, x_grid, beta=20.0):
    """Smooth surrogate using log-sum-exp for the max."""
    errs = []
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        err_f = np.abs(f_model - f_ref) / f_ref
        errs.extend(err_f.tolist())
        
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        D_model = _compute_D_vec(a_grid, om0, w0, wa, c, x_grid, f_on_x)
        err_D = np.abs(D_model - D_ref) / D_ref
        errs.extend(err_D.tolist())
    
    if not errs:
        return 1e6
    
    errs = np.array(errs)
    m = np.max(errs)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp((errs - m) * beta)) / beta)
    return 100.0 * lse

def construct():
    """Construct the six coefficients using differential evolution + local polish."""
    _check_deadline()
    
    cosmologies, a_grid = _build_validation_set()
    n_cosmo = len(cosmologies)
    
    # x_grid for integration: same as log(a_grid) but we may use finer grid
    x_grid = np.log(a_grid)
    
    f_refs = []
    D_refs = []
    for i, (om0, w0, wa) in enumerate(cosmologies):
        _check_deadline()
        f_ref, D_ref = _get_reference(om0, w0, wa, a_grid)
        f_refs.append(f_ref)
        D_refs.append(D_ref)
    
    _check_deadline()
    
    bounds = [(-10, 10)] * 6
    c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    score_init = _evaluate_score_vec(c_best, cosmologies, a_grid, f_refs, D_refs, x_grid)
    
    # Phase 1: Differential Evolution with smooth objective
    if _elapsed() < _DEADLINE - 30:
        try:
            de_result = differential_evolution(
                _objective_smooth,
                bounds,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid, 20.0),
                maxiter=40,
                popsize=12,
                tol=1e-4,
                seed=42,
                polish=False,
                updating='immediate'
            )
            if de_result.success and _elapsed() < _DEADLINE - 10:
                c_de = de_result.x
                score_de = _evaluate_score_vec(c_de, cosmologies, a_grid, f_refs, D_refs, x_grid)
                if score_de < score_init:
                    c_best = c_de
                    score_init = score_de
        except Exception:
            pass
    
    _check_deadline()
    
    # Phase 2: Local refinement with true objective
    if _elapsed() < _DEADLINE - 5:
        try:
            res = minimize(
                _evaluate_score_vec,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success:
                c_polished = res.x
                score_polished = _evaluate_score_vec(c_polished, cosmologies, a_grid, f_refs, D_refs, x_grid)
                if score_polished < score_init:
                    c_best = c_polished
                    score_init = score_polished
        except Exception:
            pass
    
    _check_deadline()
    
    # Phase 3: Tighter polish
    if _elapsed() < _DEADLINE - 2:
        try:
            res2 = minimize(
                _evaluate_score_vec,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.success:
                c_final = res2.x
                score_final = _evaluate_score_vec(c_final, cosmologies, a_grid, f_refs, D_refs, x_grid)
                if score_final < score_init:
                    c_best = c_final
                    score_init = score_final
        except Exception:
            pass
    
    c_best = np.clip(c_best, -10, 10)
    if not np.all(np.isfinite(c_best)):
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Verify gamma constraint
    gamma_ok = True
    for om0, w0, wa in cosmologies:
        g = _gamma_of_a_vec(a_grid, om0, w0, wa, c_best)
        if np.any(g < 0.1) or np.any(g > 1.2):
            gamma_ok = False
            break
    
    if not gamma_ok:
        for scale in [0.5, 0.3, 0.1, 0.0]:
            c_try = np.array([c_best[0], scale*c_best[1], scale*c_best[2],
                              scale*c_best[3], scale*c_best[4], scale*c_best[5]])
            gamma_ok = True
            for om0, w0, wa in cosmologies:
                g = _gamma_of_a_vec(a_grid, om0, w0, wa, c_try)
                if np.any(g < 0.1) or np.any(g > 1.2):
                    gamma_ok = False
                    break
            if gamma_ok:
                c_best = c_try
                break
    
    if not gamma_ok:
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return [float(x) for x in c_best]
