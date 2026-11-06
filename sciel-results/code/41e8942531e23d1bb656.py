"""Cosmology: compact dark-energy growth approximation.
Vectorized DE + local refinement against pre-computed references.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START

def _check_deadline():
    if _elapsed() > _DEADLINE:
        raise TimeoutError

def _om_matter_vec(a, om0):
    return om0 / (a ** 3)

def _gamma_of_a_vec(a, om0, w0, wa, c):
    """Compute gamma from the six-coefficient ansatz (vectorized).
    Contract: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q=1-Om(a), u=1+w0+wa*(1-a)
    """
    c0, c1, c2, c3, c4, c5 = c
    om_a = _om_matter_vec(a, om0)
    q = 1.0 - om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a_vec(a, om0, w0, wa, c):
    om_a = _om_matter_vec(a, om0)
    gamma = _gamma_of_a_vec(a, om0, w0, wa, c)
    gamma = np.clip(gamma, 0.1, 1.2)
    return om_a ** gamma

def _solve_growth_ode(om0, w0, wa, a_min=0.02):
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
                    method='DOP853', rtol=1e-8, atol=1e-10,
                    dense_output=True, max_step=0.05)
    
    if not sol.success:
        raise RuntimeError("ODE solve failed")
    
    return sol

def _get_reference(om0, w0, wa, a_grid):
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
    wa_vals = [0.0, 0.3, 0.5, -0.3, 0.8]
    
    cosmologies = []
    for om0 in om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                if 0.1 < om0 < 0.5:
                    cosmologies.append((om0, w0, wa))
    
    # Keep it manageable: ~24 cosmologies
    np.random.seed(42)
    if len(cosmologies) > 24:
        idx = np.random.choice(len(cosmologies), 24, replace=False)
        cosmologies = [cosmologies[i] for i in sorted(idx)]
    
    # Ensure LambdaCDM is included
    if (0.27, -1.0, 0.0) not in cosmologies:
        cosmologies.append((0.27, -1.0, 0.0))
    
    a_grid = np.logspace(np.log10(0.02), 0.0, 100)
    a_grid = np.unique(a_grid)
    
    return cosmologies, a_grid

def _compute_D_vec(a, om0, w0, wa, c, x_grid, f_model_grid):
    """Compute D(a)/D(1) using pre-computed f_model on x_grid via cumulative integration."""
    n = len(x_grid)
    dx = np.diff(x_grid)
    areas = 0.5 * (f_model_grid[:-1] + f_model_grid[1:]) * dx
    cum_int = np.zeros(n)
    cum_int[:-1] = np.cumsum(areas[::-1])[::-1]
    return np.exp(-cum_int)

def _evaluate_score_vec(c, cosmologies, a_grid, f_refs, D_refs, x_grid):
    """Score: 100 * max relative error."""
    max_err = 0.0
    n_a = len(a_grid)
    
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        err_f = np.abs(f_model - f_ref) / f_ref
        max_err = max(max_err, np.max(err_f))
        
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        D_model = _compute_D_vec(a_grid, om0, w0, wa, c, x_grid, f_on_x)
        
        err_D = np.abs(D_model - D_ref) / D_ref
        max_err = max(max_err, np.max(err_D))
    
    return 100.0 * max_err

def _objective_smooth(c, cosmologies, a_grid, f_refs, D_refs, x_grid, beta=30.0):
    """Smooth surrogate using log-sum-exp for the max."""
    n_cosmo = len(cosmologies)
    n_a = len(a_grid)
    total_errs = n_cosmo * 2 * n_a
    errs = np.empty(total_errs)
    
    idx = 0
    for i, (om0, w0, wa) in enumerate(cosmologies):
        f_ref = f_refs[i]
        D_ref = D_refs[i]
        
        f_model = _f_of_a_vec(a_grid, om0, w0, wa, c)
        err_f = np.abs(f_model - f_ref) / f_ref
        errs[idx:idx+n_a] = err_f
        idx += n_a
        
        a_x = np.exp(x_grid)
        f_on_x = _f_of_a_vec(a_x, om0, w0, wa, c)
        D_model = _compute_D_vec(a_grid, om0, w0, wa, c, x_grid, f_on_x)
        err_D = np.abs(D_model - D_ref) / D_ref
        errs[idx:idx+n_a] = err_D
        idx += n_a
    
    if idx == 0:
        return 1e6
    
    errs = errs[:idx]
    m = np.max(errs)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp(errs - m))) / beta
    return 100.0 * lse

def _check_gamma_constraint(c, cosmologies, a_check):
    """Check if gamma stays in [0.1, 1.2] for all cosmologies."""
    for om0, w0, wa in cosmologies:
        g = _gamma_of_a_vec(a_check, om0, w0, wa, c)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return False
    return True

def construct():
    """Construct the six coefficients using differential evolution + local polish."""
    _check_deadline()
    
    cosmologies, a_grid = _build_validation_set()
    n_cosmo = len(cosmologies)
    x_grid = np.log(a_grid)
    
    # Pre-compute references
    f_refs = []
    D_refs = []
    for i, (om0, w0, wa) in enumerate(cosmologies):
        _check_deadline()
        try:
            f_ref, D_ref = _get_reference(om0, w0, wa, a_grid)
            f_refs.append(f_ref)
            D_refs.append(D_ref)
        except Exception:
            pass
    
    # Remove any failed cosmologies
    valid_cosmos = []
    valid_f_refs = []
    valid_D_refs = []
    for i, (om0, w0, wa) in enumerate(cosmologies):
        if i < len(f_refs):
            valid_cosmos.append((om0, w0, wa))
            valid_f_refs.append(f_refs[i])
            valid_D_refs.append(D_refs[i])
    
    cosmologies = valid_cosmos
    f_refs = valid_f_refs
    D_refs = valid_D_refs
    
    _check_deadline()
    
    bounds = [(-10, 10)] * 6
    c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    score_init = _evaluate_score_vec(c_best, cosmologies, a_grid, f_refs, D_refs, x_grid)
    
    # Phase 1: Differential Evolution with smooth objective
    if _elapsed() < _DEADLINE - 20:
        try:
            de_result = differential_evolution(
                _objective_smooth,
                bounds,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid, 30.0),
                maxiter=150,
                popsize=25,
                tol=1e-6,
                seed=42,
                polish=False,
                updating='immediate'
            )
            if de_result.success and _elapsed() < _DEADLINE - 10:
                c_de = de_result.x
                # Check gamma constraint
                a_check = np.logspace(np.log10(0.02), 0.0, 50)
                if _check_gamma_constraint(c_de, cosmologies, a_check):
                    score_de = _evaluate_score_vec(c_de, cosmologies, a_grid, f_refs, D_refs, x_grid)
                    if score_de < score_init:
                        c_best = c_de
                        score_init = score_de
        except Exception:
            pass
    
    _check_deadline()
    
    # Phase 2: Local refinement with Nelder-Mead (derivative-free, good for nonsmooth)
    if _elapsed() < _DEADLINE - 8:
        try:
            res = minimize(
                _evaluate_score_vec,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            c_nm = res.x
            a_check = np.logspace(np.log10(0.02), 0.0, 50)
            if _check_gamma_constraint(c_nm, cosmologies, a_check):
                score_nm = _evaluate_score_vec(c_nm, cosmologies, a_grid, f_refs, D_refs, x_grid)
                if score_nm < score_init:
                    c_best = c_nm
                    score_init = score_nm
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B polish with smooth surrogate
    if _elapsed() < _DEADLINE - 5:
        try:
            res2 = minimize(
                _objective_smooth,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid, 100.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_lbfgs = res2.x
            a_check = np.logspace(np.log10(0.02), 0.0, 50)
            if _check_gamma_constraint(c_lbfgs, cosmologies, a_check):
                score_lbfgs = _evaluate_score_vec(c_lbfgs, cosmologies, a_grid, f_refs, D_refs, x_grid)
                if score_lbfgs < score_init:
                    c_best = c_lbfgs
                    score_init = score_lbfgs
        except Exception:
            pass
    
    # Phase 4: Fine local search with small perturbations
    if _elapsed() < _DEADLINE - 3:
        directions = [
            np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.1]),
            np.array([1.0, -0.4, 0.3, -0.2, 0.15, -0.1]),
            np.array([0.8, 0.3, 0.2, 0.4, -0.2, 0.3]),
            np.array([1.0, 0.2, -0.1, -0.3, 0.25, -0.2]),
            np.array([0.9, -0.2, 0.15, 0.1, 0.1, 0.2]),
            np.array([1.0, 0.1, 0.1, -0.1, -0.15, 0.15]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ]
        
        a_check = np.logspace(np.log10(0.02), 0.0, 50)
        
        for direction in directions:
            if _elapsed() > _DEADLINE - 3:
                break
            for step in [0.05, 0.02, 0.01, 0.005]:
                if _elapsed() > _DEADLINE - 3:
                    break
                for sign in [1.0, -1.0]:
                    c_try = c_best + sign * step * direction
                    c_try = np.clip(c_try, -10, 10)
                    
                    if not _check_gamma_constraint(c_try, cosmologies, a_check):
                        continue
                    
                    score_try = _evaluate_score_vec(c_try, cosmologies, a_grid, f_refs, D_refs, x_grid)
                    if score_try < score_init:
                        c_best = c_try
                        score_init = score_try
                        break
    
    # Phase 5: Final tight polish
    if _elapsed() < _DEADLINE - 2:
        try:
            res3 = minimize(
                _evaluate_score_vec,
                c_best,
                args=(cosmologies, a_grid, f_refs, D_refs, x_grid),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            c_final = res3.x
            a_check = np.logspace(np.log10(0.02), 0.0, 50)
            if _check_gamma_constraint(c_final, cosmologies, a_check):
                score_final = _evaluate_score_vec(c_final, cosmologies, a_grid, f_refs, D_refs, x_grid)
                if score_final < score_init:
                    c_best = c_final
                    score_init = score_final
        except Exception:
            pass
    
    c_best = np.clip(c_best, -10, 10)
    if not np.all(np.isfinite(c_best)):
        c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Final gamma constraint verification
    a_check = np.logspace(np.log10(0.02), 0.0, 100)
    if not _check_gamma_constraint(c_best, cosmologies, a_check):
        for scale in [0.8, 0.6, 0.4, 0.2, 0.0]:
            c_try = np.array([c_best[0], scale*c_best[1], scale*c_best[2],
                              scale*c_best[3], scale*c_best[4], scale*c_best[5]])
            if _check_gamma_constraint(c_try, cosmologies, a_check):
                c_best = c_try
                break
        else:
            c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return [float(x) for x in c_best]
