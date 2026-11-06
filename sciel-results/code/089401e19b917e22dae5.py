"""Cosmology: compact dark-energy growth approximation — differential-evolution search
against pre-computed high-fidelity reference tables."""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
_DEADLINE = None
_START_TIME = time.time()

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = _START_TIME + 82.0  # 8 s safety margin below 90 s
    return _DEADLINE

def _remaining_time():
    return _get_deadline() - time.time()

def _solve_growth_ode(Om0, wa, a_grid):
    """Solve the GR second-order ODE for f(a) and D(a) on a log-spaced grid."""
    w0 = -1.0  # standard CPL convention; w(a) = w0 + wa*(1-a)
    
    def _om_a(a):
        # Om(a) = Om0 * a^{-3} / (Om0 * a^{-3} + (1-Om0) * exp(-3*(1+w0)*((1-a)/(1+w0)) if w0 != -1 else (1-a)) + ... )
        # For CPL: E(a)^2 = Om_m * a^{-3} + Om_r * a^{-4} + Om_de * a^{-3(1+w0+wa)}
        # We use matter + dark energy only (radiation negligible for growth at a > 0.1)
        a = np.atleast_1d(a)
        Om_m = Om0
        Om_de = 1.0 - Om0
        if abs(w0 + wa) < 1e-12:
            de_term = a ** (-3.0 * (1.0 + w0))
        else:
            de_term = a ** (-3.0 * (1.0 + w0 + wa))
        E2 = Om_m * a**(-3.0) + Om_de * de_term
        return Om_m * a**(-3.0) / E2
    
    def _f_ode(x, y):
        # y = [f, f']
        a = np.exp(x)
        om = _om_a(a)
        q = 1.0 - om
        # f' = (1/2)(1 + 9/2*Om - 9/2*Om^gamma_ref) * f  ... but we use the full GR expression
        # Actually for the reference we solve the exact ODE:
        # f' = (1/2)(1 + 9/2 Om - 9/2 Om^gamma) f  is the approximation.
        # The exact GR ODE is: f'' + (1.5 + 0.5*w(a)) f' + (3/4) Om(a) f = 0
        # in terms of x = ln a, with f = d ln D / d ln a
        # Actually: a d^2 D/da^2 + (3/2 + w(a)/2) a dD/da - (3/2) Om(a) D = 0
        # Let f = d ln D / d ln a = a D'/D
        # Then f' + (3/2 + w(a)/2) f + (3/2) Om(a) * (something) ... 
        # Standard form: f' = (1/2)(1 + 9/2 Om - 9/2 Om^gamma) f is the approximation.
        # The exact linearized perturbation equation for matter density contrast delta:
        # delta'' + (1 + q) delta' + (3/2) Om delta = 0 where primes are d/d ln a
        # f = delta' / delta (for matter), so:
        # f' + (1+q) f + (3/2) Om = 0  ... no, that's not right either.
        # Let me use the standard form:
        # In terms of x = ln(a):
        # y = f, y' = df/dx
        # The exact equation for the growth rate f = d ln D / d ln a:
        # f' + (3/2)(1 + w(a)) f + (3/2) Om(a) (1 - f) = 0  ... this varies by reference.
        # 
        # Let me use the well-known exact ODE:
        # a^2 D'' + 3H(a) a D' - (3/2) Om(a) H_0^2 D = 0  (matter perturbation)
        # In terms of x = ln a: D_xx + (3/2)(1 + w) D_x - (3/2) Om D = 0
        # Let f = D_x / D, then f' = D_xx/D - (D_x/D)^2 = -f + (D_xx/D)
        # D_xx/D = -(3/2)(1+w) f + (3/2) Om
        # So f' = -(3/2)(1+w) f + (3/2) Om - f = -(5/2)(1+w) f + (3/2) Om
        # Wait, let me re-derive:
        # D_xx + (3/2)(1+w) D_x - (3/2) Om D = 0
        # D_xx = -(3/2)(1+w) D_x + (3/2) Om D
        # f = D_x/D => D_x = f D
        # D_xx = f' D + f D_x = f' D + f^2 D
        # So: f' D + f^2 D = -(3/2)(1+w) f D + (3/2) Om D
        # f' + f^2 = -(3/2)(1+w) f + (3/2) Om
        # f' = -(3/2)(1+w) f - f^2 + (3/2) Om
        
        w_a = w0 + wa * (1.0 - a)
        return [-(3.0/2.0)*(1.0 + w_a)*y[0] - y[0]**2 + (3.0/2.0)*om]
    
    n = len(a_grid)
    x_grid = np.log(a_grid)
    x0 = x_grid[0]
    x1 = x_grid[-1]
    
    # Initial conditions: at very early times f ≈ 1, D(1) = 1
    # We integrate backward from a=1 where f(1) ≈ 0.55 (rough) and D(1)=1
    # Better: integrate from x0 to x1 with f(x0) = 1.0 (early time limit)
    # and D(x0) small, then normalize D(1)=1.
    
    # Use shooting: integrate forward from x0 with f(x0)=1.0, compute D, normalize.
    y0 = [1.0, 0.0]  # f at x0; we only need f, D is obtained by integrating f
    sol = solve_ivp(_f_ode, [x0, x1], y0, t_eval=x_grid, rtol=1e-10, atol=1e-12, max_step=0.01)
    if not sol.success:
        # Fallback: return approximate values
        f_ref = Om0 * a_grid**(-3) / (Om0 * a_grid**(-3) + (1-Om0) * a_grid**(-3*(1+w0+wa)))
        f_ref = 0.55 * np.ones_like(f_ref)  # constant gamma fallback
        D_ref = np.exp(np.cumsum(-f_ref * np.diff(x_grid, prepend=x_grid[0])))
        D_ref = D_ref / D_ref[-1]
        return f_ref, D_ref
    
    f_vals = sol.y[0]
    # Integrate f to get ln D: ln D(x) = integral of f dx from x0 to x
    lnD = np.cumsum(f_vals * np.diff(x_grid, prepend=x_grid[0]))
    D_vals = np.exp(lnD)
    # Normalize so D(a=1) = 1
    D_vals = D_vals / D_vals[-1]
    
    return f_vals, D_vals

def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _get_deadline()
    
    # Pre-compute reference data for a set of validation cosmologies
    # Use a range of Om0 and wa values to span the validation space
    Om0_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    wa_vals = [-0.5, -0.3, -0.2, 0.0, 0.1, 0.2, 0.3]
    w0 = -1.0
    
    a_grid = np.logspace(-1.5, 0.0, 60)  # a from ~0.032 to 1.0
    x_grid = np.log(a_grid)
    
    n_models = len(Om0_vals) * len(wa_vals)
    n_a = len(a_grid)
    
    Om_a_arr = np.zeros((n_models, n_a))
    Q_arr = np.zeros((n_models, n_a))
    U_arr = np.zeros((n_models, n_a))
    F_ref = np.zeros((n_models, n_a))
    D_ref = np.zeros((n_models, n_a))
    
    idx = 0
    for om0 in Om0_vals:
        for wa in wa_vals:
            f_ref, d_ref = _solve_growth_ode(om0, wa, a_grid)
            F_ref[idx, :] = f_ref
            D_ref[idx, :] = d_ref
            Om_a_arr[idx, :] = _om_a_internal(om0, wa, a_grid)
            Q_arr[idx, :] = 1.0 - Om_a_arr[idx, :]
            U_arr[idx, :] = w0 + wa * (1.0 - a_grid) + 1.0  # u = 1 + w0 + wa*(1-a)
            idx += 1
    
    # Quadrature weights for trapezoidal rule on x_grid
    w_q = np.zeros(n_a)
    if n_a > 1:
        w_q[0] = (x_grid[1] - x_grid[0]) / 2.0
        w_q[-1] = (x_grid[-1] - x_grid[-2]) / 2.0
        if n_a > 2:
            w_q[1:-1] = (x_grid[2:] - x_grid[:-2]) / 2.0
    
    # Pre-compute factorized arrays for fast evaluation
    a_grid_2d = np.ones((n_models, 1)) * a_grid[np.newaxis, :]
    wa_2d = np.zeros((n_models, 1))
    for i, (om0, wa) in enumerate(zip(
        [Om0_vals[i // len(wa_vals)] for i in range(n_models)],
        [wa_vals[i % len(wa_vals)] for i in range(n_models)]
    )):
        wa_2d[i, 0] = wa

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        Gamma = (c0 + c1*Q_arr + c2*Q_arr**2 + c3*U_arr + c4*wa_2d*a_grid_2d + c5*U_arr*Q_arr)
        
        # Constraint: Gamma in [0.1, 1.2]
        gamma_violation = np.maximum(0.1 - Gamma, 0.0) + np.maximum(Gamma - 1.2, 0.0)
        if np.max(gamma_violation) > 1e-8:
            return 1e4 + np.max(gamma_violation)
        
        F_approx = Om_a_arr ** Gamma
        
        # Relative error in f
        rel_err_f = np.abs(F_approx - F_ref) / np.maximum(np.abs(F_ref), 1e-12)
        
        # D approximation: D(a)/D(1) = exp(-integral from ln(a) to 0 of f dx)
        # = exp(integral from 0 to ln(a) of f dx) ... wait, let's be careful.
        # D(a)/D(1) = exp(-∫_{ln a}^{0} f(x) dx) = exp(∫_0^{ln a} f(x) dx) for a < 1
        # Actually: D(a)/D(1) = exp(∫_1^a f(a')/a' da') = exp(∫_{0}^{ln a} f(x') dx')
        # Since a < 1, ln a < 0, so this integral is from 0 to ln a (negative direction)
        # = -exp(∫_{ln a}^{0} f(x) dx)
        
        # Using cumulative integration from a_grid[0] to a_grid[i]:
        # ln D(i) = ∫_{x0}^{xi} f(x) dx
        # D(i)/D(1) = exp(ln D(i) - ln D(1)) = exp(∫_{x0}^{xi} f dx - ∫_{x0}^{x1} f dx)
        # = exp(-∫_{xi}^{x1} f dx)
        
        integrand = F_approx * w_q[np.newaxis, :]
        cum_int = np.cumsum(integrand, axis=1)
        total_int = cum_int[:, -1:]  # ∫ from x0 to x1
        ln_D_ratio = -(total_int - cum_int)  # -∫ from xi to x1
        D_approx = np.exp(ln_D_ratio)
        
        rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-12)
        
        max_err = np.max(np.maximum(rel_err_f, rel_err_D))
        return float(max_err)
    
    # Differential evolution with time guard
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = objective(best_coeffs)
    
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    
    def _check_time():
        return time.time() > deadline - 3.0
    
    # Use a callback to check time
    from scipy.optimize import _minimize
    # We'll just run DE with a moderate popsize and maxiter, relying on the 82s budget
    
    try:
        if _remaining_time() > 20:
            res = differential_evolution(
                objective,
                bounds,
                maxiter=80,
                popsize=8,
                tol=1e-5,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            if res.x is not None:
                cand = np.clip(res.x, -10.0, 10.0)
                s = objective(cand)
                if s < best_score:
                    best_coeffs = cand
                    best_score = s
    except Exception:
        pass
    
    # Local polish with L-BFGS-B if time remains
    if _remaining_time() > 8:
        try:
            from scipy.optimize import minimize
            res2 = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.x is not None:
                cand = np.clip(res2.x, -10.0, 10.0)
                s = objective(cand)
                if s < best_score:
                    best_coeffs = cand
                    best_score = s
        except Exception:
            pass
    
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    return list(best_coeffs)

def _om_a_internal(om0, wa, a_grid):
    """Compute Om(a) for CPL dark energy (matter + DE only)."""
    w0 = -1.0
    a = np.atleast_1d(a_grid)
    Om_m = om0
    Om_de = 1.0 - om0
    de_term = a ** (-3.0 * (1.0 + w0 + wa))
    E2 = Om_m * a**(-3.0) + Om_de * de_term
    return Om_m * a**(-3.0) / E2

# EVOLVE-BLOCK-END
