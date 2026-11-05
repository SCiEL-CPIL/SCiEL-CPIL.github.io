"""Improved growth-index ansatz with first-order f-ODE reference and structured initialization."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 52.0  # 52s safety margin before 60s evaluator limit

def _check_deadline():
    return time.time() > _DEADLINE

def _omega_m_of_a(a, Om0, w0, wa):
    """Compute Omega_m(a) for flat CPL dark-energy cosmology."""
    a = np.asarray(a, dtype=float)
    # For CPL w(a) = w0 + wa*(1-a):
    # rho_DE(a)/rho_DE(1) = exp(-3*(1+w0+wa)*ln(a) + 3*wa*(a-1))
    # rho_m(a)/rho_m(1) = a^(-3)
    # Omega_m(a) = Om0 * a^(-3) / (Om0 * a^(-3) + (1-Om0) * exp(...))
    
    if np.isscalar(a) or a.ndim == 0:
        a = np.array([a])
        scalar_input = True
    else:
        scalar_input = False
    
    Om_m = Om0 / a**3
    
    # DE density ratio
    if abs(wa) < 1e-12:
        # Constant w
        if abs(1.0 + w0) < 1e-12:
            rho_de_ratio = np.ones_like(a)
        else:
            rho_de_ratio = a**(-3.0*(1.0 + w0))
    else:
        exponent = -3.0 * np.log(a) * (1.0 + w0 + wa) + 3.0 * wa * (a - 1.0)
        rho_de_ratio = np.exp(exponent)
    
    denom = Om_m + (1.0 - Om0) * rho_de_ratio
    Om_a = Om_m / denom
    
    if scalar_input:
        return float(Om_a[0])
    return Om_a

def _gamma_from_coeffs(a, Om0, w0, wa, coeffs):
    """Compute gamma from the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_from_gamma(a, Om0, w0, wa, coeffs):
    """Compute f(a) = Om(a)^gamma."""
    Om_a = _omega_m_of_a(a, Om0, w0, wa)
    gamma = _gamma_from_coeffs(a, Om0, w0, wa, coeffs)
    # Clip gamma to avoid numerical issues
    gamma_clipped = np.clip(gamma, 0.05, 1.5)
    f = Om_a**gamma_clipped
    return f

def _growth_ode_reference(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=200):
    """
    Solve the first-order ODE for f(a) = d ln D / d ln a to get reference f and D.
    
    The first-order ODE is:
    df/da = -(1+gamma) * f + (3/2) * Om(a)/a * (1 - f)
    where gamma = d ln f / d ln Om(a)
    
    Actually, let's use the standard second-order ODE for D(a) but solve it
    from early times where D ~ a, which is more stable.
    
    ODE: a^2 D'' + [3a + (a/2) d ln E^2 / da] D' - (3/2) Om(a) D = 0
    
    We integrate from a_min to a_max with ICs D(a_min) ~ a_min, D'(a_min) ~ 1.
    Then normalize D(1) = 1.
    """
    a_grid = np.linspace(a_min, a_max, n_points)
    
    def hubble2(a):
        """E^2(a) = (H/H0)^2"""
        Om_m = Om0 / a**3
        if abs(wa) < 1e-12:
            if abs(1.0 + w0) < 1e-12:
                rho_de = (1.0 - Om0)
            else:
                rho_de = (1.0 - Om0) * a**(-3.0*(1.0 + w0))
        else:
            exponent = -3.0 * np.log(a) * (1.0 + w0 + wa) + 3.0 * wa * (a - 1.0)
            rho_de = (1.0 - Om0) * np.exp(exponent)
        return Om_m + rho_de
    
    def dH2_da(a):
        """d(H^2)/da"""
        Om_m = Om0 / a**3
        dOm_m_da = -3.0 * Om0 / a**4
        
        if abs(wa) < 1e-12:
            if abs(1.0 + w0) < 1e-12:
                drho_de_da = 0.0
            else:
                rho_de = (1.0 - Om0) * a**(-3.0*(1.0 + w0))
                drho_de_da = -3.0*(1.0 + w0) * rho_de / a
        else:
            exponent = -3.0 * np.log(a) * (1.0 + w0 + wa) + 3.0 * wa * (a - 1.0)
            rho_de = (1.0 - Om0) * np.exp(exponent)
            # d(exponent)/da = -3*(1+w0+wa)/a + 3*wa
            dexp_da = -3.0*(1.0 + w0 + wa)/a + 3.0*wa
            drho_de_da = rho_de * dexp_da
        
        return dOm_m_da + drho_de_da
    
    def ode_funcs(a, y):
        D, Dp = y
        H2 = hubble2(a)
        dH2 = dH2_da(a)
        Hp_over_H = 0.5 * dH2 / H2
        Om_a = Om0 / a**3
        Dpp = -[3.0/a + Hp_over_H] * Dp + 1.5 * Om_a / a * D
        return [Dp, Dpp]
    
    # Initial conditions at a_min: D ~ a, D' ~ 1
    y0 = [a_min, 1.0]
    
    try:
        sol = solve_ivp(ode_funcs, [a_min, a_max], y0, t_eval=a_grid,
                       method='RK45', rtol=1e-10, atol=1e-12, max_step=0.005)
        if not sol.success:
            return None, None, None
    except Exception:
        return None, None, None
    
    a_vals = sol.t
    D_vals = sol.y[0]
    Dp_vals = sol.y[1]
    
    # Normalize D(1) = 1
    # Find index closest to a=1
    idx_1 = np.argmin(np.abs(a_vals - 1.0))
    D_norm = D_vals / D_vals[idx_1]
    Dp_norm = Dp_vals / D_vals[idx_1]
    
    # f(a) = a * D'(a) / D(a)
    f_vals = a_vals * Dp_norm / D_norm
    
    # Filter out non-finite values
    mask = np.isfinite(f_vals) & np.isfinite(D_norm) & (D_norm > 0) & (f_vals > 0)
    
    if np.sum(mask) < 50:
        return None, None, None
    
    return a_vals[mask], f_vals[mask], D_norm[mask]

def _compute_D_from_f(f_vals, a_vals):
    """Compute D(a)/D(1) = exp(-∫_a^1 f(a')/a' da') using trapezoidal rule."""
    n = len(a_vals)
    if n < 2:
        return np.ones_like(a_vals)
    
    integrand = f_vals / a_vals
    da = np.diff(a_vals)
    trap = 0.5 * (integrand[:-1] + integrand[1:]) * da
    
    # Cumulative integral from right (from a_max down to a_min)
    cum_int = np.zeros(n)
    for i in range(n - 2, -1, -1):
        cum_int[i] = cum_int[i + 1] + trap[i]
    
    # cum_int[i] = ∫_{a_i}^{a_max} f(a')/a' da'
    # We want D(a)/D(1) = exp(-∫_a^1 f(a')/a' da')
    # If a_max = 1, then cum_int[i] = ∫_{a_i}^1 f(a')/a' da'
    D_norm = np.exp(-cum_int)
    
    # Normalize: D(1) should be 1. At a=1 (last point), cum_int=0, so D=1. Good.
    return D_norm

def _objective(coeffs, ref_data, a_eval_grid):
    """
    Compute max relative error of f and D across all reference cosmologies.
    ref_data: list of (a_ref, f_ref, D_ref, Om0, w0, wa)
    a_eval_grid: common evaluation grid (subset of a_ref where we evaluate)
    """
    max_err = 0.0
    
    for a_ref, f_ref, D_ref, Om0, w0, wa in ref_data:
        # Interpolate reference to evaluation grid
        mask = (a_eval_grid >= a_ref[0]) & (a_eval_grid <= a_ref[-1])
        if np.sum(mask) < 10:
            continue
        a_eval = a_eval_grid[mask]
        f_ref_eval = np.interp(a_eval, a_ref, f_ref)
        D_ref_eval = np.interp(a_eval, a_ref, D_ref)
        
        # Compute model
        f_model = _f_from_gamma(a_eval, Om0, w0, wa, coeffs)
        D_model = _compute_D_from_f(f_model, a_eval)
        
        # Check gamma bounds
        gamma = _gamma_from_coeffs(a_eval, Om0, w0, wa, coeffs)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6
        
        # Relative errors
        rel_f = np.abs(f_model - f_ref_eval) / np.abs(f_ref_eval)
        rel_D = np.abs(D_model - D_ref_eval) / np.abs(D_ref_eval)
        
        err = max(np.max(rel_f), np.max(rel_D))
        max_err = max(max_err, err)
    
    return max_err

def construct():
    """Find 6 coefficients minimizing max relative error."""
    
    # Structured initialization: c0=0.55, c3=-0.5 to cancel leading w0 term
    # gamma ≈ 0.55 + c3*(1+w0+wa(1-a))
    # For w0=-1, wa=0: gamma ≈ 0.55 + c3*0 = 0.55 (good)
    # For w0=-0.5: gamma ≈ 0.55 + c3*0.5, so c3=-0.5 gives gamma ≈ 0.55-0.25=0.30? No.
    # u = 1+w0+wa(1-a). For w0=-0.5, a=1: u=0.5. c3*u = -0.5*0.5 = -0.25.
    # gamma = 0.55 - 0.25 = 0.30. That's too low.
    # Let's think: for matter domination, gamma should be ~0.55.
    # The term c3*u should be small. Maybe c3 should be small.
    # Let's use the strategy: c0=0.55, and let the optimizer figure out the rest.
    # But initialize c3=-0.1 as a gentle correction.
    
    best_coeff = np.array([0.55, 0.0, 0.0, -0.1, 0.0, 0.0])
    best_score = np.inf
    
    # Representative validation cosmologies
    val_cosmos = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.3),   # Positive wa
        (0.30, -1.0, -0.3),  # Negative wa
        (0.30, -0.5, 0.0),   # w0 > -1
        (0.30, -1.5, 0.0),   # w0 < -1 (phantom)
        (0.25, -1.0, 0.0),   # Lower Om0
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.30, -0.5, 0.5),   # CPL
    ]
    
    # Evaluation grid
    a_eval_grid = np.linspace(0.05, 1.0, 100)
    
    # Pre-compute reference data
    ref_data = []
    for Om0, w0, wa in val_cosmos:
        if _check_deadline():
            break
        try:
            a_ref, f_ref, D_ref = _growth_ode_reference(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=200)
            if a_ref is not None:
                ref_data.append((a_ref, f_ref, D_ref, Om0, w0, wa))
        except Exception:
            continue
    
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Multiple starting points with structured initialization
    starts = [
        [0.55, 0.0, 0.0, -0.1, 0.0, 0.0],
        [0.55, 0.1, 0.0, -0.2, 0.0, 0.0],
        [0.55, -0.1, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.1, -0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, -0.1, 0.1, 0.0],
        [0.55, 0.0, 0.0, -0.1, 0.0, 0.1],
        [0.55, 0.05, 0.05, -0.15, 0.05, 0.05],
        [0.55, -0.05, 0.05, -0.05, -0.05, 0.05],
    ]
    
    for start in starts:
        if _check_deadline():
            break
        
        def obj(coeffs):
            if _check_deadline():
                return 1e6
            return _objective(coeffs, ref_data, a_eval_grid)
        
        try:
            res = minimize(obj, start, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeff = res.x.copy()
        except Exception:
            continue
    
    # Try a few more local optimizations from the best found so far
    if not _check_deadline():
        for perturb in [0.01, -0.01, 0.05, -0.05]:
            if _check_deadline():
                break
            trial = best_coeff + perturb * np.random.randn(6)
            trial = np.clip(trial, -10, 10)
            
            def obj(coeffs):
                if _check_deadline():
                    return 1e6
                return _objective(coeffs, ref_data, a_eval_grid)
            
            try:
                res = minimize(obj, trial, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeff = res.x.copy()
            except Exception:
                continue
    
    # Clip to [-10, 10]
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()
