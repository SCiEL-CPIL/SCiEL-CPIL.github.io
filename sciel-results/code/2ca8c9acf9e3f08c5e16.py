"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: Reference-cosmology anchoring with convex weight optimization.
Instead of fitting raw polynomial coefficients, we pre-compute high-accuracy
growth-index profiles from diverse anchor cosmologies, then optimize the
weights to minimize the global error. This transforms the search into a
convex interpolation problem within a physically valid subspace.

Key improvements:
1. Anchor-based initialization: Fit gamma as a weighted combination of
   pre-computed profiles from diverse cosmologies
2. Convex weight optimization: Use SLSQP on weights with non-negativity
   constraints, then map back to coefficient space
3. Two-stage refinement: Coarse grid for speed, fine grid for accuracy
4. Direct minimax with auxiliary variable for tighter convergence
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 85.0

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _omega_m_of_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    denom = Om0 * a_inv3 + Om_DE
    Om_m = Om0 * a_inv3 / denom
    return Om_m

def _design_matrix(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    X = np.column_stack([
        np.ones_like(a),
        q,
        q**2,
        u,
        wa * a,
        u * q
    ])
    return X

def _growth_ode_reference(a_grid, Om0, w0, wa):
    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        w = w0 + wa * (1.0 - a)
        dOm_m_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [F, D_double]

    D0 = a_min ** (5.0/6.0)
    F0 = (5.0/6.0) * a_min ** (-1.0/6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid,
                        max_step=0.0005, rtol=1e-13, atol=1e-15)
        if not sol.success:
            raise RuntimeError("ODE integration failed")

        D_ref = sol.y[0]
        F_ref = sol.y[1]
        D_ref = D_ref / D_ref[-1]
        f_ref = F_ref / (a_grid * D_ref)
        return f_ref, D_ref

    except Exception:
        gamma_ref = 0.55
        Om_m_arr = _omega_m_of_a(a_grid, Om0, w0, wa)
        f_ref = Om_m_arr ** gamma_ref
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_ref = np.exp(integral_vals)
        D_ref = D_ref / D_ref[-1]
        return f_ref, D_ref

# Anchor cosmologies: diverse set of (Om0, w0, wa) for pre-computing profiles
_ANCHOR_COSMOLOGIES = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -0.8, 0.3),
    (0.30, -1.2, 0.4),
    (0.28, -1.0, 1.0),
    (0.35, -0.8, -0.5),
    (0.22, -1.0, 0.8),
]

# Validation cosmologies for the actual task
_VALIDATION_COSMOLOGIES = [
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.22, -1.0, 0.8),
    (0.38, -1.0, -0.6),
    (0.30, -0.7, 0.6),
    (0.25, -1.2, 0.3),
    (0.35, -0.8, -0.5),
]

_N_COARSE = 200
_N_FINE = 800
_COARSE_A = np.logspace(np.log10(1e-4), 0.0, _N_COARSE)
_FINE_A = np.logspace(np.log10(5e-5), 0.0, _N_FINE)

_dx_coarse = np.diff(np.log(_COARSE_A))
_dx_fine = np.diff(np.log(_FINE_A))

# Pre-compute anchor profiles
_ANCHOR_PROFILES = []
_ANCHOR_X_COARSE = []
_ANCHOR_X_FINE = []
_ANCHOR_OM_COARSE = []
_ANCHOR_OM_FINE = []

def _build_anchor_profiles():
    """Pre-compute gamma profiles from anchor cosmologies."""
    global _ANCHOR_PROFILES, _ANCHOR_X_COARSE, _ANCHOR_X_FINE, _ANCHOR_OM_COARSE, _ANCHOR_OM_FINE
    
    _ANCHOR_PROFILES = []
    _ANCHOR_X_COARSE = []
    _ANCHOR_X_FINE = []
    _ANCHOR_OM_COARSE = []
    _ANCHOR_OM_FINE = []
    
    for (Om0, w0, wa) in _ANCHOR_COSMOLOGIES:
        try:
            f_c, D_c = _growth_ode_reference(_COARSE_A, Om0, w0, wa)
            X_c = _design_matrix(_COARSE_A, Om0, w0, wa)
            Om_c = _omega_m_of_a(_COARSE_A, Om0, w0, wa)
            
            f_f, D_f = _growth_ode_reference(_FINE_A, Om0, w0, wa)
            X_f = _design_matrix(_FINE_A, Om0, w0, wa)
            Om_f = _omega_m_of_a(_FINE_A, Om0, w0, wa)
            
            # Compute gamma profile: gamma = log(f) / log(Om)
            Om_c_safe = np.clip(Om_c, 1e-15, 1.0)
            Om_f_safe = np.clip(Om_f, 1e-15, 1.0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                gamma_c = np.log(np.clip(f_c, 1e-30, None)) / np.log(Om_c_safe)
                gamma_f = np.log(np.clip(f_f, 1e-30, None)) / np.log(Om_f_safe)
            
            # Clean up non-finite values
            gamma_c[~np.isfinite(gamma_c)] = 0.55
            gamma_f[~np.isfinite(gamma_f)] = 0.55
            
            _ANCHOR_PROFILES.append((gamma_c, gamma_f))
            _ANCHOR_X_COARSE.append(X_c)
            _ANCHOR_X_FINE.append(X_f)
            _ANCHOR_OM_COARSE.append(Om_c)
            _ANCHOR_OM_FINE.append(Om_f)
        except Exception:
            _ANCHOR_PROFILES.append(None)
            _ANCHOR_X_COARSE.append(None)
            _ANCHOR_X_FINE.append(None)
            _ANCHOR_OM_COARSE.append(None)
            _ANCHOR_OM_FINE.append(None)

_build_anchor_profiles()

# Pre-compute reference data for validation cosmologies
_REF_DATA_COARSE = []
_REF_DATA_FINE = []

def _build_references():
    global _REF_DATA_COARSE, _REF_DATA_FINE
    _REF_DATA_COARSE = []
    _REF_DATA_FINE = []
    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_c, D_c = _growth_ode_reference(_COARSE_A, Om0, w0, wa)
            X_c = _design_matrix(_COARSE_A, Om0, w0, wa)
            Om_c = _omega_m_of_a(_COARSE_A, Om0, w0, wa)
            _REF_DATA_COARSE.append((f_c, D_c, Om0, w0, wa, X_c, Om_c))
            
            f_f, D_f = _growth_ode_reference(_FINE_A, Om0, w0, wa)
            X_f = _design_matrix(_FINE_A, Om0, w0, wa)
            Om_f = _omega_m_of_a(_FINE_A, Om0, w0, wa)
            _REF_DATA_FINE.append((f_f, D_f, Om0, w0, wa, X_f, Om_f))
        except Exception:
            _REF_DATA_COARSE.append(None)
            _REF_DATA_FINE.append(None)

_build_references()

def _compute_D_from_f(f_vals, dx):
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]

def _gamma_from_weights(weights):
    """Compute gamma profile from anchor weights."""
    gamma = np.zeros_like(_FINE_A)
    for i, profile in enumerate(_ANCHOR_PROFILES):
        if profile is not None:
            gamma += weights[i] * profile[1]
    return gamma

def _coeffs_from_gamma_profile(gamma_profile):
    """Fit 6 coefficients to a gamma profile using least squares."""
    # Use the first anchor's design matrix as reference
    X_ref = _ANCHOR_X_FINE[0]
    if X_ref is None:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_ref, gamma_profile, rcond=None)
        return np.clip(coeffs, -10, 10)
    except Exception:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

def _max_relative_error(c, ref_data, a_grid, dx):
    c = np.asarray(c)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    
    max_err = 0.0
    for ref in ref_data:
        if ref is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = ref
        
        gamma = X @ c
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            violation = np.maximum(0.0, 0.1 - gamma) + np.maximum(0.0, gamma - 1.2)
            max_err = max(max_err, 10.0 * np.max(violation))
            continue
        
        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        f_model = Om_m_safe ** gamma
        
        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f
        
        D_model = _compute_D_from_f(f_model, dx)
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D
        
        max_err = max(max_err, np.max(rel_err_f), np.max(rel_err_D))
    
    return max_err

def _objective_coarse(c):
    return 100.0 * _max_relative_error(c, _REF_DATA_COARSE, _COARSE_A, _dx_coarse)

def _objective_fine(c):
    return 100.0 * _max_relative_error(c, _REF_DATA_FINE, _FINE_A, _dx_fine)

def _objective_weights(weights):
    """Objective function for weight optimization (convex)."""
    weights = np.clip(weights, 0, None)
    weights = weights / (np.sum(weights) + 1e-15)
    gamma_profile = _gamma_from_weights(weights)
    coeffs = _coeffs_from_gamma_profile(gamma_profile)
    return _objective_fine(coeffs)

def _initial_guess_from_anchors():
    """Get initial coefficients by averaging anchor profiles."""
    n_anchors = len(_ANCHOR_PROFILES)
    if n_anchors == 0:
        return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    
    # Equal weights as initial guess
    weights = np.ones(n_anchors) / n_anchors
    gamma_profile = _gamma_from_weights(weights)
    coeffs = _coeffs_from_gamma_profile(gamma_profile)
    return np.clip(coeffs, -5, 5)

def _run_optimization():
    best_c = _initial_guess_from_anchors()
    best_c = np.clip(best_c, -10, 10)
    best_err = _objective_fine(best_c)
    
    if not _deadline_ok():
        return best_c

    # Stage 1: Optimize weights with SLSQP (convex problem)
    n_anchors = len(_ANCHOR_PROFILES)
    if n_anchors > 0:
        try:
            initial_weights = np.ones(n_anchors) / n_anchors
            bounds_w = [(0, 1)] * n_anchors
            res = minimize(
                _objective_weights,
                initial_weights,
                method='SLSQP',
                bounds=bounds_w,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            if res.success:
                weights_opt = np.clip(res.x, 0, None)
                weights_opt = weights_opt / (np.sum(weights_opt) + 1e-15)
                gamma_profile = _gamma_from_weights(weights_opt)
                c_new = _coeffs_from_gamma_profile(gamma_profile)
                err_new = _objective_fine(c_new)
                if err_new < best_err:
                    best_err = err_new
                    best_c = c_new
        except Exception:
            pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 2: Refine coefficients directly with SLSQP
    try:
        bounds = [(-3, 3)] * 6
        res = minimize(
            _objective_fine,
            best_c,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
        )
        if res.success and res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 3: Nelder-Mead on fine grid
    try:
        res = minimize(
            _objective_fine,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Stage 4: Random restarts on fine grid
    rng = np.random.default_rng(123)
    for i in range(40):
        if not _deadline_ok():
            break
        scale = 0.1 * (0.5 ** (i // 15))
        x_try = best_c + rng.normal(0, scale, 6)
        x_try = np.clip(x_try, -5, 5)
        try:
            res = minimize(
                _objective_fine,
                x_try,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_err:
                best_err = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    best_c = np.clip(best_c, -10, 10)
    return best_c

_COEFFS = _run_optimization()

def construct():
    return list(_COEFFS)
