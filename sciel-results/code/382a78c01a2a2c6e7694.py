"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
# Use 85s as a safe deadline (evaluator kills at 90s, leave 5s margin)
_DEADLINE = _START_TIME + 85.0

# Global cache for reference data
_REF_CACHE = None

def _om_a_vec(a, Om0, w0, wa):
    """Vectorized matter density fraction."""
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.maximum(a, 1e-10)
    w_eff = 1.0 + w0 + wa
    # E^2 = Om_m + Om_de
    # Om_m = Om0 * a^-3
    # Om_de = (1-Om0) * exp(-3(1+w0+wa)ln(a) + 3wa*a)
    de_term = (1.0 - Om0) * np.exp(-3.0 * w_eff * np.log(a_safe) + 3.0 * wa * a_safe)
    m_term = Om0 * np.exp(-3.0 * np.log(a_safe))
    E2 = m_term + de_term
    return m_term / E2

def _gamma_vec(a, Om0, w0, wa, c):
    """Vectorized gamma."""
    c0, c1, c2, c3, c4, c5 = c
    Om_a = _om_a_vec(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_vec(a, Om0, w0, wa, c):
    """Vectorized f = Om(a)^gamma."""
    Om_a = np.maximum(_om_a_vec(a, Om0, w0, wa), 1e-12)
    g = _gamma_vec(a, Om0, w0, wa, c)
    return np.exp(g * np.log(Om_a))

def _D_vec(a, Om0, w0, wa, c):
    """Vectorized normalized D(a)/D(1)."""
    a = np.asarray(a, dtype=np.float64)
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    a_m = a[mask]
    a_min = max(np.min(a_m), 1e-6)
    
    # Use a dense grid for integration accuracy
    # We integrate from log(a_min) to 0
    n = 1000
    ln_grid = np.linspace(np.log(a_min), 0.0, n)
    a_grid = np.exp(ln_grid)
    
    f_grid = _f_vec(a_grid, Om0, w0, wa, c)
    
    # Cumulative trapezoidal integration of f(e^x) dx
    # We want I(a) = - integral_a^1 f(e^x) dx = integral_0^{ln(a)} f(e^x) dx
    # Let's compute cumulative sum from 0 to ln(a_min) and flip?
    # Easier: compute cumulative integral from ln(a_min) to 0.
    # Let J(y) = integral_{ln(a_min)}^y f(e^x) dx.
    # Then integral_a^1 f(e^x) dx = integral_{ln(a)}^0 f(e^x) dx = J(0) - J(ln(a)).
    # D(a)/D(1) = exp(- (J(0) - J(ln(a)))) = exp(J(ln(a)) - J(0)).
    
    dx = np.diff(ln_grid)
    trap = 0.5 * (f_grid[1:] + f_grid[:-1]) * dx
    cum = np.concatenate([[0.0], np.cumsum(trap)])
    # cum[i] is integral from ln_grid[0] to ln_grid[i]
    J_0 = cum[-1]
    
    ln_m = np.log(a_m)
    # Interpolate J(ln(a))
    J_ln_a = np.interp(ln_m, ln_grid, cum)
    
    D[mask] = np.exp(J_ln_a - J_0)
    return D

def _build_refs():
    """Build reference data using ODE."""
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    
    # Representative models covering a range of w0, wa
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.30, -1.5, 1.0),
        (0.30, -0.5, -0.5),
        (0.20, -1.2, 0.8),
        (0.40, -0.7, -0.3),
        (0.30, -1.0, 0.3), # Add a few more for robustness
        (0.30, -0.8, -0.2),
    ]
    
    # Dense grid for evaluation
    # a from 0.1 to 1.0
    a_vals = np.linspace(0.1, 1.0, 500)
    ln_a_eval = np.log(a_vals)
    
    ref_models = []
    for Om0, w0, wa in models:
        def ode(ln_a, y):
            f = y[0]
            a = np.exp(ln_a)
            Om_a = _om_a_vec(np.array([a]), Om0, w0, wa)[0]
            # E^2 = Om_m + Om_de
            # d(E^2)/da = -3 Om_m/a - 3(1+w0+wa/a) Om_de
            # dln(E^2)/dln(a) = a/E^2 * d(E^2)/da
            # Let's compute derivatives carefully
            # Om_m = Om0 a^-3
            # Om_de = (1-Om0) a^-3(1+w0) e^(3wa a)  <-- Wait, standard CPL is p = w0 + wa(1-a)
            # rho_de ~ a^-3(1+w) where w = w0 + wa(1-a)
            # rho_de ~ a^-3(1+w0+wa) e^(3wa a) ?
            # Let's stick to the vectorized _om_a_vec logic which is consistent.
            
            # Recalculate E2 and dE2/dlna for ODE
            a_safe = a
            w_eff = 1.0 + w0 + wa
            de_term = (1.0 - Om0) * np.exp(-3.0 * w_eff * ln_a + 3.0 * wa * a_safe)
            m_term = Om0 * np.exp(-3.0 * ln_a)
            E2 = m_term + de_term
            
            # d(m_term)/dlna = -3 m_term
            # d(de_term)/dlna = de_term * (-3 w_eff + 3 wa a)
            dE2_dlna = -3.0 * m_term + de_term * (-3.0 * w_eff + 3.0 * wa * a_safe)
            
            dlnE2_dlna = dE2_dlna / E2
            HpH = 0.5 * dlnE2_dlna
            
            fp = -1.5 * Om_a / E2 - (3.0/a + HpH) * f - f * f
            return [fp]
        
        f0 = 0.55 * Om0**0.55
        try:
            sol = solve_ivp(ode, [0.0, np.log(0.1)], [f0], t_eval=ln_a_eval,
                            method='RK45', rtol=1e-10, atol=1e-12)
            f_ref = sol.y[0] if sol.success else None
        except Exception:
            f_ref = None
            
        if f_ref is None or len(f_ref) != len(a_vals):
            # Fallback to seed approximation if ODE fails
            f_ref = _f_vec(a_vals, Om0, w0, wa, [0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        # Reference D via trapezoid on the same dense grid
        # We already have f_ref on a_vals (which is ln_grid effectively)
        # Let's integrate f_ref over ln_a_eval
        dx = np.diff(ln_a_eval)
        trap = 0.5 * (f_ref[1:] + f_ref[:-1]) * dx
        cum = np.concatenate([[0.0], np.cumsum(trap)])
        J_0 = cum[-1]
        D_ref = np.exp(cum - J_0)
        
        ref_models.append({
            'Om0': Om0, 'w0': w0, 'wa': wa,
            'a': a_vals, 'f_ref': f_ref, 'D_ref': D_ref
        })
        
    _REF_CACHE = ref_models
    return ref_models

def _score(c):
    """Smooth surrogate: log-sum-exp soft-max of relative errors + gamma penalty."""
    refs = _build_refs()
    all_errors = []
    gamma_violation = 0.0
    
    for m in refs:
        Om0, w0, wa = m['Om0'], m['w0'], m['wa']
        a = m['a']
        f_ref = m['f_ref']
        D_ref = m['D_ref']
        
        f_ap = _f_vec(a, Om0, w0, wa, c)
        D_ap = _D_vec(a, Om0, w0, wa, c)
        
        ef = np.abs(f_ap - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        eD = np.abs(D_ap - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        all_errors.extend(ef.tolist())
        all_errors.extend(eD.tolist())
        
        g = _gamma_vec(a, Om0, w0, wa, c)
        viol_low = np.maximum(0.1 - g, 0)
        viol_high = np.maximum(g - 1.2, 0)
        gamma_violation = max(gamma_violation, float(np.max(viol_low)), float(np.max(viol_high)))
    
    if gamma_violation > 0:
        # Hard penalty for constraint violation
        return 10.0 * gamma_violation + float(np.max(all_errors))
    
    # Log-sum-exp soft-max with temperature T:
    # soft_max = T * log(sum(exp(x_i / T)))
    # As T -> 0, approaches max(x_i). Use small T for sharpness.
    err_arr = np.array(all_errors, dtype=np.float64)
    T = 0.01  # Small temperature for near-max behavior
    scaled = err_arr / T
    max_scaled = np.max(scaled)
    if max_scaled > 500:
        # Overflow protection
        return float(max_scaled * T)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return float(T * log_sum)

def _optimize():
    """Run global and local optimization with smooth surrogate."""
    # Seed with constant gamma
    seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = seed.copy()
    best_score = _score(seed)
    
    bounds = [(-10.0, 10.0)] * 6
    
    # 1. Differential Evolution for global exploration
    # Use the smooth surrogate to help DE navigate
    if time.time() < _DEADLINE - 25.0:
        try:
            de_res = differential_evolution(
                _score, bounds, 
                maxiter=200, popsize=15, 
                tol=1e-6, 
                seed=42, 
                polish=False
            )
            if de_res.fun < best_score:
                best_c = de_res.x
                best_score = de_res.fun
        except Exception:
            pass

    # 2. Nelder-Mead for local refinement using smooth surrogate
    if time.time() < _DEADLINE - 8.0:
        try:
            nm_res = minimize(
                _score, best_c, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if nm_res.fun < best_score:
                best_c = nm_res.x
                best_score = nm_res.fun
        except Exception:
            pass
            
    # 3. Final high-precision polish with tight tolerances
    if time.time() < _DEADLINE - 2.0:
        try:
            nm_res2 = minimize(
                _score, best_c, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if nm_res2.fun < best_score:
                best_c = nm_res2.x
                best_score = nm_res2.fun
        except Exception:
            pass

    # Ensure bounds are respected
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return best_c

# Precompute references at import to maximize time for optimization
try:
    _build_refs()
except Exception:
    _REF_CACHE = None

_optimized = None
def construct():
    global _optimized
    if _optimized is None:
        _optimized = _optimize()
    return _optimized
