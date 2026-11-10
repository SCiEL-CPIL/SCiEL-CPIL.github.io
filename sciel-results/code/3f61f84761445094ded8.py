"""
Strategy: Direct Minimax (Chebyshev) Optimization via Sequential Quadratic Programming (SQP).
Instead of fitting a least-squares proxy for gamma and then polishing, we directly minimize
the maximum relative error (L-infinity norm) of the observables f and D using a custom
SQP implementation. This treats the problem as a constrained non-linear optimization where
the objective is the worst-case error across a dense, high-fidelity grid of cosmologies and
scale factors. We use the ODE-integrated reference data to define the exact error surface
and apply active-set logic to focus on the points that define the envelope of the error.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize

# Global deadline guard
START_TIME = time.time()
DEADLINE = 85.0  # 5s margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    if np.isscalar(a):
        if a <= 0:
            return 1.0
        de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
        om_m = Om0 * a**(-3.0)
        om_de = (1.0 - Om0) * de_factor
        return om_m / (om_m + om_de)
    a = np.asarray(a, dtype=float)
    de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
    om_m = Om0 * a**(-3.0)
    om_de = (1.0 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _compute_reference_data():
    """
    Pre-compute reference f(a) and D(a) for a dense grid of cosmologies and a-values.
    We use a slightly finer grid to ensure the minimax objective is well-defined.
    """
    Om0s = [0.25, 0.30, 0.35]
    w0s = [-1.2, -1.0, -0.8]
    was = [-0.5, 0.0, 0.5]
    
    cosmos = []
    for Om0 in Om0s:
        for w0 in w0s:
            for wa in was:
                cosmos.append((Om0, w0, wa))
                
    a_vals = np.geomspace(0.05, 1.0, 120)
    ref_data = []
    
    for Om0, w0, wa in cosmos:
        f_refs = np.zeros_like(a_vals)
        D_refs = np.zeros_like(a_vals)
        
        a_min = a_vals[0]
        
        def ode(t, y):
            a = np.exp(t)
            om_a = _Om_of_a(a, Om0, w0, wa)
            f_val = y[0]
            D_val = y[1]
            df_dlna = f_val - f_val**2 - 1.5 * om_a
            dD_dlna = -f_val * D_val
            return [df_dlna, dD_dlna]
        
        f_init = Om0 ** 0.55
        D_init = 1.0
        t_end = np.log(a_min)
        
        sol = solve_ivp(
            ode, [0.0, t_end], [f_init, D_init],
            method='DOP853',
            rtol=1e-10,
            atol=1e-13,
            dense_output=True
        )
        
        if not sol.success:
            continue
        
        t_vals = np.log(a_vals)
        y_all = sol.sol(t_vals)
        f_refs = y_all[0]
        D_refs = y_all[1]
        
        ref_data.append((Om0, w0, wa, a_vals, f_refs, D_refs))
        
    return ref_data

def _error_function(coeffs, ref_data, active_indices=None):
    """
    Compute max relative error for f and D across all reference data.
    If active_indices is provided, only compute error on those points (for SQP steps).
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    max_idx = -1
    
    # Pre-compute features for all points if not active
    if active_indices is None:
        all_indices = []
        for idx, (Om0, w0, wa, a_vals, f_refs, D_refs) in enumerate(ref_data):
            for i in range(len(a_vals)):
                all_indices.append((idx, i))
        active_indices = all_indices
    
    for c_idx, a_idx in active_indices:
        Om0, w0, wa, a_vals, f_refs, D_refs = ref_data[c_idx]
        
        a = a_vals[a_idx]
        om_a = _Om_of_a(a, Om0, w0, wa)
        
        if om_a <= 0 or om_a >= 1:
            return 1e10, -1
            
        q = 1.0 - om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        if gamma < 0.1 or gamma > 1.2:
            return 1e10, -1
            
        f_approx = om_a ** gamma
        f_ref = f_refs[a_idx]
        
        if f_ref == 0:
            return 1e10, -1
            
        rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
        
        if rel_err_f > max_err:
            max_err = rel_err_f
            max_idx = (c_idx, a_idx)
            
    # For D, we need the integral. In SQP, we might approximate D error or just focus on f first.
    # However, the task requires minimizing the max error of BOTH.
    # Computing D for every single point in an active set is expensive.
    # Strategy: In the SQP inner loop, we primarily drive f-error.
    # Then, we do a full evaluation for D-error to update the active set.
    
    return max_err, max_idx

def _full_error_eval(coeffs, ref_data):
    """
    Full evaluation of max error for f and D.
    Returns (max_err, list of active (c_idx, a_idx) pairs that are near the max).
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err_f = 0.0
    max_err_D = 0.0
    active_f = []
    active_D = []
    
    for c_idx, (Om0, w0, wa, a_vals, f_refs, D_refs) in enumerate(ref_data):
        om_a_vals = _Om_of_a(a_vals, Om0, w0, wa)
        
        if np.any(om_a_vals <= 0) or np.any(om_a_vals >= 1):
            return 1e10, []
            
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10, []
            
        f_approx = om_a_vals ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs(f_approx - f_refs) / np.abs(f_refs)
            rel_err_f = np.nan_to_num(rel_err_f, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err_f = max(max_err_f, np.max(rel_err_f))
        
        # Identify active points for f (within 90% of max)
        if max_err_f > 0:
            threshold_f = 0.9 * max_err_f
            for i in range(len(a_vals)):
                if rel_err_f[i] >= threshold_f:
                    active_f.append((c_idx, i))
                    
        x = np.log(a_vals)
        cum_int = cumulative_trapezoid(f_approx, x, initial=0.0)
        integrals = cum_int[-1] - cum_int
        D_approx = np.exp(-integrals)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
            rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err_D = max(max_err_D, np.max(rel_err_D))
        
        if max_err_D > 0:
            threshold_D = 0.9 * max_err_D
            for i in range(len(a_vals)):
                if rel_err_D[i] >= threshold_D:
                    active_D.append((c_idx, i))
                    
    max_err = max(max_err_f, max_err_D)
    
    # Combine active sets, limit size to prevent SQP from becoming too large
    active_set = active_f + active_D
    if len(active_set) > 50:
        # Simple heuristic: keep the ones with highest individual error
        # For simplicity in this implementation, we just truncate, but a proper sort would be better
        # Let's sort by error value if we had it, but we don't store it here.
        # We'll just take the first 50 unique ones.
        seen = set()
        unique_active = []
        for idx in active_set:
            if idx not in seen:
                unique_active.append(idx)
                seen.add(idx)
                if len(unique_active) >= 50:
                    break
        active_set = unique_active
    else:
        # Deduplicate
        seen = set()
        unique_active = []
        for idx in active_set:
            if idx not in seen:
                unique_active.append(idx)
                seen.add(idx)
        active_set = unique_active
        
    return max_err, active_set

def _optimize():
    if time.time() > DEADLINE - 25:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    ref_data = _compute_reference_data()
    
    if len(ref_data) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial guess: Standard Linder coefficients
    # gamma = 0.55 + 0.3*(w0+wa)*(1-Om)/Om ... approx
    # Our ansatz: c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # Let's start with c0=0.55, others 0.
    best_x = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Get initial error and active set
    best_err, active_set = _full_error_eval(best_x, ref_data)
    
    # SQP Loop
    max_sqps = 50
    for i_sqps in range(max_sqps):
        if time.time() > DEADLINE - 10:
            break
            
        if len(active_set) == 0:
            break
            
        # Define the sub-problem: Minimize max error on active set
        # We use a linear approximation of the error function around best_x
        # Since error is non-smooth (max), we use a standard minimax approach:
        # Minimize t subject to e_i(x) <= t for i in active_set
        # Linearize e_i(x) at best_x.
        
        # Gradient of error w.r.t coeffs for each active point
        # e_f = |f_approx - f_ref| / f_ref
        # e_D = |D_approx - D_ref| / D_ref
        
        # We will use scipy.optimize.minimize with a custom objective that
        # only evaluates the active set, using a smooth approximation (soft-max)
        # to make it differentiable.
        
        def soft_max_objective(x, active_set, ref_data, temp=0.01):
            c0, c1, c2, c3, c4, c5 = x
            errors = []
            for c_idx, a_idx in active_set:
                Om0, w0, wa, a_vals, f_refs, D_refs = ref_data[c_idx]
                a = a_vals[a_idx]
                om_a = _Om_of_a(a, Om0, w0, wa)
                if om_a <= 0 or om_a >= 1:
                    return 1e10
                q = 1.0 - om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10
                f_approx = om_a ** gamma
                f_ref = f_refs[a_idx]
                if f_ref == 0:
                    return 1e10
                e_f = abs(f_approx - f_ref) / abs(f_ref)
                errors.append(e_f)
                
                # Approximate D error using local derivative
                # dD/D = -f * da/a ... roughly
                # For speed, we skip D in the inner soft-max loop and rely on the full eval
                # to update the active set. This is a hybrid SQP.
            
            if not errors:
                return 1e10
            errors = np.array(errors)
            # Soft max
            return temp * np.log(np.sum(np.exp(errors / temp)))
        
        try:
            res = minimize(
                soft_max_objective, best_x, args=(active_set, ref_data),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 100, 'maxfev': 500}
            )
            
            if res.fun < best_err:
                # Verify with full evaluation
                new_err, new_active = _full_error_eval(res.x, ref_data)
                if new_err < best_err:
                    best_err = new_err
                    best_x = res.x
                    active_set = new_active
                else:
                    # If full error didn't improve, keep old active set but maybe shrink temp?
                    pass
            else:
                # No improvement, break
                break
                
        except Exception:
            break
            
    # Final polish with full error function using Nelder-Mead
    if time.time() < DEADLINE - 5:
        try:
            res2 = minimize(
                lambda x: _full_error_eval(x, ref_data)[0],
                best_x,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 200, 'maxfev': 1000}
            )
            if res2.fun < best_err:
                best_err = res2.fun
                best_x = res2.x
        except Exception:
            pass
            
    best_x = np.clip(best_x, -10, 10)
    
    return best_x.tolist()

# Run optimization at module load
_t0 = time.time()
try:
    _coeffs = _optimize()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Verify coefficients are valid
if not all(np.isfinite(c) for c in _coeffs):
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
# EVOLVE-BLOCK-END
