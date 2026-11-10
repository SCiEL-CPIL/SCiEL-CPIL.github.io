"""
Strategy: Global Chebyshev-minimax fit.
Instead of local optimization or constrained sub-spaces, we treat the problem
as finding coefficients that minimize the maximum relative error across a
broad, representative set of CPL cosmologies and scale factors.
We use a dense grid and a robust minimization (Nelder-Mead on the max-error
function, initialized by a coarse grid search or differential evolution) to
find the global minimum of the worst-case error.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid

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
    This dense grid allows the minimax optimizer to balance errors globally.
    """
    # Representative cosmologies covering a broad range of CPL parameters
    # We use a denser grid of parameters to ensure the fit is robust
    Om0s = [0.25, 0.30, 0.35]
    w0s = [-1.2, -1.0, -0.8]
    was = [-0.5, 0.0, 0.5]
    
    cosmos = []
    for Om0 in Om0s:
        for w0 in w0s:
            for wa in was:
                # Skip unphysical or edge cases if any, though CPL is broad
                cosmos.append((Om0, w0, wa))
                
    # a values: log spaced to capture early and late times
    # More points to better capture the worst-case error locations
    a_vals = np.geomspace(0.05, 1.0, 80)
    
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
        
        # Use a good initial guess for f
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
        
        # Evaluate at a_vals
        t_vals = np.log(a_vals)
        y_all = sol.sol(t_vals)
        f_refs = y_all[0]
        D_refs = y_all[1]
        
        ref_data.append((Om0, w0, wa, a_vals, f_refs, D_refs))
        
    return ref_data

def _error_function(coeffs, ref_data):
    """
    Compute max relative error for f and D across all reference data.
    coeffs: [c0, c1, c2, c3, c4, c5]
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
        om_a_vals = _Om_of_a(a_vals, Om0, w0, wa)
        
        if np.any(om_a_vals <= 0) or np.any(om_a_vals >= 1):
            return 1e10
            
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        # Constraint: Gamma must lie in [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_approx = om_a_vals ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs(f_approx - f_refs) / np.abs(f_refs)
            rel_err_f = np.nan_to_num(rel_err_f, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err = max(max_err, np.max(rel_err_f))
        
        x = np.log(a_vals)
        cum_int = cumulative_trapezoid(f_approx, x, initial=0.0)
        integrals = cum_int[-1] - cum_int
        D_approx = np.exp(-integrals)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
            rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err = max(max_err, np.max(rel_err_D))
        
    return max_err

def _optimize():
    # Pre-compute reference data
    if time.time() > DEADLINE - 25:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    ref_data = _compute_reference_data()
    
    if len(ref_data) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Full 6-parameter optimization
    # Initial guess: standard gamma ~ 0.55
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds for the 6 parameters
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    
    best_x = x0
    best_err = _error_function(best_x, ref_data)
    
    if time.time() > DEADLINE - 20:
        return best_x
    
    # Phase 1: Global search with differential evolution
    # Use a smaller population and fewer iterations to fit within time budget,
    # but enough to escape local minima.
    try:
        def de_callback(xk, convergence=None):
            return time.time() > DEADLINE - 15
        
        de_result = differential_evolution(
            _error_function, bounds, args=(ref_data,),
            maxiter=30,
            popsize=15,
            tol=1e-5,
            seed=42,
            init='sobol',
            callback=de_callback,
            polish=False
        )
        if de_result.fun < best_err:
            best_err = de_result.fun
            best_x = de_result.x
    except Exception:
        pass
    
    if time.time() > DEADLINE - 10:
        return best_x
    
    # Phase 2: Local refinement with Nelder-Mead
    # Nelder-Mead is robust for non-smooth functions like max-error
    try:
        res = minimize(
            _error_function, best_x, args=(ref_data,),
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 5000}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() > DEADLINE - 5:
        return best_x
    
    # Phase 3: Final ultra-precise polish
    try:
        res2 = minimize(
            _error_function, best_x, args=(ref_data,),
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500, 'maxfev': 2000}
        )
        if res2.fun < best_err:
            best_err = res2.fun
            best_x = res2.x
    except Exception:
        pass
    
    # Clamp to valid range
    best_x = np.clip(best_x, -10, 10)
    
    return best_x

# Run optimization at module load
_t0 = time.time()
try:
    _coeffs = _optimize().tolist()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Verify coefficients are valid
if not all(np.isfinite(c) for c in _coeffs):
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
# EVOLVE-BLOCK-END
