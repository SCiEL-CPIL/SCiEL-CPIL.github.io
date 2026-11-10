"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid

# Global deadline guard
START_TIME = time.time()
DEADLINE = 85.0  # 5s margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    if a <= 0:
        return 1.0
    # Om_m = Om0 * a^-3
    # Om_de = (1-Om0) * exp(-3 * ((1+w0)ln(a) + wa(a-1)))
    de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
    om_m = Om0 * a**(-3.0)
    om_de = (1.0 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _compute_reference_data():
    """
    Pre-compute reference f(a) and D(a) for a grid of cosmologies and a-values.
    This is done once to speed up the optimization loop.
    """
    # Representative cosmologies covering the parameter space
    # We include a mix of LambdaCDM, wCDM, and CPL models
    cosmos = [
        (0.30, -1.0, 0.0),   # LambdaCDM
        (0.30, -0.9, 0.0),   # wCDM
        (0.30, -1.0, 0.5),   # CPL positive wa
        (0.30, -1.0, -0.5),  # CPL negative wa
        (0.25, -0.8, 0.3),   # Low Om0, less negative w0
        (0.35, -1.2, 0.2),   # High Om0, more negative w0
        (0.28, -0.95, -0.2), # Mixed
        (0.32, -1.05, 0.4),  # Mixed
        (0.30, -1.0, 1.0),   # High wa
        (0.30, -1.0, -1.0),  # Low wa
    ]
    
    # a values: log spaced to capture early and late times
    # The integral for D is sensitive to the early times where f is larger
    a_vals = np.geomspace(0.05, 1.0, 100)
    
    ref_data = []
    
    for Om0, w0, wa in cosmos:
        f_refs = np.zeros_like(a_vals)
        D_refs = np.zeros_like(a_vals)
        
        # ODE: dy/dlna = [f - f^2 - 1.5*Om(a), -f*D]
        # y = [f, D]
        
        def ode(t, y):
            a = np.exp(t)
            om_a = _Om_of_a(a, Om0, w0, wa)
            f_val = y[0]
            D_val = y[1]
            df_dlna = f_val - f_val**2 - 1.5 * om_a
            dD_dlna = -f_val * D_val
            return [df_dlna, dD_dlna]
        
        # Initial conditions at a=1 (t=0)
        # f(1) is approximated by Om0^0.55, but for reference we solve ODE.
        # The ODE is stable. We integrate backwards from a=1 to a_min.
        f_init = Om0 ** 0.55
        D_init = 1.0
        
        t_end = np.log(a_vals[0])
        
        try:
            sol = solve_ivp(
                ode, [0.0, t_end], [f_init, D_init],
                method='RK45',
                rtol=1e-9,
                atol=1e-12,
                dense_output=True
            )
            
            if not sol.success:
                continue
                
            # Evaluate at a_vals
            for i, a in enumerate(a_vals):
                t = np.log(a)
                y = sol.sol(t)
                f_refs[i] = y[0]
                D_refs[i] = y[1]
                
        except Exception:
            continue
            
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
        # Compute q, u vectorized
        om_a_vals = np.array([_Om_of_a(a, Om0, w0, wa) for a in a_vals])
        
        # Check validity
        if np.any(om_a_vals <= 0) or np.any(om_a_vals >= 1):
            return 1e10
            
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        # Check gamma bounds [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_approx = om_a_vals ** gamma
        
        # Relative error for f
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs(f_approx - f_refs) / np.abs(f_refs)
            rel_err_f = np.nan_to_num(rel_err_f, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err = max(max_err, np.max(rel_err_f))
        
        # Compute D_approx
        x = np.log(a_vals)
        
        # Integrate f_approx from x to 0 (which is ln(1)=0)
        # D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
        # cumulative_trapezoid integrates from start to end.
        # We want integral from x_i to 0.
        # Let's integrate f_approx w.r.t x from x[0] to x[i]
        cum_int = cumulative_trapezoid(f_approx, x, initial=0.0)
        # The integral from x_i to 0 is total_int - cum_int[i]
        # Where total_int is integral from x[0] to x[-1] (which is 0)
        total_int = cum_int[-1]
        integrals = total_int - cum_int
        
        D_approx = np.exp(-integrals)
        
        # Check D error
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
            rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err = max(max_err, np.max(rel_err_D))
        
    return max_err

def _optimize():
    # Pre-compute reference data
    if time.time() > DEADLINE - 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    ref_data = _compute_reference_data()
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Initial guess: constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_x = x0
    best_err = _error_function(best_x, ref_data)
    
    # Multi-start local optimization
    # We will try several starting points to avoid local minima
    # The landscape is likely smooth, so Nelder-Mead is a good choice
    # We also try L-BFGS-B if gradients were available, but we don't have them.
    
    # Generate diverse starting points
    # 1. Base
    starts = [x0]
    
    # 2. Perturbations around base
    rng = np.random.default_rng(42)
    for _ in range(10):
        perturb = rng.uniform(-0.1, 0.1, 6)
        x_start = x0 + perturb
        # Ensure c0 is reasonable
        x_start[0] = np.clip(x_start[0], 0.4, 0.7)
        starts.append(x_start)
        
    # 3. Physics-based starts
    # Try emphasizing q dependence
    starts.append(np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]))
    starts.append(np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]))
    starts.append(np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]))
    
    for x_start in starts:
        if time.time() > DEADLINE - 5:
            break
            
        try:
            res = minimize(
                _error_function, x_start, args=(ref_data,),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200, 'maxfev': 500}
            )
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if time.time() < DEADLINE - 3:
        try:
            res2 = minimize(
                _error_function, best_x, args=(ref_data,),
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 100, 'maxfev': 200}
            )
            if res2.fun < best_err:
                best_err = res2.fun
                best_x = res2.x
        except Exception:
            pass
            
    return np.clip(best_x, -10, 10)

# Run optimization at module load
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
