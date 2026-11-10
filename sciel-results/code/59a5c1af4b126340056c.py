"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp

# Global deadline guard
START_TIME = time.time()
DEADLINE = 85.0  # 5s margin

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    if a <= 0:
        return 1.0
    # Om_m = Om0 * a^-3
    # Om_de = (1-Om0) * exp(-3 * integral_1^a (1+w(x))/x dx)
    # 1+w(x) = 1+w0+wa(1-x)
    # integral (1+w0)/x + wa(1-x)/x dx = (1+w0)ln(a) + wa(a-1)
    # So Om_de = (1-Om0) * exp(-3 * ((1+w0)ln(a) + wa(a-1)))
    de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
    om_m = Om0 * a**(-3.0)
    om_de = (1.0 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _compute_reference_data():
    """
    Pre-compute reference f(a) and D(a) for a grid of cosmologies and a-values.
    Returns a list of tuples: (Om0, w0, wa, a_vals, f_refs, D_refs)
    """
    # Representative cosmologies
    # Covering a reasonable range of CPL parameters
    cosmos = [
        (0.30, -1.0, 0.0),   # LambdaCDM
        (0.30, -0.9, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -0.8, 0.3),
        (0.35, -1.2, 0.2),
        (0.28, -0.95, -0.2),
        (0.32, -1.05, 0.4),
    ]
    
    # a values: log spaced to capture early and late times
    a_vals = np.geomspace(0.05, 1.0, 50)
    
    ref_data = []
    
    for Om0, w0, wa in cosmos:
        f_refs = np.zeros_like(a_vals)
        D_refs = np.zeros_like(a_vals)
        
        # We need to solve the ODE for f and D
        # f' = f - f^2 - 1.5 Om(a)  (with respect to ln a)
        # D' = -f D (with respect to ln a)
        
        # We can integrate backwards from a=1 where f(1) ~ Om0^0.55 and D(1)=1
        # Or integrate forwards from a=0.05? 
        # The ODE is stiff near a=0? No, usually well-behaved.
        # Let's integrate from a=1 down to a_min.
        
        a_min = a_vals[0]
        a_max = 1.0
        
        # Define ODE: dy/dlna = [f - f^2 - 1.5*Om(a), -f*y2]
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
        f_init = Om0 ** 0.55
        D_init = 1.0
        
        # Integrate from t=0 to t=ln(a_min)
        t_end = np.log(a_min)
        
        sol = solve_ivp(
            ode, [0.0, t_end], [f_init, D_init],
            method='RK45',
            rtol=1e-9,
            atol=1e-12,
            dense_output=True
        )
        
        if not sol.success:
            # Fallback to simple Euler if solver fails
            n_steps = 1000
            ts = np.linspace(0.0, t_end, n_steps + 1)
            f = f_init
            D = D_init
            for i in range(n_steps):
                a_i = np.exp(ts[i])
                om_i = _Om_of_a(a_i, Om0, w0, wa)
                df = f - f**2 - 1.5 * om_i
                dD = -f * D
                f += df * (ts[i+1] - ts[i])
                D += dD * (ts[i+1] - ts[i])
                if i < n_steps:
                    # Store intermediate? No, just use dense output if available
                    pass
            
            # This fallback is messy, let's just return zeros if it fails
            # But solve_ivp is robust.
            continue
            
        # Evaluate at a_vals
        for i, a in enumerate(a_vals):
            t = np.log(a)
            if t <= t_end or t >= 0:
                y = sol.sol(t)
                f_refs[i] = y[0]
                D_refs[i] = y[1]
            else:
                # Should not happen
                f_refs[i] = f_init
                D_refs[i] = D_init
                
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
        # Compute q, u
        om_a_vals = np.array([_Om_of_a(a, Om0, w0, wa) for a in a_vals])
        
        # Check validity
        if np.any(om_a_vals <= 0) or np.any(om_a_vals >= 1):
            return 1e10
            
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
        
        # Check gamma bounds
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
        
        from scipy.integrate import cumulative_trapezoid
        cum_int = cumulative_trapezoid(f_approx, x, initial=0.0)
        
        integrals = cum_int[-1] - cum_int
        D_approx = np.exp(-integrals)
        
        # Check D error
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
            rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err = max(max_err, np.max(rel_err_D))
        
    return max_err

def _optimize():
    # Pre-compute reference data
    if time.time() > DEADLINE - 20:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    ref_data = _compute_reference_data()
    
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds
    bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    
    # Strategy:
    # 1. Global search with differential_evolution (small population, few iterations)
    # 2. Local polish with Nelder-Mead or L-BFGS-B
    
    best_x = x0
    best_err = _error_function(best_x, ref_data)
    
    if time.time() > DEADLINE - 15:
        return best_x
        
    # Differential Evolution with time-aware stopping
    try:
        # Use a callback to stop if time is running out
        def de_callback(xk, convergence=None):
            return time.time() > DEADLINE - 10
        
        de_result = differential_evolution(
            _error_function, bounds, args=(ref_data,),
            maxiter=30,
            popsize=12,
            tol=1e-5,
            seed=42,
            init='sobol',
            callback=de_callback,
            polish=False  # We'll do our own polish
        )
        if de_result.fun < best_err:
            best_err = de_result.fun
            best_x = de_result.x
    except Exception:
        pass
        
    if time.time() > DEADLINE - 8:
        return best_x
        
    # Local Polish with tighter tolerance
    try:
        res = minimize(
            _error_function, best_x, args=(ref_data,),
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 500}
        )
        if res.fun < best_err:
            best_err = res.fun
            best_x = res.x
    except Exception:
        pass
        
    if time.time() > DEADLINE - 3:
        return best_x
        
    # Second polish pass with even tighter tolerance
    try:
        res2 = minimize(
            _error_function, best_x, args=(ref_data,),
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 100, 'maxfev': 200}
        )
        if res2.fun < best_err:
            best_err = res2.fun
            best_x = res2.x
    except Exception:
        pass
        
    return np.clip(best_x, -10, 10)

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
