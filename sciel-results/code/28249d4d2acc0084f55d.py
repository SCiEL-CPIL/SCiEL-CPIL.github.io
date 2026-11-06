"""Improved growth-index approximation with Differential Evolution and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

_start_time = time.time()
_DEADLINE = 85.0 # 5s safety margin from 90s limit

# Validation cosmologies (Om0, w0, wa)
_cosmologies = [
    (0.3, -1.0, 0.0),
    (0.3, -0.9, 0.1),
    (0.3, -1.1, 0.2),
    (0.3, -0.8, -0.2),
    (0.3, -1.2, 0.3),
    (0.2, -1.0, 0.0),
    (0.4, -1.0, 0.0),
    (0.3, -0.5, 0.0),
]

# Scale factor grid
_a_grid = np.linspace(0.05, 1.0, 50) # Slightly finer grid for better accuracy
_ref_data = {}
_ref_precomputed = False

def _precompute_references():
    """Solve the ODE for each cosmology to get reference f and D."""
    global _ref_precomputed
    if _ref_precomputed:
        return
    
    for Om0, w0, wa in _cosmologies:
        Ode0 = 1.0 - Om0
        a_vals = _a_grid
        
        def dlnHdx(a):
            Om_a = Om0 * a**(-3)
            if Ode0 == 0:
                Ode_a = 0.0
                dOde_da = 0.0
            else:
                # log(Ode_a) = log(Ode0) - 3(1+w0)log(a) - 3wa(1-a)
                log_Ode_a = np.log(Ode0) - 3*(1+w0)*np.log(a) - 3*wa*(1-a)
                Ode_a = np.exp(log_Ode_a)
                dOde_da = Ode_a * (-3*(1+w0)/a + 3*wa)
            H2 = Om_a + Ode_a
            dH2_da = -3*Om0*a**(-4) + dOde_da
            dH2_dx = dH2_da * a
            return 0.5 * dH2_dx / H2
        
        def ode_system(x, y):
            a = np.exp(x)
            Om = Om0 * a**(-3)
            dlnHdx_val = dlnHdx(a)
            dDdx = y[1]
            d2Ddx2 = -3 * Om * y[0] - (3 + dlnHdx_val) * y[1]
            return [dDdx, d2Ddx2]
        
        # Initial conditions at a=1 (x=0)
        f1 = Om0**0.6 # Approximate initial slope, usually close to Om0^0.55
        D1 = 1.0
        D1_prime = f1 * D1
        x_vals = np.log(a_vals)
        x_min = min(x_vals)
        
        sol = solve_ivp(ode_system, [0.0, x_min], [D1, D1_prime], 
                        rtol=1e-10, atol=1e-12, method='DOP853',
                        dense_output=True)
        
        D_vals = np.zeros(len(a_vals))
        f_vals = np.zeros(len(a_vals))
        for i, x in enumerate(x_vals):
            y = sol.sol(x)
            D_vals[i] = y[0]
            f_vals[i] = y[1] / y[0]
        
        D_norm = D_vals / D_vals[-1]
        _ref_data[(Om0, w0, wa)] = (f_vals, D_norm)
    
    _ref_precomputed = True

def _compute_model_fast(coeffs, Om0, w0, wa):
    """Compute model f and D for given coefficients and cosmology."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a_vals = _a_grid
    Om_a = Om0 * a_vals**(-3)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a_vals)
    
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
    
    # Clip gamma to valid range to avoid numerical issues, 
    # but note that the evaluator might check validity separately. 
    # The task says "Gamma must lie in [0.1, 1.2] at all checked locations".
    # We should ensure our search stays within this or penalize heavily.
    # For speed in evaluation, we clip, but the final returned coeffs must be valid.
    gamma_clipped = np.clip(gamma, 0.1, 1.2)
    
    f_vals = Om_a**gamma_clipped
    
    x_vals = np.log(a_vals)
    
    # Compute integral from a to 1: D(a)/D(1) = exp(-int_a^1 f(x) dx)
    # cumulative_trapezoid(f, x) gives int_x0^x f dx
    # We want int_x^x_last f dx = int_x0^x_last - int_x0^x
    integral_forward = cumulative_trapezoid(f_vals, x_vals, initial=0.0)
    integral_to_0 = integral_forward[-1] - integral_forward
    
    D_norm = np.exp(-integral_to_0)
    
    return f_vals, D_norm, gamma

def _evaluate_error(coeffs):
    """Calculate max relative error across all cosmologies."""
    if time.time() - _start_time > _DEADLINE - 2:
        return 1e10
    
    max_err = 0.0
    for Om0, w0, wa in _cosmologies:
        f_ref, D_ref = _ref_data[(Om0, w0, wa)]
        f_mod, D_mod, gamma = _compute_model_fast(coeffs, Om0, w0, wa)
        
        # Check validity of gamma
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        err_f = np.max(np.abs((f_mod - f_ref) / f_ref))
        err_D = np.max(np.abs((D_mod - D_ref) / D_ref))
        
        max_err = max(max_err, err_f, err_D)
        
    return max_err

def _is_valid(coeffs):
    """Check if gamma is within [0.1, 1.2] for all cosmologies and a values."""
    for Om0, w0, wa in _cosmologies:
        _, _, gamma = _compute_model_fast(coeffs, Om0, w0, wa)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
    return True

def construct():
    global _start_time
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    _precompute_references()
    
    # Define bounds for coefficients
    # Gamma must be in [0.1, 1.2]. 
    # c0 is the main driver. q is in [0, 1-Om_min] approx [0, 0.8].
    # u is in [w0+wa, 1+w0] approx [-1.5, 1].
    # We set bounds to [-2, 2] for all coeffs to keep the search space manageable 
    # and ensure gamma stays in range more easily.
    bounds = [(-2.0, 2.0)] * 6
    
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = _evaluate_error(best_coeffs)
    
    # Phase 1: Differential Evolution for global search
    # We limit the time for DE to leave room for polish and fallbacks
    de_time_limit = 60.0
    de_start = time.time()
    
    def de_objective(x):
        if time.time() - de_start > de_time_limit:
            return 1e10
        return _evaluate_error(x)
    
    try:
        result_de = differential_evolution(
            de_objective, 
            bounds, 
            seed=42,
            maxiter=100,
            tol=1e-5,
            popsize=15,
            polish=False, # We will do our own polish
            init='latinhypercube'
        )
        if result_de.fun < best_score and _is_valid(result_de.x):
            best_score = result_de.fun
            best_coeffs = result_de.x
    except Exception:
        pass
    
    if time.time() - _start_time > _DEADLINE - 10:
        if _is_valid(best_coeffs):
            return np.clip(best_coeffs, -10, 10).tolist()
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Phase 2: Local Polish using Nelder-Mead
    # Start from the best DE result
    try:
        result_nm = minimize(
            _evaluate_error, 
            best_coeffs, 
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result_nm.fun < best_score and _is_valid(result_nm.x):
            best_score = result_nm.fun
            best_coeffs = result_nm.x
    except Exception:
        pass
    
    # Phase 3: Try a few other starting points if time permits
    starting_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
    ]
    
    for x0 in starting_points:
        if time.time() - _start_time > _DEADLINE - 5:
            break
        try:
            result_nm2 = minimize(
                _evaluate_error, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result_nm2.fun < best_score and _is_valid(result_nm2.x):
                best_score = result_nm2.fun
                best_coeffs = result_nm2.x
        except Exception:
            pass
    
    # Final validity check
    if _is_valid(best_coeffs):
        return np.clip(best_coeffs, -10, 10).tolist()
    
    # Fallback to safe constant gamma
    return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
