"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

# Global deadline guard
T_START = time.time()
DEADLINE = T_START + 85.0  # 5s safety margin

def construct():
    """Return six coefficients for the growth-index ansatz."""
    # Check if we have already done work (in case of re-entry, though unlikely)
    if 'construct._ref_data' not in globals():
        if time.time() > DEADLINE - 10:
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        construct._ref_data = _make_reference_data()
        construct._a_grid = np.linspace(0.1, 1.0, 300)
    
    ref_data = construct._ref_data
    a_grid = construct._a_grid
    
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Initial guess
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = _objective(best_c, ref_data, a_grid)

    # Define bounds for differential evolution
    # Coefficients should be in [-10, 10] per task, but likely much smaller for stability
    bounds = [(-5, 5), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]
    
    # 1. Global Search: Differential Evolution
    if time.time() < DEADLINE - 20:
        try:
            # Use a reasonable popsize and maxiter
            # tol controls the convergence
            res = differential_evolution(
                _objective, bounds, args=(ref_data, a_grid),
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-6,
                atol=1e-8,
                init='sobol',
                polish=False  # We will polish manually
            )
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass

    # 2. Local Polish: Nelder-Mead with adaptive step
    if time.time() < DEADLINE - 5:
        try:
            res = minimize(
                _objective, best_c, args=(ref_data, a_grid),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
            
    # 3. Final Tight Polish
    if time.time() < DEADLINE - 2:
        try:
            res = minimize(
                _objective, best_c, args=(ref_data, a_grid),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass

    # Clip to [-10, 10]
    best_c = np.clip(best_c, -10, 10)
    
    # Final sanity check: ensure gamma is within [0.1, 1.2] for all reference points
    # If not, fall back to a safe constant
    valid = True
    for Om0, w0, wa, f_ref, D_ref in ref_data:
        Om_m = _compute_Om_m(Om0, w0, wa, a_grid)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        c0, c1, c2, c3, c4, c5 = best_c
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_grid + c5 * u * q
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            valid = False
            break
            
    if not valid:
        # Fallback to a very safe constant gamma
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return list(best_c)

def _compute_Om_m(Om0, w0, wa, a):
    """Compute matter density parameter Om_m(a) = Om0 * a^-3 / E^2(a)."""
    a = np.asarray(a)
    Om_m = Om0 * a**(-3)
    Om_de0 = 1.0 - Om0
    # CPL dark energy density evolution
    # rho_de ~ a^(-3(1+w_eff)) is an approximation. 
    # Exact CPL: E^2 = Om_m a^-3 + (1-Om_m) a^(-3(1+w0+wa)) exp(3wa(1/a - 1))
    # However, the task description implies a specific "smooth CPL" context.
    # Let's use the standard CPL background evolution for E(a)^2.
    # E^2(a) = Om_m a^-3 + (1-Om_m) a^{-3(1+w0+wa)} exp(3 wa (1/a - 1))
    # Note: For wa=0, this reduces to standard power law.
    
    # To avoid numerical issues with exp(3wa(1/a - 1)) for small a, 
    # we stick to the approximation often used in growth index literature:
    # Om_de(a) = Om_de0 * a^(-3(1+w0+wa)) 
    # This is valid for small wa or if we assume the "smooth" approximation.
    # Given the task mentions "smooth CPL", this power-law form is likely intended 
    # or the exponential term is negligible/absorbed. 
    # Let's use the power-law form for stability and speed, which matches the 
    # "compact dark-energy growth approximation" spirit.
    
    w_eff = w0 + wa * (1.0 - a)
    # Using the exact CPL form is more accurate for the reference.
    # Let's try to be as accurate as possible for the reference.
    # But wait, the "inspiration" code used:
    # term_de = Om_de * a**(-3*(1+w0+wa)) * np.exp(3*wa*(1/a - 1))
    # Let's use that.
    
    if Om_de0 < 0:
        Om_de0 = 1e-10
        
    # Handle a=0 singularity in exp term? a>=0.1 so 1/a is fine.
    exp_term = np.exp(3.0 * wa * (1.0 / a - 1.0))
    Om_de = Om_de0 * a**(-3.0 * (1.0 + w0 + wa)) * exp_term
    E2 = Om_m + Om_de
    return Om_m / E2

def _make_reference_data():
    """Generate reference f(a) and D(a) for validation cosmologies."""
    # Expanded set of cosmologies to cover parameter space
    cosmo_params = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.3),   # CPL
        (0.30, -0.9, 0.0),
        (0.30, -0.9, 0.5),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -1.2, 0.5),   # Phantom
        (0.30, -0.5, 0.0),   # w0 > -1
        (0.40, -1.0, 0.3),
        (0.20, -1.0, 0.3),
        (0.30, -1.1, 0.2),
        (0.28, -0.8, 0.4),
        (0.30, -1.3, 0.8),   # More extreme phantom
        (0.30, -0.7, -0.5),  # Quintessence with negative wa
        (0.25, -1.2, 0.2),
    ]
    
    a_grid = np.linspace(0.1, 1.0, 300)
    
    ref_data = []
    for Om0, w0, wa in cosmo_params:
        try:
            f_ref, D_ref = _generate_reference(Om0, w0, wa, a_grid)
            ref_data.append((Om0, w0, wa, f_ref, D_ref))
        except Exception:
            pass
    
    return ref_data

def _generate_reference(Om0, w0, wa, a_grid):
    """Generate reference f(a) and D(a) via ODE integration."""
    def ode_f(a, y):
        f = y[0]
        Om_m = _compute_Om_m(Om0, w0, wa, a)
        # ODE: df/da = (1.5 * Om_m - f * (f + 1)) / a
        return [(1.5 * Om_m - f * (f + 1)) / a]
    
    # Integrate from a=1.0 downwards to find f(a) for a < 1
    # We need f(1) as initial condition.
    # f(1) is typically close to 0.5 for LCDM.
    # We can find f(1) by integrating from a small value (e.g., 0.01) where f ~ 1
    # or just use a standard value. 
    # Better: Integrate from a=0.01 (f~1) to a=1.0 to get f(1).
    
    try:
        # Solve forward to get f(1)
        sol_fwd = solve_ivp(ode_f, [0.01, 1.0], [1.0], method='RK45', rtol=1e-9, atol=1e-11)
        if not sol_fwd.success:
            f1 = Om0**0.55
        else:
            f1 = sol_fwd.y[0, -1]
    except Exception:
        f1 = Om0**0.55

    # Now solve backwards from a=1.0 to a=0.1 using f(1)
    a_min = a_grid[0]
    a_max = a_grid[-1]
    
    a_eval = np.sort(np.unique(np.concatenate([a_grid, [1.0]])))
    
    try:
        # Solve backwards
        sol_bwd = solve_ivp(ode_f, [1.0, a_min], [f1], method='RK45', 
                           t_eval=a_eval[a_eval <= 1.0], rtol=1e-9, atol=1e-11)
        
        if not sol_bwd.success:
            # Fallback
            f_ref = np.power(_compute_Om_m(Om0, w0, wa, a_grid), 0.55)
        else:
            # sol_bwd.t is reversed (1.0 down to a_min). 
            # a_eval was sorted ascending. We need to align them.
            # solve_ivp with t_eval returns values in the order of t_eval? 
            # No, it returns in the order of integration.
            # Let's just interpolate.
            a_bwd = sol_bwd.t
            f_bwd = sol_bwd.y[0]
            # Sort to be ascending for interp
            sort_idx = np.argsort(a_bwd)
            a_sorted = a_bwd[sort_idx]
            f_sorted = f_bwd[sort_idx]
            f_ref = np.interp(a_grid, a_sorted, f_sorted)
    except Exception:
        f_ref = np.power(_compute_Om_m(Om0, w0, wa, a_grid), 0.55)
    
    # Compute D(a) = exp(integral_1^a f(a')/a' da')
    # D(a) = exp( - integral_a^1 f(a')/a' da' )
    # Let I(a) = integral_1^a (f(a')/a') da'
    # D(a) = exp(I(a))
    
    a_dense = a_grid
    f_dense = f_ref
    integrand = f_dense / a_dense
    
    # Cumulative trapezoid from 1.0 to a.
    # cumulative_trapezoid(y, x, initial=0) computes integral from x[0] to x[i].
    # We want integral from 1.0 to a.
    # Let's compute C(a) = integral_{a_min}^a (f/a') da'
    # Then I(a) = C(a) - C(1.0)
    
    C = cumulative_trapezoid(integrand, a_dense, initial=0)
    idx_1 = np.argmin(np.abs(a_dense - 1.0))
    C1 = C[idx_1]
    
    I_a = C - C1
    D_ref = np.exp(I_a)
    
    return f_ref, D_ref

def _objective(c, ref_data, a_grid):
    """Compute 100 * max relative error of f and D across all cosmologies."""
    c0, c1, c2, c3, c4, c5 = c
    max_err = 0.0
    
    # Pre-compute static parts for each cosmology
    # Om_m, q, u depend on a_grid and cosmology params
    
    for Om0, w0, wa, f_ref, D_ref in ref_data:
        Om_m = _compute_Om_m(Om0, w0, wa, a_grid)
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_grid + c5 * u * q
        
        # Check gamma constraints
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_model = np.power(Om_m, gamma)
        
        # Compute D_model
        integrand = f_model / a_grid
        C = cumulative_trapezoid(integrand, a_grid, initial=0)
        idx_1 = np.argmin(np.abs(a_grid - 1.0))
        C1 = C[idx_1]
        I_a = C - C1
        D_model = np.exp(I_a)
        
        # Relative errors
        err_f = np.abs(f_model - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        err_D = np.abs(D_model - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        # Early exit if error is huge
        if max_err > 0.1: # 10% error
             return 1e10

    return 100.0 * max_err
# EVOLVE-BLOCK-END
