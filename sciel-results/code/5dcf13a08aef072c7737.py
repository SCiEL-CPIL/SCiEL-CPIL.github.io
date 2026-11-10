"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

_start_time = time.time()
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    de_integrand = np.power(a, 3.0 * (1.0 + w0 + wa))
    # Integral of a^3 * (1+w(a)) da from 0 to a
    # For CPL: w(a) = w0 + wa*(1-a)
    # a^3*(1+w(a)) = a^3*(1+w0+wa) - wa*a^3
    # Integral = a^4*(1+w0+wa)/4 - wa*a^4/4 = a^4*(1+w0)/4
    # Wait, let me redo this carefully.
    # a^3 * (1 + w0 + wa*(1-a)) = a^3*(1+w0+wa) - wa*a^4
    # Integral from 0 to a: (1+w0+wa)*a^4/4 - wa*a^5/5
    # But we need the ratio to total density.
    # Omega_m(a) = Om0 * a^-3 / (Om0*a^-3 + Ode0 * exp(3*int_1^a (1+w(a'))/a' da'))
    # The DE density: rho_de(a) = rho_de(1) * exp(3 * int_1^a (1+w(a'))/a' da')
    # 1+w(a) = 1+w0+wa*(1-a)
    # int_1^a (1+w(a'))/a' da' = int_1^a (1+w0+wa)/a' da' - wa*int_1^a da'
    # = (1+w0+wa)*ln(a) - wa*(a-1)
    # So rho_de(a)/rho_de(1) = exp(3*((1+w0+wa)*ln(a) - wa*(a-1)))
    # = a^(3*(1+w0+wa)) * exp(-3*wa*(a-1))
    # = a^(3*(1+w0+wa)) * exp(3*wa*(1-a))
    
    a_safe = np.where(a > 1e-10, a, 1e-10)
    de_factor = np.power(a_safe, 3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (1.0 - a_safe))
    Om = Om0 / (Om0 / np.power(a_safe, 3.0) + (1.0 - Om0) * de_factor)
    return Om

def _compute_gamma(a, c, Om0, w0, wa):
    """Compute gamma from the 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_of_a(a, c, Om0, w0, wa):
    """Compute f = Om(a)^gamma."""
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, c, Om0, w0, wa)
    return np.power(np.maximum(Om, 1e-300), gamma)

def _D_normalized(a, c, Om0, w0, wa):
    """Compute normalized growth factor D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    # Change variable: x = ln(a'), a' = exp(x)
    # integral from ln(a) to 0 of f(exp(x)) dx
    # = integral from a to 1 of f(a') / a' da'
    a_arr = np.asarray(a, dtype=float)
    a_safe = np.where(a_arr > 1e-10, a_arr, 1e-10)
    # Use numerical integration for each a
    D = np.zeros_like(a_safe)
    for i, ai in enumerate(a_safe):
        if ai >= 1.0 - 1e-12:
            D[i] = 1.0
            continue
        # Integrate from ai to 1 of f(a')/a' da'
        npts = 200
        a_grid = np.linspace(ai, 1.0, npts)
        f_vals = _f_of_a(a_grid, c, Om0, w0, wa) / a_grid
        integral = np.trapz(f_vals, a_grid)
        D[i] = np.exp(-integral)
    return D

def _ode_reference_f(a, Om0, w0, wa):
    """Solve the GR ODE for f = d ln D / d ln a."""
    # ODE: f' + (3/2 + 1/2*Omega_m(a)) * f + (3/2) * Omega_m(a) = 0
    # where ' is d/d ln a
    a_arr = np.asarray(a, dtype=float)
    a_safe = np.where(a_arr > 1e-10, a_arr, 1e-10)
    
    # Solve ODE numerically
    npts = max(len(a_safe), 100)
    a_grid = np.linspace(1e-4, 1.0, npts)
    
    def ode_rhs(ln_a, f):
        a_val = np.exp(ln_a)
        Om = _Om_of_a(a_val, Om0, w0, wa)
        return -(1.5 + 0.5 * Om) * f - 1.5 * Om
    
    from scipy.integrate import solve_ivp
    sol = solve_ivp(ode_rhs, [0.0, np.log(1e-4)], [0.0], 
                    t_eval=np.log(a_grid), method='RK45', rtol=1e-10, atol=1e-12)
    
    # Interpolate to get f at a_safe
    f_ref = np.interp(np.log(a_safe), np.log(a_grid), sol.y[0])
    return f_ref

def _ode_reference_D(a, Om0, w0, wa):
    """Solve the ODE for D(a)/D(1)."""
    a_arr = np.asarray(a, dtype=float)
    a_safe = np.where(a_arr > 1e-10, a_arr, 1e-10)
    
    npts = max(len(a_safe), 100)
    a_grid = np.linspace(1e-4, 1.0, npts)
    
    def ode_rhs_D(ln_a, D):
        a_val = np.exp(ln_a)
        Om = _Om_of_a(a_val, Om0, w0, wa)
        f = -D / a_val * (1.5 + 0.5 * Om) - 1.5 * Om
        # Actually, D' = f * D, so D' = f*D
        # f = D'/D, so we need f from the ODE
        # Let's solve for f first, then integrate D
        return None  # Placeholder
    
    # Better approach: solve for f, then integrate D
    f_ref = _ode_reference_f(a_grid, Om0, w0, wa)
    
    # D(a)/D(1) = exp(-int_a^1 f(a')/a' da')
    D = np.zeros_like(a_safe)
    for i, ai in enumerate(a_safe):
        if ai >= 1.0 - 1e-12:
            D[i] = 1.0
            continue
        mask = (a_grid >= ai) & (a_grid <= 1.0)
        if np.sum(mask) < 2:
            D[i] = 1.0
            continue
        a_sub = a_grid[mask]
        f_sub = f_ref[mask]
        integrand = f_sub / a_sub
        integral = np.trapz(integrand, a_sub)
        D[i] = np.exp(-integral)
    return D

def _error_metric(c, Om0, w0, wa):
    """Compute the worst-case relative error for given coefficients."""
    if time.time() > _DEADLINE:
        return 1e6
    
    # Test at several a values
    a_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    
    # Check gamma constraint
    gamma_vals = _compute_gamma(a_test, c, Om0, w0, wa)
    if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
        return 1e6
    
    # Compute model values
    f_model = _f_of_a(a_test, c, Om0, w0, wa)
    D_model = _D_normalized(a_test, c, Om0, w0, wa)
    
    # Compute reference values
    f_ref = _ode_reference_f(a_test, Om0, w0, wa)
    D_ref = _ode_reference_D(a_test, Om0, w0, wa)
    
    # Relative errors
    f_err = np.abs(f_model - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
    D_err = np.abs(D_model - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
    
    return np.max(np.concatenate([f_err, D_err]))

def _optimize_for_model(Om0, w0, wa):
    """Optimize coefficients for a single cosmology."""
    if time.time() > _DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Use differential evolution for global optimization
    bounds = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2)]
    
    result = differential_evolution(_error_metric, bounds, args=(Om0, w0, wa),
                                     seed=42, maxiter=50, popsize=15, tol=1e-6,
                                     polish=True)
    return result.x

def construct():
    """Find optimal 6 coefficients for the growth index formula."""
    global _start_time
    if time.time() - _start_time > _DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Test several cosmologies
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.3),    # CPL with positive wa
        (0.3, -1.0, -0.3),   # CPL with negative wa
        (0.27, -0.9, 0.2),   # Different Om0, w0
        (0.3, -1.1, 0.1),    # Phantom-like
        (0.3, -0.95, -0.1),  # Mildly varying
    ]
    
    # Start with constant gamma seed
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err = 1e6
    
    # For each cosmology, optimize and track worst-case
    for Om0, w0, wa in cosmologies:
        if time.time() > _DEADLINE:
            break
        
        c_opt = _optimize_for_model(Om0, w0, wa)
        err = _error_metric(c_opt, Om0, w0, wa)
        
        # Track the coefficient set that minimizes worst-case across all models
        # Actually, we need a single set of coefficients that works well for ALL models
        # So let's do a joint optimization
    
    # Joint optimization: minimize max error across all cosmologies
    def _joint_error(c):
        if time.time() > _DEADLINE:
            return 1e6
        max_err = 0.0
        for Om0, w0, wa in cosmologies:
            err = _error_metric(c, Om0, w0, wa)
            max_err = max(max_err, err)
            if err > 1e5:
                return err
        return max_err
    
    # Use Nelder-Mead starting from the seed
    if time.time() < _DEADLINE - 5.0:
        result = minimize(_joint_error, best_c, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < best_err:
            best_c = result.x
            best_err = result.fun
    
    # Final check
    if time.time() > _DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Clamp to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    return list(best_c)
# EVOLVE-BLOCK-END
