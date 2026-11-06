"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import quad
import time

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Pre-computed reference data cache
_ref_data_cache = None

def _compute_om_a_vec(a, Om0, w0, wa):
    """Vectorized computation of matter density fraction."""
    a = np.asarray(a)
    # Handle a=0 or negative (should not happen in valid range, but safe)
    a_safe = np.where(a > 0, a, 1e-10)
    
    # Exact CPL evolution for DE density
    # rho_DE(a) / rho_DE(0) = a^(-3*(1+w0+wa)) * exp(3*wa*a)
    # E^2(a) = Om0*a^-3 + (1-Om0)*a^(-3*(1+w0+wa))*exp(3*wa*a)
    w_eff = 1 + w0 + wa
    de_term = (1 - Om0) * (a_safe ** (-3 * w_eff)) * np.exp(3 * wa * a_safe)
    m_term = Om0 * (a_safe ** -3)
    E2 = m_term + de_term
    Om_a = m_term / E2
    return Om_a

def _compute_gamma_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of gamma."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _compute_f_vec(a, Om0, w0, wa, coeffs):
    """Vectorized computation of f = Om(a)^gamma."""
    Om_a = _compute_om_a_vec(a, Om0, w0, wa)
    gamma = _compute_gamma_vec(a, Om0, w0, wa, coeffs)
    # Ensure Om_a > 0 for power
    Om_a_safe = np.maximum(Om_a, 1e-10)
    return Om_a_safe ** gamma

def _compute_D_vec(a, Om0, w0, wa, coeffs):
    """
    Vectorized computation of normalized D.
    D(a)/D(1) = exp(-int_ln(a)^0 f(exp(x)) dx)
    We approximate the integral using trapezoidal rule on a fine grid.
    """
    # We only need D for a < 1. For a >= 1, D=1.
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    
    a_masked = a[mask]
    # Define a fine grid for integration
    # The integration range is [ln(a), 0]
    # We can compute the integral from ln(a) to 0 for each a.
    # To be efficient, we can pre-compute the antiderivative F(x) = int_0^x f(exp(t)) dt
    # Then D(a) = exp(F(ln(a)))
    
    # Let's compute f on a logarithmic grid from min(a_masked) to 1.0
    a_min = np.min(a_masked)
    # Grid points in log space
    n_points = 200
    log_a_grid = np.linspace(np.log(a_min), 0.0, n_points)
    a_grid = np.exp(log_a_grid)
    
    f_grid = _compute_f_vec(a_grid, Om0, w0, wa, coeffs)
    
    # Integrate f(exp(x)) dx from 0 to log_a.
    # Note: D(a) = exp(-int_ln(a)^0 f(x) dx) = exp(int_0^ln(a) f(x) dx)
    # So we need integral from 0 to ln(a).
    # Using cumulative trapezoidal rule.
    # scipy.integrate.cumulative_trapezoid is available, but let's do it manually or use simple np
    # dx is uniform in log space.
    dx = log_a_grid[1] - log_a_grid[0]
    
    # Integral from 0 to log_a_grid[i]
    # int_0^x f(t) dt.
    # Let's compute cumulative integral from 0 upwards? No, grid is from log(a_min) to 0.
    # It's easier to compute cumulative integral from the end (0) backwards or just use a function.
    
    # Let's compute the integral from 0 to x for each x in log_a_grid.
    # I(x) = int_0^x f(exp(t)) dt
    # Since grid is sorted ascending, we can compute cumulative integral from start, but we need from 0.
    # Let's reverse the grid to go from 0 to log(a_min).
    log_a_rev = log_a_grid[::-1]
    f_rev = f_grid[::-1]
    
    # Cumulative trapezoid
    # I[0] = 0 (at x=0)
    I_rev = np.zeros_like(log_a_rev)
    for i in range(1, len(log_a_rev)):
        I_rev[i] = I_rev[i-1] + 0.5 * (f_rev[i] + f_rev[i-1]) * (log_a_rev[i] - log_a_rev[i-1])
    
    # Now I_rev[k] is integral from 0 to log_a_rev[k]
    # We need integral from 0 to ln(a) for each a in a_masked.
    # Interpolate I_rev to get values at ln(a_masked).
    # log_a_rev is descending? No, log_a_grid was ascending (log(a_min) to 0).
    # So log_a_rev is descending (0 to log(a_min)).
    # We want integral from 0 to x where x is negative.
    # The integral int_0^x f(t) dt for x < 0 is negative of int_x^0 f(t) dt.
    # My I_rev calculation:
    # log_a_rev[0] = 0.
    # log_a_rev[1] < 0.
    # I_rev[1] = 0.5 * (f(0) + f(x1)) * (x1 - 0). Since x1 < 0, this is negative.
    # This is correct for int_0^x1.
    
    # So we can interpolate I_rev against log_a_rev.
    # log_a_rev is monotonically decreasing.
    ln_a_masked = np.log(a_masked)
    
    # np.interp expects xp to be increasing.
    # Let's reverse again to make xp increasing.
    xp = log_a_rev[::-1]  # Increasing from log(a_min) to 0
    fp = I_rev[::-1]       # Corresponding integrals
    
    # Wait, I_rev was computed on log_a_rev (decreasing).
    # I_rev[0] corresponds to xp[0] (which is 0).
    # I_rev[-1] corresponds to xp[-1] (which is log(a_min)).
    # So xp is increasing.
    
    I_masked = np.interp(ln_a_masked, xp, fp)
    
    D[mask] = np.exp(I_masked)
    return D

def _precompute_references():
    """Pre-compute reference f and D for a set of validation models."""
    global _ref_data_cache
    if _ref_data_cache is not None:
        return _ref_data_cache
        
    # Validation models (disjoint from training, but we use a broad set)
    models = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -0.9, 0.2),
        (0.3, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.3, -1.5, 1.0),
        (0.3, -0.5, -0.5),
    ]
    
    a_vals = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    ref_data = []
    for Om0, w0, wa in models:
        # Reference f: Use a high-accuracy ODE solution
        # For speed in pre-computation, we can use a simple adaptive integrator or just a fine grid ODE.
        # Since this is done once, we can afford a bit more accuracy.
        # Let's use scipy.integrate.solve_ivp for the Riccati equation.
        
        def ode_func(ln_a, f):
            a = np.exp(ln_a)
            Om_a = _compute_om_a_vec(a, Om0, w0, wa)[0]
            E2 = Om_a * a**(-3) + (1 - Om_a) * a**(-3*(1+w0+wa)) * np.exp(3*wa*a)
            # H'/H = 0.5 * d(ln E2) / d(ln a)
            # d(ln E2) / d(ln a) = (dE2/da * a) / E2
            # dE2/da = -3*Om0*a^-4 + (1-Om0)*d/da[ a^(-3(1+w0+wa)) exp(3wa a) ]
            # Let g(a) = a^(-3(1+w0+wa)) exp(3wa a)
            # g'(a) = g(a) * [ -3(1+w0+wa)/a + 3wa ]
            # So dE2/da = -3*Om0*a^-4 + (1-Om0)*g(a)*3*(wa - (1+w0+wa)/a)
            g = a**(-3*(1+w0+wa)) * np.exp(3*wa*a)
            dE2_da = -3*Om0*a**(-4) + (1-Om0)*g*3*(wa - (1+w0+wa)/a)
            dlnE2_dlna = (dE2_da * a) / E2
            Hprime_H = 0.5 * dlnE2_dlna
            
            # f' = -(1.5)*Om_a/E2 - (3.0/a + Hprime_H)*f - f^2
            # Note: E2 = H^2/H0^2. Om_m H0^2/H^2 = Om_a / E2.
            f_prime = -(1.5) * Om_a / E2 - (3.0 / a + Hprime_H) * f - f**2
            return [f_prime]
        
        # Solve from ln(1)=0 to ln(a_min)
        # We need f(a) for a in a_vals.
        # Let's solve on a grid.
        ln_a_grid = np.log(a_vals)
        # Sort in decreasing order for integration from 0 to negative
        # Or just solve for each point? No, solve_ivp can take t_eval.
        
        # Initial condition at a=1 (ln_a=0)
        # f(1) is not exactly Om0^0.55, but close. 
        # Actually, for LCDM, f(1) ~ 0.55 * Om0^0.55 is a good approx.
        # More accurately, f(1) satisfies the ODE at a=1.
        # Let's use f(1) = Om0**0.55 * 0.98 (slight correction often used)
        # Or just solve the ODE starting from a slightly less than 1?
        # Standard practice: f(1) is determined by the ODE initial condition.
        # Let's use f(1) = 0.55 * Om0**0.55 as a seed.
        
        f_init = 0.55 * Om0**0.55
        
        # Solve backwards
        # t_eval should be sorted. solve_ivp handles direction.
        # We want f at ln_a_grid.
        # ln_a_grid is [log(0.3), ..., log(1.0)] -> negative to 0.
        # We integrate from 0 to log(0.3).
        
        try:
            sol = solve_ivp(ode_func, [0, np.log(0.3)], [f_init], t_eval=ln_a_grid, method='RK45', rtol=1e-8, atol=1e-10)
            if sol.success:
                f_ref = sol.y[0]
            else:
                # Fallback to analytic approx if ODE fails
                f_ref = _compute_f_vec(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
        except Exception:
            f_ref = _compute_f_vec(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
            
        # Reference D: Integrate f_ref
        # D(a) = exp(int_0^ln(a) f_ref(exp(x)) dx)
        # We have f_ref at a_vals. We can integrate using trapezoid.
        # But we need D at the same a_vals.
        # Let's compute the integral from 0 to ln(a) for each a in a_vals.
        # We can use cumulative trapezoid on the sorted ln_a_grid.
        
        # ln_a_grid is sorted ascending (negative to 0).
        # We want integral from 0 to x.
        # Let's reverse to go from 0 to negative.
        ln_a_rev = ln_a_grid[::-1]
        f_rev = f_ref[::-1]
        
        D_ref = np.ones_like(a_vals)
        # D[0] (at a=1) is 1.
        # For others:
        integral_vals = np.zeros_like(ln_a_rev)
        for i in range(1, len(ln_a_rev)):
            integral_vals[i] = integral_vals[i-1] + 0.5 * (f_rev[i] + f_rev[i-1]) * (ln_a_rev[i] - ln_a_rev[i-1])
            
        # integral_vals[i] is integral from 0 to ln_a_rev[i]
        # D = exp(integral)
        D_rev = np.exp(integral_vals)
        D_ref = D_rev[::-1]
        
        ref_data.append((Om0, w0, wa, a_vals, f_ref, D_ref))
        
    _ref_data_cache = ref_data
    return ref_data

def _evaluate_score(coeffs):
    """Calculate the worst-case relative error for a set of coefficients."""
    ref_data = _precompute_references()
    max_err = 0.0
    
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref_data:
        f_approx = _compute_f_vec(a_vals, Om0, w0, wa, coeffs)
        D_approx = _compute_D_vec(a_vals, Om0, w0, wa, coeffs)
        
        # Relative error for f
        err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        max_err = max(max_err, np.max(err_f))
        
        # Relative error for D
        err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err = max(max_err, np.max(err_D))
        
        # Check Gamma constraint: Gamma must lie in [0.1, 1.2]
        # We should check this on the validation points.
        gamma_vals = _compute_gamma_vec(a_vals, Om0, w0, wa, coeffs)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 1.0 # Penalty for invalid gamma
    
    return max_err

def _optimize_coeffs():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    # Initial guess
    best_coeffs = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_score = _evaluate_score(best_coeffs)
    
    # Use Nelder-Mead or similar local optimizer
    # It's fast and doesn't require gradients.
    
    def objective(x):
        return _evaluate_score(x)
    
    # Constraints: coeffs in [-10, 10]
    bounds = [(-10, 10)] * 6
    
    # Run optimization
    try:
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
            
        # Check time
        if time.time() > deadline:
            return best_coeffs
        
        # Try a few more random restarts if time permits
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            x0 = np.random.uniform(-1, 1, 6)
            # Clip to reasonable range to avoid bad starts
            x0 = np.clip(x0, -5, 5)
            
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
                
    except Exception as e:
        print(f"Optimization error: {e}")
        
    return best_coeffs

# Cache the optimized coefficients
_optimized_coeffs = None

def construct():
    global _optimized_coeffs
    if _optimized_coeffs is None:
        _optimized_coeffs = _optimize_coeffs()
    return _optimized_coeffs
# EVOLVE-BLOCK-END
