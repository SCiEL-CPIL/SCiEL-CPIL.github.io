"""Improved growth-index approximation with optimized coefficients."""
import numpy as np
from scipy.optimize import minimize, dual_annealing
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    a = np.asarray(a)
    Omega_DE0 = 1.0 - Om0
    # Exact CPL density evolution:
    # rho_DE(a) = rho_DE(1) * a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    # We need to handle a=0 carefully, but our integration range is a>0.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = -3.0 * (1.0 + w0 + wa) * np.log(a) + 3.0 * wa * (a - 1.0)
        ratio_DE = np.exp(log_ratio)
        # Handle a=0 limit if necessary, but we integrate from small a
        if a == 0:
             ratio_DE = 0 if (1+w0+wa) > 0 else np.inf
             
        Om_a = Om0 * a**(-3) / (Om0 * a**(-3) + Omega_DE0 * ratio_DE)
    return Om_a

def _compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    # Clamp to valid range [0.1, 1.2]
    gamma = np.clip(gamma, 0.1, 1.2)
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, coeffs, Om0, w0, wa)
    # Om_a is always > 0 for a>0
    return Om_a ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    ln_a = np.log(a)
    
    # Use fixed quadrature points for speed and consistency
    # We integrate f(exp(x)) from ln_a to 0
    # Since a is a scalar here, we can use a simple quad or pre-computed grid
    # For speed in optimization, let's use a fixed grid integration
    # x goes from ln_a to 0
    # Let's use 100 points for integration
    x_grid = np.linspace(ln_a, 0.0, 100)
    a_grid = np.exp(x_grid)
    f_grid = _f_of_a(a_grid, coeffs, Om0, w0, wa)
    integral_val = np.trapz(f_grid, x_grid)
    return np.exp(-integral_val)

def _solve_reference_growth(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=200):
    """
    Solve the growth ODE to get reference f(a) and D(a).
    ODE: D'' + (3/a - 0.5 * d ln H^2 / d ln a) D' - 1.5 * Om(a) D = 0
    """
    a_grid = np.linspace(a_min, a_max, n_points)
    
    # Pre-compute Om(a) and q(a)
    Om_grid = _Om_of_a(a_grid, Om0, w0, wa)
    
    # Calculate H^2(a) = Om(a) * a^-3 (for flat universe, H^2 = H0^2 * sum Omega_i a^-3(1+w_i))
    # Actually H^2/H0^2 = Om0 a^-3 + Omega_DE0 a^-3(1+w_eff) ... 
    # Let's use the definition: H^2 = H0^2 (Om0 a^-3 + rho_DE(a)/rho_crit0)
    # rho_DE(a)/rho_crit0 = (1-Om0) * a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    Omega_DE0 = 1.0 - Om0
    with np.errstate(divide='ignore', invalid='ignore'):
        log_rho_DE = -3.0 * (1.0 + w0 + wa) * np.log(a_grid) + 3.0 * wa * (a_grid - 1.0)
        rho_DE_rel = Omega_DE0 * np.exp(log_rho_DE)
        H2_grid = Om0 * a_grid**(-3) + rho_DE_rel
    
    # Calculate q(a) = 0.5 * d ln H^2 / d ln a = 0.5 * a * dH2/da / H2
    # Use numerical differentiation
    dH2_da = np.gradient(H2_grid, a_grid)
    with np.errstate(divide='ignore', invalid='ignore'):
        q_grid = 0.5 * a_grid * dH2_da / H2_grid
    
    # Define ODE system
    # y = [D, D']
    # D' = v
    # v' = -(3/a - q(a))v + 1.5 * Om(a) * D
    def ode_func(a, y):
        D, v = y
        # Interpolate q and Om at current a
        # Since we are on the grid, we can use indices, but solve_ivp steps continuously.
        # It's better to use interpolation or define the function analytically if possible.
        # For simplicity and speed, let's use linear interpolation on the pre-computed grids.
        # However, solve_ivp evaluates at arbitrary points.
        # Let's create interpolation functions.
        q_interp = np.interp(a, a_grid, q_grid)
        Om_interp = np.interp(a, a_grid, Om_grid)
        
        D_prime = v
        v_prime = -(3.0/a - q_interp) * v + 1.5 * Om_interp * D
        return [D_prime, v_prime]
    
    # Initial conditions at a_min
    # In matter domination, D ~ a, so D(a_min) = a_min, D'(a_min) = 1
    y0 = [a_min, 1.0]
    
    # Solve ODE
    try:
        sol = solve_ivp(ode_func, [a_min, a_max], y0, method='RK45', max_step=(a_max-a_min)/1000, rtol=1e-8, atol=1e-10)
        if not sol.success:
            # Fallback to simpler integration if solve_ivp fails
            # Use simple RK4 on the grid
            D_vals = np.zeros_like(a_grid)
            v_vals = np.zeros_like(a_grid)
            D_vals[0] = a_min
            v_vals[0] = 1.0
            dt = a_grid[1] - a_grid[0]
            for i in range(len(a_grid)-1):
                a_i = a_grid[i]
                D_i = D_vals[i]
                v_i = v_vals[i]
                q_i = q_grid[i]
                Om_i = Om_grid[i]
                
                def rhs(D, v):
                    return v, -(3.0/a_i - q_i)*v + 1.5*Om_i*D
                
                k1_D, k1_v = rhs(D_i, v_i)
                k2_D, k2_v = rhs(D_i + 0.5*dt*k1_D, v_i + 0.5*dt*k1_v)
                k3_D, k3_v = rhs(D_i + 0.5*dt*k2_D, v_i + 0.5*dt*k2_v)
                k4_D, k4_v = rhs(D_i + dt*k3_D, v_i + dt*k3_v)
                
                D_vals[i+1] = D_i + dt/6.0*(k1_D + 2*k2_D + 2*k3_D + k4_D)
                v_vals[i+1] = v_i + dt/6.0*(k1_v + 2*k2_v + 2*k3_v + k4_v)
            
            sol_y = np.array([D_vals, v_vals])
            sol_t = a_grid
        else:
            sol_y = sol.y
            sol_t = sol.t
            
    except Exception:
        return None, None

    # Normalize D such that D(1) = 1
    # Find D at a=1 (last point)
    D_at_1 = sol_y[0, -1]
    D_norm = sol_y[0] / D_at_1
    v_norm = sol_y[1] / D_at_1
    
    # Calculate f(a) = d ln D / d ln a = a * D' / D
    with np.errstate(divide='ignore', invalid='ignore'):
        f_ref = sol_t * v_norm / D_norm
        
    # Interpolate to a standard grid for easy lookup
    a_standard = np.linspace(0.1, 1.0, 90) # 0.1 to 1.0 step 0.01
    
    # Ensure f and D are positive and valid
    f_ref_interp = np.interp(a_standard, sol_t, f_ref)
    D_ref_interp = np.interp(a_standard, sol_t, D_norm)
    
    # Clean up any non-finite values
    f_ref_interp[~np.isfinite(f_ref_interp)] = 0.55
    D_ref_interp[~np.isfinite(D_ref_interp)] = 1.0
    
    return a_standard, f_ref_interp, D_ref_interp

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline to be safe
    
    # Representative cosmologies
    # (Om0, w0, wa)
    test_models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.1),   # CPL
        (0.30, -0.9, 0.0),   # w0 != -1
        (0.30, -1.1, 0.0),   # Phantom
        (0.30, -1.0, 0.3),   # Strong CPL
        (0.27, -1.0, 0.0),   # Different Om
        (0.32, -1.0, 0.0),   # Different Om
        (0.30, -0.8, -0.2),  # Mixed
        (0.30, -1.2, 0.5),   # Strong CPL
        (0.30, -0.95, 0.1),  # Near LCDM
    ]
    
    a_standard = np.linspace(0.1, 1.0, 90)
    
    # Pre-compute references
    references = []
    for Om0, w0, wa in test_models:
        a_ref, f_ref, D_ref = _solve_reference_growth(Om0, w0, wa)
        if a_ref is not None:
            references.append((Om0, w0, wa, a_ref, f_ref, D_ref))
            
    if not references:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    def objective(coeffs):
        """Smooth max relative error."""
        max_err = 0.0
        # Use a smooth max approximation: log-sum-exp
        # err_smooth = log(sum(exp(err_i / eps))) * eps
        # But we want min max, so we can just use a high power mean or log-sum-exp
        # Let's use a large epsilon for log-sum-exp to approximate max
        eps = 0.01
        sum_exp = 0.0
        count = 0
        
        for Om0, w0, wa, a_ref, f_ref, D_ref in references:
            # Compute approximations
            f_approx = _f_of_a(a_ref, coeffs, Om0, w0, wa)
            D_approx = _D_normalized(a_ref, coeffs, Om0, w0, wa)
            
            # Relative errors
            err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
            err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
            
            # Combine errors
            all_errs = np.concatenate([err_f, err_D])
            all_errs = all_errs[np.isfinite(all_errs)]
            
            if len(all_errs) > 0:
                # Log-sum-exp for smooth max
                max_e = np.max(all_errs)
                if max_e > 10: # Avoid overflow
                     sum_exp += np.sum(np.exp((all_errs - max_e) / eps))
                     smooth_max = max_e + eps * np.log(sum_exp)
                else:
                     sum_exp += np.sum(np.exp(all_errs / eps))
                     smooth_max = eps * np.log(sum_exp)
                
                if smooth_max > max_err:
                    max_err = smooth_max
                count += len(all_errs)
                
        if count == 0:
            return 1e6
        return max_err

    # Initial guess: constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_score = objective(x0)
    
    # Global Optimization: Dual Annealing
    # Bounds: [-10, 10]
    bounds = [(-10.0, 10.0)] * 6
    
    if time.time() < deadline - 10:
        try:
            # Use dual_annealing for global search
            # maxiter controls the number of annealing steps
            # It might be slow, so we limit it
            res_global = dual_annealing(
                objective, 
                bounds, 
                maxiter=100, 
                seed=42, 
                restarts=20,
                verbose=0
            )
            
            if res_global.fun < best_score:
                best_score = res_global.fun
                best_coeffs = res_global.x
                
        except Exception:
            pass

    # Local Optimization: L-BFGS-B
    # Use the best point from global search or x0
    x_start = best_coeffs.copy()
    
    if time.time() < deadline - 5:
        try:
            # L-BFGS-B with tight tolerances
            res_local = minimize(
                objective, 
                x_start, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_coeffs = res_local.x
                
        except Exception:
            pass
            
    # Final Polish: Nelder-Mead from best point
    if time.time() < deadline - 2:
        try:
            res_nm = minimize(
                objective, 
                best_coeffs, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_coeffs = res_nm.x
                
        except Exception:
            pass
            
    # Clip to bounds
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    return best_coeffs.tolist()

# EVOLVE-BLOCK-END
