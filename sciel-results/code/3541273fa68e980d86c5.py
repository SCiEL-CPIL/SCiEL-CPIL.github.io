"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def _compute_Om(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    a_inv3 = a ** (-3)
    a_inv3_cpl = a ** (-3 * (1 + w0 + wa))
    denom = Om0 * a_inv3 + (1 - Om0) * a_inv3_cpl
    return Om0 * a_inv3 / denom

def _compute_Hprime_over_H(a, Om0, w0, wa):
    """Compute H'/H = (1/2) d ln(H^2/H0^2) / d ln a."""
    a_inv3 = a ** (-3)
    a_inv3_cpl = a ** (-3 * (1 + w0 + wa))
    A = Om0 * a_inv3
    B = (1 - Om0) * a_inv3_cpl
    dlnH2_dlna = -3 * (A + (1 + w0 + wa) * B) / (A + B)
    return 0.5 * dlnH2_dlna

def _solve_reference(Om0, w0, wa, a_grid, rtol=1e-8, atol=1e-10):
    """
    Solve the growth ODE for a given cosmology and return f_ref and D_ref on a_grid.
    Returns (f_ref_array, D_ref_array) or None if failed.
    """
    # Create a fine grid for integration
    x_grid = np.log(a_grid)
    # Sort in decreasing order (from x=0 to negative values)
    sort_idx = np.argsort(x_grid)
    x_sorted = x_grid[sort_idx]
    
    # We integrate from x=0 to each x value
    # Use the growing mode solution: D(0)=1, D'(0) = -f(1) where f(1) ~ 0.5*Om0^(3/5)
    # Actually, for the reference, we need the actual GR solution
    # The growing mode has D ~ a at early times, so D' ~ 1 at large x (small a)
    # At x=0 (a=1), we need to match the condition that the solution is the growing mode
    
    # Let's use the fact that for the growing mode, at early times D ~ a
    # So we can integrate backwards from a small a where D ~ a, D' ~ 1
    
    # Start from a small value where matter domination holds
    a_start = min(a_grid) * 0.5
    x_start = np.log(a_start)
    
    # At a_start, D ~ a_start, D' ~ 1 (in terms of x = ln a)
    # Actually D ~ a means D = a * C, so D' = dD/dx = C = D/a
    # So D' = D/a at early times
    
    D_start = a_start  # Normalize so D ~ a
    Dprime_start = 1.0  # dD/dx = 1 when D ~ a
    
    def growth_ode(x, y):
        a_x = np.exp(x)
        if a_x <= 0:
            return [0, 0]
        Om_x = _compute_Om(a_x, Om0, w0, wa)
        Hprime_over_H = _compute_Hprime_over_H(a_x, Om0, w0, wa)
        
        D, Dprime = y
        if x == 0:
            # Avoid singularity at x=0
            Ddoubleprime = -(Hprime_over_H) * Dprime + (3/2) * Om_x * D
        else:
            Ddoubleprime = -(3/x + Hprime_over_H) * Dprime + (3/2) * Om_x * D
        return [Dprime, Ddoubleprime]
    
    try:
        # Integrate from x_start to 0
        sol = solve_ivp(growth_ode, [x_start, 0], [D_start, Dprime_start], 
                       method='RK45', rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            return None
        
        # Get D and D' at x=0 to normalize
        D_at_0 = sol.y[0, -1]
        Dprime_at_0 = sol.y[1, -1]
        
        # Now integrate from 0 to each x in x_grid
        # Actually, let's just use the dense output and interpolate
        # But we need values at specific x_grid points
        
        # Let's re-integrate from 0 to min(x_grid) using the normalized solution
        # First, normalize so D(0) = 1
        D_norm = D_at_0
        Dprime_norm = Dprime_at_0
        
        # The growing mode condition: at early times, D ~ a
        # We need to find the correct initial condition at x=0
        # such that the solution matches D ~ a at early times
        
        # Alternative: integrate from x_start to 0, then from 0 to x_grid
        # Use the solution at x=0 as initial condition for forward integration
        
        # Let's integrate from 0 to min(x_grid)
        x_end = min(x_grid)
        
        def growth_ode_from_zero(x, y):
            a_x = np.exp(x)
            if a_x <= 0:
                return [0, 0]
            Om_x = _compute_Om(a_x, Om0, w0, wa)
            Hprime_over_H = _compute_Hprime_over_H(a_x, Om0, w0, wa)
            
            D, Dprime = y
            if abs(x) < 1e-10:
                Ddoubleprime = -(Hprime_over_H) * Dprime + (3/2) * Om_x * D
            else:
                Ddoubleprime = -(3/x + Hprime_over_H) * Dprime + (3/2) * Om_x * D
            return [Dprime, Ddoubleprime]
        
        # We need the correct initial condition at x=0
        # From the backward integration, we have D(0) and D'(0)
        # Normalize D(0) = 1
        D0 = D_at_0
        Dprime0 = Dprime_at_0
        
        # Integrate from 0 to x_end
        sol2 = solve_ivp(growth_ode_from_zero, [0, x_end], [1.0, Dprime0/D0], 
                        method='RK45', rtol=rtol, atol=atol, dense_output=True)
        if not sol2.success:
            return None
        
        # Evaluate at x_grid points
        D_ref = np.zeros_like(a_grid)
        Dprime_ref = np.zeros_like(a_grid)
        f_ref = np.zeros_like(a_grid)
        
        for i, x in enumerate(x_grid):
            if x >= 0:
                D_ref[i] = 1.0
                Dprime_ref[i] = Dprime0 / D0
            else:
                # Interpolate from sol2
                D_val = sol2.sol(x)[0]
                Dprime_val = sol2.sol(x)[1]
                D_ref[i] = D_val
                Dprime_ref[i] = Dprime_val
            f_ref[i] = -Dprime_ref[i] / D_ref[i]
        
        return f_ref, D_ref
    except:
        return None

def _compute_error(coeffs, cosmologies, a_grid, deadline, reference_cache=None):
    """Compute max relative error for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    if reference_cache is None:
        reference_cache = {}
    
    max_err = 0.0
    for cosmo in cosmologies:
        Om0, w0, wa = cosmo
        cosmo_key = (Om0, w0, wa)
        
        if cosmo_key not in reference_cache:
            ref = _solve_reference(Om0, w0, wa, a_grid)
            if ref is None:
                continue
            reference_cache[cosmo_key] = ref
        
        f_ref, D_ref = reference_cache[cosmo_key]
        
        for i, a in enumerate(a_grid):
            if a <= 0 or a >= 1:
                continue
            
            Om_a = _compute_Om(a, Om0, w0, wa)
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            if gamma < 0.1 or gamma > 1.2:
                return float('inf')
            
            if Om_a <= 0:
                return float('inf')
            f_approx = Om_a ** gamma
            
            # Compute D_ratio_approx = exp(-integral of f_approx from ln(a) to 0)
            # Use numerical integration
            x0 = np.log(a)
            
            def integrand(x):
                a_x = np.exp(x)
                Om_x = _compute_Om(a_x, Om0, w0, wa)
                q_x = 1 - Om_x
                u_x = 1 + w0 + wa * (1 - a_x)
                gamma_x = c0 + c1 * q_x + c2 * q_x**2 + c3 * u_x + c4 * wa * a_x + c5 * u_x * q_x
                if gamma_x < 0.1 or gamma_x > 1.2:
                    return float('inf')
                if Om_x <= 0:
                    return float('inf')
                return Om_x ** gamma_x
            
            try:
                result = solve_ivp(lambda x, y: [-integrand(x)], [x0, 0], [0.0], 
                                  method='RK45', rtol=1e-8, atol=1e-10)
                if not result.success:
                    return float('inf')
                integral_val = -result.y[0, -1]
                D_ratio_approx = np.exp(-integral_val)
            except:
                return float('inf')
            
            # Compute relative errors
            f_ref_i = f_ref[i]
            D_ref_i = D_ref[i]
            
            if abs(f_ref_i) > 1e-10:
                err_f = abs(f_approx - f_ref_i) / abs(f_ref_i)
            else:
                err_f = abs(f_approx - f_ref_i)
            
            if abs(D_ref_i) > 1e-10:
                err_D = abs(D_ratio_approx - D_ref_i) / abs(D_ref_i)
            else:
                err_D = abs(D_ratio_approx - D_ref_i)
            
            max_err = max(max_err, err_f, err_D)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return max_err

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s safety margin
    
    # Validation cosmologies (typical values)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.3),    # CPL with wa=0.3
        (0.3, -1.0, -0.5),   # CPL with wa=-0.5
        (0.25, -1.0, 0.0),   # Lower Om0
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.3, -0.9, 0.1),    # Different w0
        (0.3, -1.1, -0.2),   # Different w0, wa
    ]
    
    # a grid
    a_grid = np.linspace(0.1, 0.99, 20)
    
    # Precompute reference solutions
    reference_cache = {}
    for cosmo in cosmologies:
        if time.time() > deadline:
            break
        Om0, w0, wa = cosmo
        cosmo_key = (Om0, w0, wa)
        ref = _solve_reference(Om0, w0, wa, a_grid)
        if ref is not None:
            reference_cache[cosmo_key] = ref
    
    # Physically motivated initial guess: c0=0.55, c3=-0.5 to cancel leading w0 term
    best_coeffs = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_err = _compute_error(best_coeffs, cosmologies, a_grid, deadline, reference_cache)
    
    # Optimization
    def objective(coeffs):
        if time.time() > deadline:
            return best_err
        err = _compute_error(coeffs, cosmologies, a_grid, deadline, reference_cache)
        return err
    
    try:
        # First pass: Nelder-Mead with moderate tolerance
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                         options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
    except:
        pass
    
    # Second pass: Polish with tighter tolerance if time permits
    if time.time() < deadline - 10:
        try:
            result2 = minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if result2.fun < best_err:
                best_coeffs = result2.x
                best_err = result2.fun
        except:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
