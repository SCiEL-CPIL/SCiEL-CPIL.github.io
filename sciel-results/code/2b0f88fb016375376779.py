"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def _compute_Om(a, Om0, w0, wa):
    """Compute Om(a) for CPL dark energy."""
    a = np.asarray(a)
    a_inv3 = a ** (-3)
    a_inv3_cpl = a ** (-3 * (1 + w0 + wa))
    denom = Om0 * a_inv3 + (1 - Om0) * a_inv3_cpl
    return Om0 * a_inv3 / denom

def _compute_Hprime_over_H(a, Om0, w0, wa):
    """Compute H'/H = (1/2) d ln(H^2/H0^2) / d ln a."""
    a = np.asarray(a)
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
    # We integrate the ODE: D'' + (3/x + H'/H)D' - (3/2)Om(a)D = 0
    # Variable x = ln(a). a = exp(x).
    # D ~ a as a -> 0 (x -> -inf).
    # We integrate from x_start (small a) to x=0 (a=1).
    
    x_grid = np.log(a_grid)
    x_min = np.min(x_grid)
    
    # Start from very early times where matter dominates
    x_start = x_min - 2.0 
    if x_start < -10: x_start = -10
    
    def growth_ode(x, y):
        a_x = np.exp(x)
        Om_x = _compute_Om(a_x, Om0, w0, wa)
        Hprime_over_H = _compute_Hprime_over_H(a_x, Om0, wa, w0) # Note: args order in helper is a, Om0, w0, wa
        
        # Fix arg order for helper call
        # Helper: _compute_Hprime_over_H(a, Om0, w0, wa)
        Hprime_over_H = _compute_Hprime_over_H(a_x, Om0, w0, wa)
        
        D, Dprime = y
        if abs(x) < 1e-10:
            Ddoubleprime = -(Hprime_over_H) * Dprime + (3/2) * Om_x * D
        else:
            Ddoubleprime = -(3/x + Hprime_over_H) * Dprime + (3/2) * Om_x * D
        return [Dprime, Ddoubleprime]
    
    try:
        # Initial conditions at x_start: D ~ a, D' ~ 1
        D_start = np.exp(x_start)
        Dprime_start = 1.0
        
        sol = solve_ivp(growth_ode, [x_start, 0], [D_start, Dprime_start], 
                       method='RK45', rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            return None
        
        D_at_0 = sol.y[0, -1]
        Dprime_at_0 = sol.y[1, -1]
        
        # Normalize D(0) = 1
        D_norm_factor = D_at_0
        Dprime_norm_factor = Dprime_at_0
        
        # Evaluate on grid
        f_ref = np.zeros_like(a_grid)
        D_ref = np.zeros_like(a_grid)
        
        for i, x in enumerate(x_grid):
            if x >= 0:
                D_val = 1.0
                Dprime_val = Dprime_norm_factor / D_norm_factor
            else:
                y_vals = sol.sol(x)
                D_val = y_vals[0] / D_norm_factor
                Dprime_val = y_vals[1] / D_norm_factor
            
            D_ref[i] = D_val
            f_ref[i] = -Dprime_val / D_val
        
        return f_ref, D_ref
    except:
        return None

def _compute_error_fast(coeffs, cosmologies, a_grid, reference_cache):
    """
    Compute max relative error using vectorized operations.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for cosmo in cosmologies:
        Om0, w0, wa = cosmo
        cosmo_key = (Om0, w0, wa)
        
        if cosmo_key not in reference_cache:
            continue
            
        f_ref, D_ref = reference_cache[cosmo_key]
        a_vals = a_grid
        
        # Compute Om(a) for all a
        Om_a = _compute_Om(a_vals, Om0, w0, wa)
        q = 1 - Om_a
        u = 1 + w0 + wa * (1 - a_vals)
        
        # Compute gamma
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
        
        # Check bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return float('inf')
        
        # Compute f_approx
        f_approx = Om_a ** gamma
        
        # Compute D_approx
        # D(a)/D(1) = exp(-integral_{ln a}^0 f(x) dx)
        # Let x = ln a. dx = da/a. 
        # Integral from x_a to 0 of f(x) dx.
        # We can approximate this using the trapezoidal rule on the grid.
        # Since a_grid is sorted, we can compute cumulative integrals.
        
        x_vals = np.log(a_vals)
        
        # We need integral from x_a to 0.
        # Let's compute cumulative integral from 0 down to x_min.
        # I(x) = integral_x^0 f(x') dx'
        # This is - integral_0^x f(x') dx'
        
        # Sort a_vals in descending order (x descending) to integrate from 0 to -inf
        # a_vals is typically linspace(0.1, 0.99) -> x is negative.
        # Let's just use np.cumsum on a fine subgrid or the grid itself if dense enough.
        # For speed in objective, we use the grid points provided.
        
        # f_approx is defined on a_vals.
        # We want J(a) = integral_{ln a}^0 f(x) dx.
        # Let's compute the integral from 0 to each x_i.
        # K(x_i) = integral_0^{x_i} f(x) dx.
        # J(a_i) = -K(x_i).
        
        # Use trapezoidal rule.
        # x_vals is sorted ascending (a_vals ascending).
        # We need integral from 0 to x_i (where x_i < 0).
        # This is - (integral from x_i to 0).
        
        # Let's compute cumulative integral from the smallest x to 0.
        # Let x_sorted = np.sort(x_vals)
        # We need to handle the fact that x_vals might not include 0.
        # We assume a_grid goes up to 0.99, so x_max < 0.
        
        # To be safe and fast, we interpolate f onto a fine grid? No, that's slow.
        # We use the existing grid.
        
        # Sort indices by x
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        f_sorted = f_approx[sort_idx]
        
        # We need integral from x_sorted[i] to 0.
        # Let's compute cumulative integral from x_sorted[0] to x_sorted[j].
        # dx = x_sorted[1:] - x_sorted[:-1]
        # integral = np.cumsum(0.5 * (f_sorted[1:] + f_sorted[:-1]) * dx)
        
        # But we need integral to 0.
        # Let's append x=0, f(0) = Om(0)^gamma. Om(0) is not 1.
        # Wait, a=1 is x=0. a_grid max is 0.99.
        # We can extrapolate or just integrate up to max(x).
        # The error in D is dominated by the integral.
        
        # Let's compute the integral from x_sorted[0] to 0.
        # We need f at x=0.
        a_zero = 1.0
        Om_zero = _compute_Om(a_zero, Om0, w0, wa)
        q_zero = 1 - Om_zero
        u_zero = 1 + w0 + wa * (1 - a_zero)
        gamma_zero = c0 + c1 * q_zero + c2 * q_zero**2 + c3 * u_zero + c4 * wa * a_zero + c5 * u_zero * q_zero
        f_zero = Om_zero ** gamma_zero
        
        # Extend grid with x=0
        x_ext = np.append(x_sorted, 0.0)
        f_ext = np.append(f_sorted, f_zero)
        
        # Trapezoidal cumulative integral
        dx = np.diff(x_ext)
        avg_f = 0.5 * (f_ext[:-1] + f_ext[1:])
        d_int = dx * avg_f
        cum_int = np.cumsum(d_int)
        
        # cum_int[j] is integral from x_ext[0] to x_ext[j+1].
        # We want integral from x_i to 0.
        # Total integral from x_ext[0] to 0 is cum_int[-1].
        # Integral from x_i to 0 = Total - Integral from x_ext[0] to x_i.
        
        # Index of x_i in x_ext is i+1 (since we appended 0 at end, and x_sorted was sorted asc).
        # Wait, x_sorted is ascending (most negative to least negative).
        # x_ext = [x_min, ..., x_max, 0]
        # cum_int[k] = int(x_ext[0], x_ext[k+1])
        
        # We want int(x_i, 0).
        # int(x_i, 0) = int(x_ext[0], 0) - int(x_ext[0], x_i)
        # int(x_ext[0], 0) = cum_int[-1]
        # int(x_ext[0], x_i) corresponds to the segment ending at x_i.
        
        # Let's map original indices to sorted indices.
        # inv_sort = np.argsort(sort_idx)
        
        D_approx = np.zeros_like(a_vals)
        
        # For each point in original grid:
        # Find its position in x_ext.
        # Since x_ext is sorted, we can use searchsorted or just map.
        # Actually, we can compute the integral for all sorted points and then unsort.
        
        # int(x_sorted[i], 0) = cum_int[-1] - cum_int[i]  (since cum_int[i] is int(x_ext[0], x_ext[i+1]) = int(x_sorted[i], x_sorted[i])? No.)
        
        # Let's be precise.
        # x_ext = [x_0, x_1, ..., x_N, 0]
        # cum_int[0] = int(x_0, x_1)
        # cum_int[k] = int(x_0, x_{k+1})
        
        # We want I(x_j) = int(x_j, 0) = int(x_0, 0) - int(x_0, x_j)
        # int(x_0, 0) = cum_int[-1]
        # int(x_0, x_j) is the sum of d_int[0] to d_int[j-1].
        # So int(x_0, x_j) = cum_int[j-1] for j>0.
        # For j=0, int(x_0, x_0) = 0.
        
        # So I(x_j) = cum_int[-1] - (0 if j==0 else cum_int[j-1])
        
        I_sorted = np.zeros_like(x_sorted)
        I_sorted[0] = cum_int[-1]
        I_sorted[1:] = cum_int[-1] - cum_int[:-1]
        
        # Unsort I_sorted back to original order
        I_orig = np.zeros_like(a_vals)
        I_orig[sort_idx] = I_sorted
        
        D_approx = np.exp(-I_orig)
        
        # Calculate errors
        err_f = np.abs(f_approx - f_ref) / np.abs(f_ref)
        err_D = np.abs(D_approx - D_ref) / np.abs(D_ref)
        
        current_max = np.max(np.concatenate([err_f, err_D]))
        if current_max > max_err:
            max_err = current_max
            
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    # Validation cosmologies
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -1.0, -0.5),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.3, -0.9, 0.1),
        (0.3, -1.1, -0.2),
    ]
    
    # a grid for optimization (sparse for speed)
    a_grid_opt = np.linspace(0.1, 0.99, 10)
    
    # Precompute references
    reference_cache = {}
    for cosmo in cosmologies:
        if time.time() > deadline:
            break
        Om0, w0, wa = cosmo
        ref = _solve_reference(Om0, w0, wa, a_grid_opt)
        if ref is not None:
            reference_cache[(Om0, w0, wa)] = ref
    
    # Initial guess
    best_coeffs = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_err = _compute_error_fast(best_coeffs, cosmologies, a_grid_opt, reference_cache)
    
    # Optimization
    def objective(coeffs):
        if time.time() > deadline:
            return best_err
        err = _compute_error_fast(coeffs, cosmologies, a_grid_opt, reference_cache)
        return err
    
    try:
        # Use Powell or Nelder-Mead
        # Powell is often faster for this type of problem
        result = minimize(objective, best_coeffs, method='Powell', 
                         options={'maxiter': 100, 'ftol': 1e-5})
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
    except:
        pass
    
    # Refine on a denser grid if time permits
    if time.time() < deadline - 10:
        a_grid_refine = np.linspace(0.1, 0.99, 50)
        reference_cache_refine = {}
        for cosmo in cosmologies:
            if time.time() > deadline - 5:
                break
            Om0, w0, wa = cosmo
            ref = _solve_reference(Om0, w0, wa, a_grid_refine)
            if ref is not None:
                reference_cache_refine[(Om0, w0, wa)] = ref
        
        if reference_cache_refine:
            best_err_refine = _compute_error_fast(best_coeffs, cosmologies, a_grid_refine, reference_cache_refine)
            
            def objective_refine(coeffs):
                if time.time() > deadline:
                    return best_err_refine
                err = _compute_error_fast(coeffs, cosmologies, a_grid_refine, reference_cache_refine)
                return err
            
            try:
                result2 = minimize(objective_refine, best_coeffs, method='Powell', 
                                 options={'maxiter': 50, 'ftol': 1e-6})
                if result2.fun < best_err_refine:
                    best_coeffs = result2.x
                    best_err_refine = result2.fun
            except:
                pass
    
    # Final check and clipping
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
