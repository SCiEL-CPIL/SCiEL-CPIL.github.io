"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    """
    Fit six coefficients for the growth-index formula:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1 - Om(a), u = 1 + w0 + wa*(1-a), f = Om(a)^gamma
    
    We minimize the maximum relative error of f and D(a)/D(1) over a grid of
    CPL parameters and scale factors.
    """
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Representative CPL parameters for fitting
    # w0, wa pairs covering typical dark energy models
    w0_wa_pairs = [
        (-1.0, 0.0),   # Lambda CDM
        (-0.9, 0.1),
        (-1.1, -0.5),
        (-0.8, 0.3),
        (-1.2, -0.3),
        (-0.7, 0.2),
    ]
    
    # Scale factors to evaluate
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    # Matter density parameters (typical values)
    Om0_values = [0.25, 0.27, 0.30]
    
    # Generate reference data by solving the ODE
    reference_data = []
    for Om0 in Om0_values:
        for w0, wa in w0_wa_pairs:
            # Compute reference f and D using the ODE
            f_ref, D_ref = _solve_growth_ode(Om0, w0, wa, a_values)
            for a, f_r, D_r in zip(a_values, f_ref, D_ref):
                # Compute q, u for this cosmology
                Om_a = Om0 * a**3 / (Om0 * a**3 + (1 - Om0) * a**(-3 * (w0 + wa * (1 - a))))
                q = 1 - Om_a
                u = 1 + w0 + wa * (1 - a)
                reference_data.append((Om_a, q, u, wa, a, f_r, D_r))
    
    # Convert to arrays
    Om_a_arr = np.array([d[0] for d in reference_data])
    q_arr = np.array([d[1] for d in reference_data])
    u_arr = np.array([d[2] for d in reference_data])
    wa_arr = np.array([d[3] for d in reference_data])
    a_arr = np.array([d[4] for d in reference_data])
    f_ref_arr = np.array([d[5] for d in reference_data])
    D_ref_arr = np.array([d[6] for d in reference_data])
    
    # Filter out invalid or extreme values
    valid_mask = (f_ref_arr > 0) & (D_ref_arr > 0) & np.isfinite(f_ref_arr) & np.isfinite(D_ref_arr)
    Om_a_arr = Om_a_arr[valid_mask]
    q_arr = q_arr[valid_mask]
    u_arr = u_arr[valid_mask]
    wa_arr = wa_arr[valid_mask]
    a_arr = a_arr[valid_mask]
    f_ref_arr = f_ref_arr[valid_mask]
    D_ref_arr = D_ref_arr[valid_mask]
    
    if len(Om_a_arr) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def compute_errors(params):
        c0, c1, c2, c3, c4, c5 = params
        gamma = c0 + c1 * q_arr + c2 * q_arr**2 + c3 * u_arr + c4 * wa_arr * a_arr + c5 * u_arr * q_arr
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            # Penalize out-of-bounds gamma
            penalty = np.sum(np.abs(gamma - np.clip(gamma, 0.1, 1.2)))
            return np.concatenate([np.full(len(f_ref_arr), 10.0), [penalty]])
        
        f_model = Om_a_arr ** gamma
        
        # Relative error for f
        rel_err_f = np.abs(f_model - f_ref_arr) / np.maximum(f_ref_arr, 1e-10)
        
        # Compute D from model: D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # For simplicity in the objective, we can approximate or compute D directly
        # But for speed, let's just use f errors primarily and add D errors if time permits
        # Actually, let's compute D properly
        
        D_model = _compute_D_from_f(f_model, a_arr, Om_a_arr, wa_arr)
        rel_err_D = np.abs(D_model - D_ref_arr) / np.maximum(D_ref_arr, 1e-10)
        
        # Combined error: maximize relative error
        max_rel_err = np.maximum(np.max(rel_err_f), np.max(rel_err_D))
        
        # Return vector of errors for least_squares
        return np.concatenate([rel_err_f, rel_err_D])
    
    def _compute_D_from_f(f_vals, a_vals, Om_vals, wa_vals):
        """Compute D(a)/D(1) from f values using numerical integration."""
        D_vals = np.zeros_like(a_vals)
        for i, a in enumerate(a_vals):
            if a >= 1.0:
                D_vals[i] = 1.0
                continue
            # Integrate from ln(a) to 0: -integral of f(exp(x)) dx
            # Use simple trapezoidal rule on the grid
            x_vals = np.log(a_vals)
            f_x_vals = f_vals
            # Find points where x >= ln(a)
            mask = x_vals >= np.log(a)
            if np.sum(mask) < 2:
                D_vals[i] = 1.0
                continue
            x_int = x_vals[mask]
            f_int = f_x_vals[mask]
            # Integrate from ln(a) to 0
            # Add endpoint x=0 if not present
            if x_int[-1] < 0:
                x_int = np.append(x_int, 0.0)
                f_int = np.append(f_int, f_vals[-1])  # f at a=1
            integral = np.trapz(f_int, x_int)
            # The integral is from ln(a) to 0, so D = exp(-integral)
            D_vals[i] = np.exp(-integral)
        return D_vals
    
    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Bounds for coefficients
    bounds = ([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])
    
    best_params = x0
    best_score = float('inf')
    
    # Try multiple initial points
    initial_points = [
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.1, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.55, 0.05, 0.05, 0.05, 0.05, 0.05]),
    ]
    
    for x_init in initial_points:
        if time.time() > deadline:
            break
        
        try:
            result = least_squares(compute_errors, x_init, bounds=bounds, max_nfev=1000, ftol=1e-8)
            if result.success or result.cost < best_score:
                # Evaluate the max relative error for this result
                errors = compute_errors(result.x)
                max_err = np.max(errors)
                if max_err < best_score:
                    best_score = max_err
                    best_params = result.x
        except Exception:
            continue
    
    # Ensure coefficients are within [-10, 10]
    best_params = np.clip(best_params, -10, 10)
    
    return list(best_params)

def _solve_growth_ode(Om0, w0, wa, a_values):
    """Solve the growth ODE to get reference f and D values."""
    from scipy.integrate import solve_ivp
    
    a_min = min(a_values)
    a_max = max(a_values)
    
    # Transform to x = ln(a)
    x_min = np.log(a_min)
    x_max = np.log(a_max)
    
    def H2_of_x(x):
        a = np.exp(x)
        Om_m = Om0 * a**3 / (Om0 * a**3 + (1 - Om0) * a**(-3 * (w0 + wa * (1 - a))))
        Om_de = 1 - Om_m
        w_eff = w0 + wa * (1 - a)
        return Om_m + Om_de * (1 + 3 * w_eff)
    
    def ode_system(x, y):
        # y[0] = D, y[1] = D' (with respect to x)
        D, Dp = y
        H2 = H2_of_x(x)
        # D'' + (3/H2 - 1) D' - (3/2) D = 0 approximately
        # More precisely: D'' + (3/2)(1 + Om_m/(1-Om_m)) D' - (3/2) Om_m D = 0
        Om_m = Om0 * np.exp(3*x) / (Om0 * np.exp(3*x) + (1 - Om0) * np.exp(-3*(w0 + wa*(1-np.exp(x)))*x))
        coeff = (3/2) * (1 + Om_m / (1 - Om_m))
        return [Dp, -coeff * Dp + (3/2) * Om_m * D]
    
    # Integrate from x_max to x_min
    x_eval = np.linspace(x_max, x_min, len(a_values))
    sol = solve_ivp(ode_system, [x_max, x_min], [1.0, 0.0], t_eval=x_eval, rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        # Fallback to constant gamma
        f_vals = np.zeros(len(a_values))
        D_vals = np.zeros(len(a_values))
        for i, a in enumerate(a_values):
            Om_m = Om0 * a**3 / (Om0 * a**3 + (1 - Om0) * a**(-3 * (w0 + wa * (1 - a))))
            gamma = 0.55
            f_vals[i] = Om_m ** gamma
            D_vals[i] = a ** (1 + gamma)  # Approximate
        return f_vals, D_vals
    
    D_vals = sol.y[0]
    Dp_vals = sol.y[1]
    
    # Compute f = D' / D
    f_vals = Dp_vals / D_vals
    
    # Normalize D so D(1) = 1
    D_vals = D_vals / D_vals[0]  # D at x_max (a=1)
    
    return f_vals, D_vals
# EVOLVE-BLOCK-END
