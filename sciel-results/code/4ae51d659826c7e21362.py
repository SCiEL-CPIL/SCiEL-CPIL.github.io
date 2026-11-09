"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Define the representative validation cosmologies
    # Using a dense grid to ensure we catch the worst-case errors
    om_m_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_values = [-1.2, -1.0, -0.8]
    wa_values = [-0.3, 0.0, 0.3]
    
    test_models = []
    for om in om_m_values:
        for w0 in w0_values:
            for wa in wa_values:
                test_models.append((om, w0, wa))
                
    # 2. Pre-compute reference data
    a_eval = np.linspace(0.1, 1.0, 100)
    ln_a_eval = np.log(a_eval)
    
    ref_data = []
    
    for om_m, w0, wa in test_models:
        # Define ODE RHS
        def ode_rhs(ln_a, f):
            a = np.exp(ln_a)
            exp_wa = 3 * (1 + w0 + wa)
            de_term = (1 - om_m) * (a ** exp_wa)
            mat_term = om_m * (a ** 3)
            e2 = mat_term + de_term
            om_a = mat_term / e2 if e2 > 0 else 0.0
            
            coeff = 1.5 * (1 + om_a) + (2.0/3.0) * w0 + (2.0/3.0) * wa * (1 - a)
            return -f**2 - f * coeff + 0.5 * om_a

        # Initial condition
        a_init = 0.1
        ln_a_init = np.log(a_init)
        exp_wa_init = 3 * (1 + w0 + wa)
        de_term_init = (1 - om_m) * (a_init ** exp_wa_init)
        mat_term_init = om_m * (a_init ** 3)
        e2_init = mat_term_init + de_term_init
        om_a_init = mat_term_init / e2_init if e2_init > 0 else 0.0
        f_init = om_a_init ** 0.55
        
        t_span = (ln_a_init, 0.0)
        t_eval = ln_a_eval
        
        try:
            sol = solve_ivp(ode_rhs, t_span, [f_init], t_eval=t_eval, rtol=1e-8, atol=1e-10)
            if not sol.success:
                continue
                
            f_ref = sol.y[0]
            
            # Calculate D_ref
            ln_a_rev = ln_a_eval[::-1]
            f_ref_rev = f_ref[::-1]
            
            # np.trapezoid is available in newer numpy, cumtrapz in older. 
            # Using np.trapezoid for consistency with current program style if available, 
            # but falling back to manual trapezoidal if needed. 
            # The current program used np.trapezoid in one place and np.cumtrapz in another. 
            # Let's use a robust manual cumulative trapezoid to be safe.
            def cum_trapezoid(y, x):
                """Cumulative trapezoidal integration"""
                d = np.diff(x)
                ad_y = (y[1:] + y[:-1]) * 0.5
                return np.concatenate(([0.0], np.cumsum(ad_y * d)))

            int_vals = cum_trapezoid(f_ref_rev, ln_a_rev)
            int_vals_rev = int_vals[::-1]
            D_ref = np.exp(int_vals_rev)
            
            # Pre-compute static parts of the ansatz inputs
            exp_wa = 3 * (1 + w0 + wa)
            de_term = (1 - om_m) * (a_eval ** exp_wa)
            mat_term = om_m * (a_eval ** 3)
            e2 = mat_term + de_term
            om_a = mat_term / e2
            
            q = 1 - om_a
            u = 1 + w0 + wa * (1 - a_eval)
            
            ref_data.append({
                'f_ref': f_ref,
                'D_ref': D_ref,
                'om_a': om_a,
                'q': q,
                'u': u,
                'wa': wa,
                'a': a_eval
            })
        except Exception:
            continue

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Pre-flatten reference data for fast access
    f_refs = np.array([d['f_ref'] for d in ref_data])
    D_refs = np.array([d['D_ref'] for d in ref_data])
    om_as = np.array([d['om_a'] for d in ref_data])
    qs = np.array([d['q'] for d in ref_data])
    us = np.array([d['u'] for d in ref_data])
    was = np.array([d['wa'] for d in ref_data])
    as_ = np.array([d['a'] for d in ref_data])
    
    # Flatten for error calculation
    f_refs_flat = f_refs.flatten()
    D_refs_flat = D_refs.flatten()
    om_as_flat = om_as.flatten()
    qs_flat = qs.flatten()
    us_flat = us.flatten()
    was_flat = was.flatten()
    as_flat = as_.flatten()
    
    # 3. Define fast error calculation
    def calculate_errors(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Compute gamma
        gamma = c0 + c1*qs_flat + c2*qs_flat**2 + c3*us_flat + c4*was_flat*as_flat + c5*us_flat*qs_flat
        
        # Check bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return None # Signal violation
        
        f_approx = om_as_flat ** gamma
        
        # Error in f
        err_f = np.abs(f_approx - f_refs_flat) / np.abs(f_refs_flat)
        
        # Calculate D_approx
        # We need to integrate f_approx over ln_a for each model separately
        # f_approx is flattened, shape (N_models, N_points)
        f_approx_grid = f_approx.reshape(len(ref_data), -1)
        
        D_approx_flat = np.zeros_like(f_refs_flat)
        
        for i in range(len(ref_data)):
            f_row = f_approx_grid[i]
            # Integrate from 0 to ln_a
            # We have ln_a_eval (ascending). We need integral from 0 (end) to ln_a (start).
            # Reverse to start from 0
            f_rev = f_row[::-1]
            ln_a_rev = ln_a_eval[::-1]
            
            int_vals = cum_trapezoid(f_rev, ln_a_rev)
            int_vals_rev = int_vals[::-1]
            D_approx_flat[i * len(a_eval) : (i+1) * len(a_eval)] = np.exp(int_vals_rev)
            
        # Error in D
        err_D = np.abs(D_approx_flat - D_refs_flat) / np.abs(D_refs_flat)
        
        return np.concatenate([err_f, err_D])

    # 4. Optimization using Log-Sum-Exp surrogate for Max Error
    def objective(x):
        if time.time() > deadline:
            return 1e10
            
        errors = calculate_errors(x)
        if errors is None:
            return 1e10
            
        # Smooth approximation to max error
        # LSE = log(sum(exp(beta * errors)))
        # We want to minimize this.
        # Scale errors to prevent overflow. Max error is typically < 0.5 (50%).
        # beta=50 is a good balance.
        beta = 50.0
        scaled_errors = beta * errors
        max_val = np.max(scaled_errors)
        lse = max_val + np.log(np.sum(np.exp(scaled_errors - max_val)))
        
        # Normalize by number of points to keep magnitude reasonable
        return lse / len(errors)

    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_x = x0
    best_score = objective(x0)
    
    # Use Powell's method, which is robust for this smooth surrogate
    try:
        res = minimize(objective, x0, method='Powell', 
                       options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass

    # Final check and clip
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)

# EVOLVE-BLOCK-END
