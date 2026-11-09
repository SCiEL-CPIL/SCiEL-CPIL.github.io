import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit

    # 1. Define Training Cosmologies
    # We need a set that is representative of "disjoint public validation cosmologies".
    # Standard LCDM, CPL variations, Phantom, Quintessence.
    # Om: 0.25 - 0.35
    # w0: -1.5 to 0.5
    # wa: -1.0 to 1.0
    
    cosmologies = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.3),   # CPL
        (0.30, -1.0, -0.3),  # CPL
        (0.25, -1.0, 0.0),   # Low Om LCDM
        (0.35, -1.0, 0.0),   # High Om LCDM
        (0.30, -0.5, 0.5),   # Quintessence-like
        (0.30, -1.5, 0.5),   # Phantom-like
        (0.30, -0.8, 0.2),   # General CPL
        (0.30, -1.2, 0.0),   # Phantom w0
        (0.30, -0.2, 0.8),   # Strong wa
        (0.28, -1.1, 0.4),   # Mixed
        (0.32, -0.9, -0.4),  # Mixed
    ]
    
    # Scale factors
    # Logarithmic spacing is important for the integral in D(a)
    a_vals = np.logspace(-4, 0, 100)
    x_vals = np.log(a_vals)
    dx = x_vals[1] - x_vals[0] # constant step in log space
    
    # 2. Pre-compute Reference Solutions
    ref_om_a = np.zeros((len(cosmologies), len(a_vals)))
    ref_f = np.zeros((len(cosmologies), len(a_vals)))
    ref_D = np.zeros((len(cosmologies), len(a_vals)))
    ref_q = np.zeros((len(cosmologies), len(a_vals)))
    ref_u = np.zeros((len(cosmologies), len(a_vals)))
    ref_wa_a = np.zeros((len(cosmologies), len(a_vals)))
    
    for i, (om, w0, wa) in enumerate(cosmologies):
        if time.time() - start_time > timeout:
            break
        try:
            # E^2(a) = Om a^-3 + (1-Om) a^(-3(1+w0+wa))
            def E_sq(a):
                return om * np.power(a, -3.0) + (1.0 - om) * np.power(a, -3.0 * (1.0 + w0 + wa))
            
            # ODE for D: D'' + (3 + H'/H) D' - (3/2) (Om(a)/a^2) D = 0
            # In terms of x = ln a:
            # Let D(x) = D(a). d/da = (1/a) d/dx.
            # D' = (1/a) D_x
            # D'' = (1/a^2) D_xx - (1/a^2) D_x
            # Substitute: (1/a^2)(D_xx - D_x) + (3 + H'/H)(1/a)D_x - (3/2)(Om(a)/a^2)D = 0
            # Multiply by a^2:
            # D_xx - D_x + (3 + H'/H) D_x - 1.5 Om(a) D = 0
            # D_xx + (2 + H'/H) D_x - 1.5 Om(a) D = 0
            
            def ode(x, y):
                a = np.exp(x)
                D, D_x = y
                e2 = E_sq(a)
                if e2 <= 0:
                    return [0, 0]
                # H'/H = 0.5 * (E^2)' / E^2
                # (E^2)' = d/da E^2 = -3 om a^-4 - 3(1+w0+wa)(1-om) a^(-3(1+w0+wa)-1)
                # In terms of x: dE^2/dx = a * dE^2/da
                # dE^2/dx = -3 om a^-3 - 3(1+w0+wa)(1-om) a^(-3(1+w0+wa))
                # Let A = om a^-3, B = (1-om) a^(-3(1+w0+wa))
                # E^2 = A + B
                # dE^2/dx = -3A - 3(1+w0+wa)B
                A = om * np.power(a, -3.0)
                B = (1.0 - om) * np.power(a, -3.0 * (1.0 + w0 + wa))
                dE2_dx = -3.0 * A - 3.0 * (1.0 + w0 + wa) * B
                h_prime_over_h = 0.5 * dE2_dx / e2
                om_a = A / e2
                
                D_xx = 1.5 * om_a * D - (2.0 + h_prime_over_h) * D_x
                return [D_x, D_xx]
            
            x_min = np.log(1e-4)
            x_max = 0.0
            a0 = np.exp(x_min)
            # Initial conditions: D ~ a, D_x ~ 1
            y0 = [a0, 1.0]
            
            sol = solve_ivp(ode, [x_min, x_max], y0, rtol=1e-9, atol=1e-11, dense_output=True)
            
            if not sol.success:
                continue
                
            D_arr = sol.sol(x_vals)[0]
            D_x_arr = sol.sol(x_vals)[1]
            
            # Normalize D(1) = 1
            D_norm_factor = D_arr[-1]
            D_norm = D_arr / D_norm_factor
            
            # f = d ln D / d ln a = D_x / D
            f_arr = D_x_arr / D_arr
            
            # Store reference data
            ref_D[i, :] = D_norm
            ref_f[i, :] = f_arr
            
            # Store auxiliary variables for the ansatz
            e2_vals = om * np.power(a_vals, -3.0) + (1.0 - om) * np.power(a_vals, -3.0 * (1.0 + w0 + wa))
            om_a_vals = (om * np.power(a_vals, -3.0)) / e2_vals
            ref_om_a[i, :] = om_a_vals
            ref_q[i, :] = 1.0 - om_a_vals
            ref_u[i, :] = 1.0 + w0 + wa * (1.0 - a_vals)
            ref_wa_a[i, :] = wa * a_vals
            
        except Exception:
            continue

    if not np.any(ref_D != 0):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Mask for valid data (all should be valid if loop completed)
    valid_mask = np.any(ref_D != 0, axis=1)
    ref_om_a = ref_om_a[valid_mask]
    ref_f = ref_f[valid_mask]
    ref_D = ref_D[valid_mask]
    ref_q = ref_q[valid_mask]
    ref_u = ref_u[valid_mask]
    ref_wa_a = ref_wa_a[valid_mask]

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Compute gamma for all points
        # q, u, wa_a are pre-computed arrays of shape (N_cos, N_a)
        gamma = c0 + c1 * ref_q + c2 * ref_q**2 + c3 * ref_u + c4 * ref_wa_a + c5 * ref_u * ref_q
        
        # Enforce constraint gamma in [0.1, 1.2]
        # Using clip is a common trick. If the optimal solution lies outside, 
        # the optimizer will push against the boundary.
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        
        # Calculate f_app
        f_app = np.power(ref_om_a, gamma_clipped)
        
        # Calculate D_app
        # D(a)/D(1) = exp( - ∫_{ln(a)}^0 f(exp(x)) dx )
        # We have x_vals increasing from x_min to 0.
        # Integral I_i = ∫_{x_i}^{x_max} f dx.
        # Using trapezoidal rule on log-spaced grid.
        # f_app is defined at x_vals.
        # Let's compute cumulative integral from the end (x=0) backwards?
        # Or forward and subtract.
        # Forward cumtrapz: C_i = ∫_{x_0}^{x_i} f dx.
        # Total = C_last.
        # ∫_{x_i}^{x_max} f dx = Total - C_i.
        
        # Since dx is constant in log space:
        # trapezoidal rule: ∫_{x_i}^{x_j} f dx ≈ sum_{k=i}^{j-1} 0.5 * (f_k + f_{k+1}) * dx
        # Let's use np.cumsum on the trapezoids.
        f_avg = 0.5 * (f_app[:, :-1] + f_app[:, 1:])
        int_steps = f_avg * dx
        # Cumulative sum along axis 1
        # C[i, j] = sum_{k=0}^{j-1} int_steps[i, k]
        # We want integral from x_j to x_end (x_max).
        # Total integral for row i is int_steps[i, :].sum()
        # Integral from x_j to x_max is Total_i - C[i, j]
        
        cum_int = np.concatenate(([0.0] * f_app.shape[0], np.cumsum(int_steps, axis=1).flatten()), axis=1).reshape(f_app.shape[0], -1)
        # Wait, cumsum shape is (N, N-1). We need (N, N).
        # Let's do it properly.
        cum_int = np.zeros_like(f_app)
        cum_int[:, 1:] = np.cumsum(int_steps, axis=1)
        
        total_int = cum_int[:, -1]
        # Integral from x_i to x_max (which is x_last)
        # Note: x_max corresponds to the last index.
        # The integral from x_i to x_max is total_int - cum_int[:, i]
        # However, cum_int[:, i] is ∫_{x_0}^{x_i}.
        # So ∫_{x_i}^{x_max} = total_int - cum_int[:, i].
        
        exponents = -(total_int[:, np.newaxis] - cum_int)
        D_app = np.exp(exponents)
        
        # Relative Errors
        # f error
        err_f = np.abs(f_app - ref_f) / np.maximum(ref_f, 1e-10)
        # D error
        err_D = np.abs(D_app - ref_D) / np.maximum(ref_D, 1e-10)
        
        # Max error
        max_err = np.max(np.concatenate([err_f.flatten(), err_D.flatten()]))
        
        # Penalty for gamma constraint violation (if we didn't clip, or if we want to discourage boundary solutions)
        # Since we clipped, the error is calculated with the clipped gamma.
        # This is a valid strategy: the "effective" gamma is the clipped one.
        
        return max_err

    # Optimization
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_coeffs = x0
    best_score = objective(x0)
    
    # Global Search: Differential Evolution
    # Bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    try:
        # DE is robust but can be slow. 
        # maxiter=100, popsize=15 (default)
        # We check time frequently.
        def de_callback(xk, convergence):
            if time.time() - start_time > timeout - 5:
                raise Exception("Timeout in DE")
            
        res_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6, 
            callback=de_callback,
            seed=42
        )
        
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_coeffs = res_de.x
            
    except Exception:
        pass
        
    # Local Polish: Nelder-Mead
    if time.time() - start_time < timeout - 2:
        try:
            res_nm = minimize(
                objective, 
                best_coeffs, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-8}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_coeffs = res_nm.x
        except Exception:
            pass

    # Final Polish: Powell (derivative-free, good for non-smooth)
    if time.time() - start_time < timeout - 1:
        try:
            res_pw = minimize(
                objective, 
                best_coeffs, 
                method='Powell', 
                options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-8}
            )
            if res_pw.fun < best_score:
                best_score = res_pw.fun
                best_coeffs = res_pw.x
        except Exception:
            pass

    # Clip final coefficients
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()
