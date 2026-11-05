import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s safety margin

    # Fixed grid for optimization and reference
    # Use a log-spaced grid for better resolution at low a
    a_grid = np.geomspace(0.05, 1.0, 100)
    x_grid = np.log(a_grid)
    
    # Representative cosmologies for training/search
    # Om0, w0, wa
    train_cosmo = [
        (0.30, -1.0, 0.0),   # LambdaCDM
        (0.25, -1.2, -0.3),  # Phantom
        (0.35, -0.8, 0.3),   # Quintessence
        (0.28, -1.1, 0.2),   # Mixed
        (0.32, -0.9, -0.1),  # Near CDL
    ]

    def get_omega_m(Om0, w0, wa, a_vals):
        OmL0 = 1.0 - Om0
        # rho_L(a) proportional to a^(-3(1+w0+wa)) * exp(3wa*a)
        # rho_m(a) proportional to a^(-3)
        # Om(a) = Om0 * a^-3 / (Om0 * a^-3 + OmL0 * a^(-3(1+w0+wa)) * exp(3wa*a))
        
        # Handle a=0 or very small carefully, though grid starts at 0.05
        a_safe = np.maximum(a_vals, 1e-10)
        
        term_m = a_safe ** (-3.0)
        exponent_L = -3.0 * (1.0 + w0 + wa) + 3.0 * wa * a_safe
        # Note: a^(-3(1+w0+wa)) * exp(3wa*a) = exp(-3(1+w0+wa)*ln(a) + 3wa*a)
        # Using log form is more stable for small a if exponents are large?
        # Let's stick to power for now as a > 0.05
        term_L = np.exp(exponent_L)
        
        denom = Om0 * term_m + OmL0 * term_L
        return (Om0 * term_m) / denom

    def get_reference_growth(Om0, w0, wa, a_vals):
        OmL0 = 1.0 - Om0
        
        def omega_m(a):
            # Scalar or array
            if np.isscalar(a):
                a_arr = np.array([a])
            else:
                a_arr = a
            return get_omega_m(Om0, w0, wa, a_arr)

        def H2(a):
            om = omega_m(a)
            # H^2(a) = H0^2 * (Om0 a^-3 + OmL0 a^(-3(1+w0+wa)) exp(3wa a))
            # We only need relative H2 or d(H2)/da
            # Let's compute F(a) = Om0 a^-3 + OmL0 a^(-3(1+w0+wa)) exp(3wa a)
            a_safe = np.maximum(np.asarray(a), 1e-10)
            term_m = a_safe ** (-3.0)
            exponent_L = -3.0 * (1.0 + w0 + wa) + 3.0 * wa * a_safe
            term_L = np.exp(exponent_L)
            return Om0 * term_m + OmL0 * term_L

        def growth_ode(a, y):
            D, Dp = y
            om = omega_m(a)[0]
            
            # Numerical derivative for H2
            h = 1e-5
            H2_val = H2(a)
            H2_p = (H2(a + h) - H2(a - h)) / (2.0 * h)
            H_over_H_prime = 0.5 * H2_p / H2_val
            
            coeff1 = 3.0 / a + H_over_H_prime
            coeff2 = 1.5 * om / (a * a)
            
            return [Dp, -coeff1 * Dp + coeff2 * D]

        # Integrate from a_min to 1.0
        a_min = np.min(a_vals)
        a_max = 1.0
        
        # Initial condition at a_min
        # Matter dominated approx: D ~ a^1.5, D' = 1.5 a^0.5
        D_init = 1.0
        Dp_init = 1.5 * np.sqrt(a_min)
        
        sol = solve_ivp(growth_ode, [a_min, a_max], [D_init, Dp_init], 
                        t_eval=a_vals, method='RK45', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            return None, None
        
        D = sol.y[0]
        Dp = sol.y[1]
        
        # Normalize D(1) = 1
        D_norm = D / D[-1]
        
        # f = a * D' / D
        f_ref = a_vals * Dp / D
        
        return f_ref, D_norm

    # Pre-compute reference data for training set
    ref_data = []
    for Om0, w0, wa in train_cosmo:
        if time.time() - start_time > time_limit * 0.3:
            break
        f_ref, D_ref = get_reference_growth(Om0, w0, wa, a_grid)
        if f_ref is not None:
            ref_data.append((Om0, w0, wa, f_ref, D_ref))

    if not ref_data:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    def get_approx(Om0, w0, wa, c, a_vals):
        Om_a = get_omega_m(Om0, w0, wa, a_vals)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_vals + c[5]*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return None, None, False
            
        f_app = Om_a ** gamma
        
        # Calculate D_app
        # D(a)/D(1) = exp(- \int_{\ln a}^0 f(\exp(x)) dx)
        # Let x = ln(a). We have a_vals, so x_vals = ln(a_vals).
        # We need I(x) = \int_x^0 f(e^t) dt.
        # Note: x is negative for a < 1.
        # Let K(x) = \int_0^x f(e^t) dt. Then I(x) = -K(x).
        # We can compute K(x) for all points in x_grid using cumulative_trapezoid.
        
        x_vals = np.log(a_vals)
        
        # Ensure x_vals are sorted for cumulative_trapezoid (they are, as a_vals is geomspace)
        # cumulative_trapezoid(y, x, initial=0) computes \int_{x[0]}^x y dx
        # We want \int_0^x.
        # Let's shift the grid or handle the offset.
        # Easier: Compute integral from x_min to x, then subtract integral from x_min to 0?
        # Or just compute \int_{x_i}^{x_{i+1}} and sum backwards from 0?
        
        # Let's use a standard cumulative sum approach on the x-grid.
        # f_vals = f_app
        # dx = np.diff(x_vals)
        # trapz_area = 0.5 * (f_vals[1:] + f_vals[:-1]) * dx
        # We want J(x_i) = \int_{x_i}^0 f dx.
        # Since x_i <= 0, J(x_i) = - \int_0^{x_i} f dx.
        
        # Let's compute C(x_i) = \int_{x_0}^{x_i} f dx using cumulative_trapezoid.
        # C(0) is the value at index where x=0 (last element if a_max=1).
        # Let idx_0 = len(x_vals) - 1 (assuming a_grid ends at 1.0)
        # Then \int_0^{x_i} f dx = C(x_i) - C(0).
        # So J(x_i) = -(C(x_i) - C(0)) = C(0) - C(x_i).
        
        if len(x_vals) > 1:
            C = cumulative_trapezoid(f_app, x_vals, initial=0.0)
            C_0 = C[-1] # Since a_grid ends at 1.0, x[-1]=0
            J = C_0 - C
            D_app = np.exp(-J)
        else:
            D_app = np.ones_like(a_vals)
            
        return f_app, D_app, True

    def objective(c):
        max_err = 0.0
        for Om0, w0, wa, f_ref, D_ref in ref_data:
            f_app, D_app, valid = get_approx(Om0, w0, wa, c, a_grid)
            if not valid:
                return 1e6 # Penalty for gamma out of bounds
            
            # Relative errors
            # Avoid division by zero
            denom_f = np.maximum(np.abs(f_ref), 1e-10)
            denom_D = np.maximum(np.abs(D_ref), 1e-10)
            
            err_f = np.abs(f_app - f_ref) / denom_f
            err_D = np.abs(D_app - D_ref) / denom_D
            
            local_max = np.max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max
                
        return max_err

    # Initial guess
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Bounds
    bounds = [(-10, 10) for _ in range(6)]
    
    best_c = c0
    best_score = objective(best_c)
    
    if time.time() - start_time > time_limit * 0.4:
        return best_c

    try:
        # Use L-BFGS-B
        res = minimize(objective, c0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50})
        
        if res.fun < best_score:
            best_c = res.x
            best_score = res.fun
            
        # Polish with Nelder-Mead if time permits
        if time.time() - start_time < time_limit * 0.8:
             res2 = minimize(objective, best_c, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
             if res2.fun < best_score:
                 best_c = res2.x
                 best_score = res2.fun
                 
    except Exception as e:
        pass # Keep best_c found so far

    # Final check
    if time.time() - start_time > time_limit * 0.9:
        return best_c

    # Ensure finite and within bounds
    best_c = np.clip(best_c, -10, 10)
    if not np.all(np.isfinite(best_c)):
        return c0

    return best_c
