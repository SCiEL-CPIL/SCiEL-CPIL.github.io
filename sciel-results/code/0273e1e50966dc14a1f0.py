import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    # Representative validation cosmologies
    # (Omega_m0, Omega_k0, w0, wa)
    # Assuming flat universe for simplicity unless specified, but let's allow Omega_k=0.
    # The task says "smooth CPL dark-energy cosmologies".
    # Standard values: Om0 ~ 0.3, Ok ~ 0.
    # w0, wa vary.
    
    models = [
        (0.3, 0.0, -1.0, 0.0),  # LCDM
        (0.3, 0.0, -1.0, 0.5),  # CPL positive wa
        (0.3, 0.0, -0.5, -0.5), # CPL negative wa, w0 > -1
        (0.3, 0.0, -1.5, 0.5),  # w0 < -1
        (0.3, 0.0, -1.0, -0.5), # CPL negative wa
        (0.2, 0.0, -1.0, 0.0),  # Lower Om
        (0.4, 0.0, -1.0, 0.0),  # Higher Om
    ]
    
    # Grid of a values for evaluation
    # a from 0.01 to 1.0
    # Use log spacing for better coverage of early times
    a_vals = np.geomspace(0.01, 1.0, 100)
    
    def get_true_growth(params, a_grid):
        Om0, Ok0, w0, wa = params
        # Omega_Lambda0 = 1 - Om0 - Ok0 (assuming flat if Ok=0, but let's be general)
        # If Ok is not 0, we need Omega_L0. Let's assume flat for simplicity in models, 
        # but the ODE solver should handle general curvature if we define E(a) correctly.
        # The prompt implies "smooth CPL", usually flat.
        
        def E2(a):
            Om_a = Om0 * a**(-3)
            Ok_a = Ok0 * a**(-2)
            # w(a) = w0 + wa * (1 - a)
            w_a = w0 + wa * (1 - a)
            # Integral of w(a) da from 1 to a? 
            # rho_DE = rho_DE0 * exp(-3 * integral (1+w) da/a)
            # integral (1 + w0 + wa(1-a)) da/a = (1+w0) ln a + wa(a - 1)
            # So rho_DE(a) = rho_DE0 * a^(-(1+w0)) * exp(-wa(a-1)) = rho_DE0 * a^(-(1+w0)) * exp(wa(1-a))
            # Let's check: d ln rho / d ln a = -3(1+w).
            # ln rho = -3(1+w0) ln a - 3 wa (a-1) + C.
            # rho = rho0 * a^(-(1+w0)) * exp(-3 wa (a-1)).
            # Wait, the exponent is -3 * integral (1+w) da/a.
            # integral (1+w0+wa-wa*a) da/a = (1+w0)ln a + wa*a - wa.
            # So rho_DE(a) = rho_DE(1) * a^(-(1+w0)) * exp(-wa(a-1)).
            # Let's re-verify.
            # d rho / da = -3 H rho (1+w).
            # d ln rho / d ln a = -3(1+w).
            # ln(rho/rho0) = -3 [ (1+w0) ln a + wa (a-1) ].
            # rho = rho0 * a^(-(1+w0)) * exp(-3 wa (a-1)).
            # Yes.
            
            # If Ok0 is not 0, we need to normalize.
            # Let's assume flat universe for the models provided.
            Omd0 = 1.0 - Om0 - Ok0
            Omd_a = Omd0 * a**(-(1+w0)) * np.exp(-3*wa*(a-1))
            
            return Om_a + Ok_a + Omd_a
        
        def ode_func(x, y):
            a = np.exp(x)
            # y[0] = F, y[1] = F'
            # F'' + (3/2 + 1/2 d ln E^2 / d ln a) F' + 3/2 Om(a) F = 0
            # d ln E^2 / d ln a = (1/E^2) d E^2 / d ln a
            # E^2 = Om a^-3 + Ok a^-2 + Omd a^(-(1+w0)) exp(-3wa(a-1))
            # d E^2 / d ln a = -3 Om a^-3 - 2 Ok a^-2 - 3(1+w(a)) Omd(a)
            
            Om_a = Om0 * a**(-3)
            Ok_a = Ok0 * a**(-2)
            w_a = w0 + wa * (1 - a)
            Omd0 = 1.0 - Om0 - Ok0
            Omd_a = Omd0 * a**(-(1+w0)) * np.exp(-3*wa*(a-1))
            
            E2_val = Om_a + Ok_a + Omd_a
            dE2_dlna = -3*Om_a - 2*Ok_a - 3*(1+w_a)*Omd_a
            
            H_prime_coeff = 0.5 * (3.0 + dE2_dlna / E2_val)
            
            F = y[0]
            F_prime = y[1]
            
            F_double_prime = -H_prime_coeff * F_prime - 1.5 * Om_a * F
            return [F_prime, F_double_prime]
        
        # Integrate from x=0 (a=1) backwards to x_min
        # F(1) = 1, F'(1) = f(1)
        # f(1) approx 1.0 for LCDM? f(1) = Om(1)^gamma ~ 0.3^0.55 ~ 0.43.
        # Actually f(1) is determined by the ODE. We can shoot or just integrate forward from small a?
        # Better to integrate from a=1 backwards to a_min, but we need F'(1).
        # We can use the approximation f(1) ~ Om(1)^gamma_seed to initialize, or just solve the ODE with a shooting method?
        # Or simpler: integrate from a_min to 1, normalize F(1)=1, and get F'(1)/F(1).
        # But the ODE is stable forward in time (a increasing) for the growing mode.
        # So integrate from a=0.01 to 1.0.
        
        x_min = np.log(a_grid[0])
        x_max = 0.0
        # Initial conditions at x_min: F=1 (arbitrary scale), F' ~ Om(a_min)^0.55
        a_min = a_grid[0]
        Om_min = Om0 * a_min**(-3)
        # In matter dominated era, f ~ 1. So F' ~ F.
        # Let's use F=1, F'=1.
        y0 = [1.0, 1.0]
        
        sol = solve_ivp(ode_func, [x_min, x_max], y0, t_eval=a_grid_x, method='RK45', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            return None, None
            
        F = sol.y[0]
        F_prime = sol.y[1]
        
        # Normalize F(1) = 1
        F_norm = F / F[-1]
        F_prime_norm = F_prime / F[-1]
        
        # f_true = F' / F
        f_true = F_prime_norm / F_norm
        
        return f_true, F_norm

    # Precompute a_grid_x
    a_grid_x = np.log(a_vals)
    
    # Cache true values
    true_data = []
    for m in models:
        f_t, D_t = get_true_growth(m, a_vals)
        if f_t is not None:
            true_data.append((m, f_t, D_t))
            
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        gamma_violation = 0.0
        
        for m, f_t, D_t in true_data:
            Om0, Ok0, w0, wa = m
            
            # Calculate gamma(a)
            # q = 1 - Om(a)
            # u = 1 + w0 + wa(1-a)
            # gamma = c0 + c1 q + c2 q^2 + c3 u + c4 wa a + c5 u q
            
            Om_a_arr = Om0 * a_vals**(-3)
            Ok_a_arr = Ok0 * a_vals**(-2)
            w_a_arr = w0 + wa * (1 - a_vals)
            Omd0 = 1.0 - Om0 - Ok0
            Omd_a_arr = Omd0 * a_vals**(-(1+w0)) * np.exp(-3*wa*(a_vals-1))
            
            E2_arr = Om_a_arr + Ok_a_arr + Omd_a_arr
            Om_frac = Om_a_arr / E2_arr
            
            q = 1.0 - Om_frac
            u = 1.0 + w0 + wa * (1 - a_vals)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
            
            # Check gamma constraints [0.1, 1.2]
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                gamma_violation += 100.0 * (np.max(np.abs(gamma - 0.65)) - 0.55 + 1) # Penalty
            
            f_approx = Om_frac**gamma
            
            # Relative error in f
            # Avoid division by zero if f_t is 0, but f_t > 0
            err_f = np.abs(f_approx - f_t) / np.abs(f_t)
            
            # Calculate D_approx
            # D(a)/D(1) = exp(- integral_ln(a)^0 f(exp(x)) dx)
            # x = ln a. integral from ln(a) to 0 of f(e^x) dx.
            # Let's integrate f_approx over x from ln(a) to 0.
            # Since we have discrete points, use cumulative integration.
            # a_vals is increasing. x_vals is increasing.
            # We want integral from x_i to 0.
            # Int = - (CumulativeIntegral[x_i] - CumulativeIntegral[0])?
            # Let I(x) = integral_0^x f(t) dt.
            # Then integral_x^0 f(t) dt = I(0) - I(x) = -I(x) since I(0)=0.
            # So D_approx(a_i) = exp(-I(x_i)).
            
            # Compute I(x) using cumulative trapezoid
            # x_vals = np.log(a_vals)
            # dx = np.diff(x_vals)
            # avg_f = (f_approx[:-1] + f_approx[1:]) / 2
            # I_diff = avg_f * dx
            # I_vals = np.concatenate(([0], np.cumsum(I_diff)))
            # This gives I(x_i) for i=0..N-1, with I(0)=0 at index 0?
            # No, x_vals[0] is ln(a_min). I(x_vals[0]) is integral from 0 to x_min?
            # The formula is integral from ln(a) to 0.
            # If we integrate from x_min to 0, we get total integral.
            # Let's integrate f_approx from x_vals[0] to 0.
            # Let J(x) = integral_x^0 f(t) dt.
            # J(0) = 0.
            # J(x_i) = integral_{x_i}^0 f(t) dt.
            # We can compute this by integrating backwards or using cumulative sum from the end.
            
            # Simpler: 
            # integral = np.zeros_like(a_vals)
            # integral[-1] = 0
            # for i in range(len(a_vals)-2, -1, -1):
            #     integral[i] = integral[i+1] + 0.5 * (f_approx[i] + f_approx[i+1]) * (x_vals[i+1] - x_vals[i])
            
            x_vals = np.log(a_vals)
            integral = np.zeros_like(a_vals)
            # Backward cumulative trapezoid
            dx = np.diff(x_vals)
            avg_f = 0.5 * (f_approx[:-1] + f_approx[1:])
            steps = avg_f * dx
            # integral[i] is integral from x_i to 0
            # integral[N-1] = 0
            # integral[N-2] = steps[N-2]
            # integral[N-3] = steps[N-3] + integral[N-2]
            # So integral = cumsum(steps)[::-1][::-1]?
            # cumsum(steps) gives integral from x_0 to x_i.
            # We want integral from x_i to x_max (0).
            # Total = integral from x_0 to x_max.
            # integral_i = Total - integral from x_0 to x_i.
            
            cum_int = np.concatenate(([0], np.cumsum(steps)))
            total_int = cum_int[-1]
            integral = total_int - cum_int
            
            D_approx = np.exp(-integral)
            
            # Relative error in D
            err_D = np.abs(D_approx - D_t) / np.abs(D_t)
            
            # Max error for this model
            model_err = max(np.max(err_f), np.max(err_D))
            if model_err > max_err:
                max_err = model_err
                
        return max_err * 100.0 + gamma_violation

    # Initial guess
    # Constant gamma ~ 0.55
    # c0=0.55, others 0
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    start_time = time.time()
    best_x = x0
    best_score = objective(x0)
    
    # Use Nelder-Mead for robustness
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.success or result.fun < best_score:
        best_x = result.x
        best_score = result.fun
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < 60:
        # Try L-BFGS-B for finer polish
        result2 = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        if result2.fun < best_score:
            best_x = result2.x
            best_score = result2.fun
            
    # Ensure coefficients are finite and in range
    # The problem states coefficients in [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)

# EVOLVE-BLOCK-END
