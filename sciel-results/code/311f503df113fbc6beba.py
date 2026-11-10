import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit

    # Helper functions for CPL cosmology
    om0 = 0.3
    
    def get_e2(a, w0, wa):
        """Calculate E(a)^2 for CPL dark energy."""
        # Standard CPL formula:
        # rho_de/rho_m0 = a^(-3(1+w0+wa)) * exp(3wa(1-a))
        # rho_m/rho_m0 = a^(-3)
        # E^2 = Om * a^-3 + Ol * a^(-3(1+w0+wa)) * exp(3wa(1-a))
        a = np.asarray(a)
        if np.isscalar(a) or a.ndim == 0:
            a = np.array([a])
            scalar_input = True
        else:
            scalar_input = False
            
        x = 3 * wa * (1 - a)
        rho_de = a**(-3 * (1 + w0 + wa)) * np.exp(x)
        rho_m = a**(-3)
        e2 = om0 * rho_m + (1 - om0) * rho_de
        
        if scalar_input:
            return e2[0]
        return e2

    def get_om_m(a, w0, wa):
        """Calculate Omega_m(a)."""
        a = np.asarray(a)
        if np.isscalar(a) or a.ndim == 0:
            a = np.array([a])
            scalar_input = True
        else:
            scalar_input = False
            
        e2_val = get_e2(a, w0, wa)
        rho_m = a**(-3)
        om_m = om0 * rho_m / e2_val
        
        if scalar_input:
            return om_m[0]
        return om_m

    # 1. Define Validation Cosmologies
    # Representative CPL parameters
    w0_vals = [-1.0, -0.5, 0.0, 0.5]
    wa_vals = [-0.5, 0.0, 0.5]
    
    # Scale factors for evaluation
    a_grid = np.linspace(0.1, 1.0, 20)

    # Pre-compute reference data
    # We need f_exact(a) and D_exact(a) for each model
    ref_data = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            if time.time() - start_time > deadline:
                break
                
            def ode_f(t, y):
                a = np.exp(t)
                f = y[0]
                om = get_om_m(a, w0, wa)
                ol = 1 - om
                w = w0 + wa * (1 - a)
                h_prime_h = 0.5 * om - 0.5 * ol * (1 + 3 * w)
                return [f**2 - (2 + h_prime_h) * f - 1.5 * om]

            # Integrate from a=0.01 (ln a ~ -4.6) to a=1 (ln a = 0)
            # At a=0.01, f ~ 1
            t_span = (np.log(0.01), 0.0)
            try:
                sol = solve_ivp(ode_f, t_span, [1.0], t_eval=np.log(a_grid), method='RK45', rtol=1e-8, atol=1e-10)
            except Exception:
                continue
            
            if not sol.success:
                continue

            f_exact = sol.y[0]
            
            # Calculate D_exact
            # D(a) = exp( - integral_ln(a)^0 f(exp(x)) dx )
            # We have f(t) at t_grid.
            # We need int_{t_i}^0 f(t) dt.
            # Use trapezoidal rule on the sol.t, sol.y arrays.
            dC = np.zeros_like(sol.t)
            dC[1:] = 0.5 * (sol.y[0,1:] + sol.y[0,:-1]) * (sol.t[1:] - sol.t[:-1])
            C = np.cumsum(dC)
            
            D_exact = np.zeros_like(a_grid)
            for i in range(len(a_grid)):
                integral_val = C[-1] - C[i]
                D_exact[i] = np.exp(-integral_val)
                
            ref_data.append((w0, wa, a_grid, f_exact, D_exact))

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 2. Define Approximation
    # Strategy: Fix c1=0, c2=0. Optimize c0, c3, c4, c5.
    # gamma = c0 + c3*u + c4*wa*a + c5*u*q
    # q = 1 - Om(a)
    # u = 1 + w0 + wa*(1-a)
    
    def approx_growth(c, w0, wa, a):
        c0, c1, c2, c3, c4, c5 = c
        # Enforce c1=0, c2=0 as per strategy
        c1 = 0.0
        c2 = 0.0
        
        a = np.asarray(a)
        om_m = get_om_m(a, w0, wa)
        q = 1 - om_m
        u = 1 + w0 + wa * (1 - a)
        
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
        
        # Constraint: gamma in [0.1, 1.2]
        # If violated, we return a large error later, but for now calculate f
        # We clip gamma for calculation to avoid negative base issues if om_m < 1 (which it is)
        # Actually om_m is always positive.
        
        f_app = om_m ** gamma
        
        # D_app calculation
        # D(a) = exp( - integral_a^1 f(a')/a' da' )
        D_app = np.zeros_like(a)
        
        # Use a fine grid for integration for each a
        # To speed up, we can precompute a dense grid of f_app for the whole range [min(a), 1]
        # But f_app depends on a.
        # Let's just do trapezoidal on a fine sub-grid for each a point.
        # 20 points * 50 sub-points = 1000 evals per model. Fast.
        
        for i, av in enumerate(a):
            if av >= 1.0:
                D_app[i] = 1.0
                continue
            
            # Create fine grid from av to 1
            a_fine = np.linspace(av, 1.0, 50)
            
            # Calculate f_app on fine grid
            om_fine = get_om_m(a_fine, w0, wa)
            q_fine = 1 - om_fine
            u_fine = 1 + w0 + wa * (1 - a_fine)
            
            gamma_fine = c0 + c3 * u_fine + c4 * wa * a_fine + c5 * u_fine * q_fine
            f_fine = om_fine ** gamma_fine
            
            # Integrate f_fine / a_fine da
            val = np.trapz(f_fine / a_fine, a_fine)
            D_app[i] = np.exp(-val)
            
        return f_app, D_app

    def objective(c):
        if time.time() - start_time > deadline:
            return 1000.0
            
        # Enforce c1=0, c2=0 in the vector passed to approx_growth
        # We will pass the full vector but approx_growth ignores c1, c2
        max_err = 0.0
        
        for w0, wa, a, f_e, D_e in ref_data:
            f_app, D_app = approx_growth(c, w0, wa, a)
            
            # Check gamma constraints
            om_m = get_om_m(a, w0, wa)
            q = 1 - om_m
            u = 1 + w0 + wa * (1 - a)
            gamma = c[0] + c[3] * u + c[4] * wa * a + c[5] * u * q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1000.0
            
            # Relative errors
            err_f = np.abs(f_app / f_e - 1.0)
            err_D = np.abs(D_app / D_e - 1.0)
            
            local_max = max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max
                
        return max_err * 100.0

    # Initial guess
    # c0 ~ 0.55 is standard. c3, c4, c5 start at 0.
    # c1, c2 are fixed to 0.
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    # We optimize all 6, but objective forces c1=c2=0.
    # It's better to optimize only 4 variables and map them.
    # Let's define a wrapper for the optimizer to only use 4 variables.
    
    def objective_4d(x):
        c = [x[0], 0.0, 0.0, x[1], x[2], x[3]]
        return objective(c)
    
    x0_4d = [0.55, 0.0, 0.0, 0.0]
    
    try:
        res = minimize(objective_4d, x0_4d, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        best_x = res.x
        best_val = res.fun
        
        # Polish with Powell
        if time.time() - start_time < deadline - 10:
            res2 = minimize(objective_4d, best_x, method='Powell', options={'maxiter': 100, 'xrtol': 1e-6})
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
    except Exception:
        best_x = x0_4d
        best_val = 1000.0

    # Construct final 6-coefficient vector
    best_c = [best_x[0], 0.0, 0.0, best_x[1], best_x[2], best_x[3]]
    
    # Clamp coefficients to [-10, 10]
    best_c = [np.clip(val, -10, 10) for val in best_c]
    
    return best_c

# EVOLVE-BLOCK-END
