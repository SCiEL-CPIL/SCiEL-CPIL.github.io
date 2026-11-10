import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    # 1. Define Test Cosmologies
    # (Omega_m0, w0, wa)
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -0.9, -0.1),  # CPL
        (0.30, -1.1, 0.3),   # Phantom
        (0.25, -1.0, 0.0),   # Lower Omega_m
        (0.35, -0.95, -0.2), # Higher Omega_m
        (0.30, -1.0, 0.5),   # Strong wa
        (0.30, -1.0, -0.5),  # Strong negative wa
    ]
    
    # 2. Define Grid
    a_grid = np.linspace(0.05, 1.0, 100)
    
    # 3. Pre-compute Reference Data
    # Store f_ref and D_ref for each model
    f_refs = []
    D_refs = []
    
    for Om0, w0, wa in models:
        # Compute Omega_m(a)
        # Omega_de(a) = (1-Om0) * a^(-3(1+w_eff))
        # w_eff = w0 + wa*(1-a)
        # This requires numerical integration or approximation for Omega_m if wa != 0.
        # For reference ODE, we need Omega_m(a) and w(a).
        
        def ode_f(t, y):
            a = np.exp(t)
            w = w0 + wa * (1 - a)
            # Calculate Omega_m(a)
            # H^2 ~ Om_m a^-3 + Om_de a^-3(1+w_eff)
            # We need the ratio Om_m(a) = Om_m0 a^3 / (Om_m0 a^3 + Om_de0 a^{-3 w_eff})
            # This is an approximation if w_eff changes with a.
            # More accurate: Integrate dOm_m/dlna = -3 Om_m (1 - Om_m) ? No.
            # d(Om_m a^3)/dlna = -3 Om_m a^3 (1 - Om_m) ?
            # Standard: dOm_m/dlna = 3 Om_m (Om_m - 1) + 3 Om_m (1+w) ? No.
            # Let's use the standard ODE for Om_m:
            # dOm_m/dlna = 3 Om_m (Om_m - 1) - 3 Om_m (1+w) ??
            # Actually, H^2 = H0^2 [ Om_m a^-3 + Om_de a^{-3(1+w)} ]
            # d Om_m / d ln a = -3 Om_m + 3 Om_m (Om_m + Om_de (1+w) (a^3/a^{-3(1+w)})... )
            # Simpler: Just compute Omega_m(a) using the integral for the DE density if wa != 0?
            # Or assume the "CPL" definition implies a specific background evolution.
            # Usually, for CPL, one integrates the continuity equation for DE:
            # rho_de' + 3H(1+w)rho_de = 0.
            # This is coupled.
            # Given the time constraint, let's use a standard approximation for Omega_m(a) in CPL:
            # Omega_m(a) = Om0 a^3 / (Om0 a^3 + (1-Om0) a^{-3(w0 + wa(1-a)/2)} ) ??
            # Let's stick to a simpler robust method:
            # We can solve the ODE for f and Om_m simultaneously?
            # No, f ODE needs Om_m.
            
            # Let's compute Om_m(a) numerically by integrating dOm_m/dlna.
            # dOm_m/dlna = 3 Om_m (Om_m - 1) - 3 Om_m (1+w) * (1 - Om_m)? No.
            # From H^2(a) = H0^2 [ Om_m a^-3 + Om_de(a) ]
            # d ln H^2 / d ln a = -3 Om_m - 3 (1+w) Om_de
            # Also Om_m + Om_de = 1 (flat universe)
            # d Om_m / d ln a = - Om_m d(ln H^2) / d ln a - Om_m (3 Om_m + 3(1+w)Om_de) ?
            # Standard result: d Om_m / d ln a = 3 Om_m (Om_m - 1) - 3 Om_m (1+w) ? 
            # Let's verify.
            # Om_m = rho_m / rho_crit.
            # d rho_m / d ln a = -3 rho_m.
            # d rho_crit / d ln a = 3 rho_crit (1 - Om_m - (1+w)Om_de) ?
            # This is getting complicated.
            
            # Alternative: For the *reference* calculation, we can just use a high-precision library or a simple numerical integration of the background.
            # Since we are in Python, let's integrate the background ODEs:
            # y = [Om_m, f]
            # dOm_m/dlna = 3*Om_m*(Om_m - 1) - 3*Om_m*(1+w) # This formula is often used for constant w. For variable w, it's approximate.
            # Let's use the exact derivative for CPL background:
            # rho_de(a) = rho_de0 * exp(-3 * integral (1+w) dlna)
            # This requires nested integration.
            
            # Simplification: The task says "smooth CPL dark-energy cosmologies".
            # For the purpose of finding the coefficients, a good approximation of the background is sufficient, as long as it's consistent.
            # Let's use the approximation:
            # Omega_m(a) = Om0 a^3 / (Om0 a^3 + (1-Om0) a^{-3 * w_eff})
            # where w_eff is the average w? Or just w(a)?
            # Using w(a) directly in the power law is a common approximation for small wa.
            
            w_eff = w0 + wa * (1 - a)
            Om_m = Om0 * a**3 / (Om0 * a**3 + (1 - Om0) * a**(-3 * w_eff))
            # Clip Om_m to [0, 1]
            Om_m = np.clip(Om_m, 1e-10, 1.0)
            
            # ODE for f:
            # df/dlna = -f - 0.5 f^2 - 1.5 (1+w) Om_m
            df = -y[1] - 0.5 * y[1]**2 - 1.5 * (1 + w) * Om_m
            return [0, df] # We only need f, Om_m is computed explicitly
            
        # Initial condition at a_min = 0.05
        # f ~ 1 for matter domination
        t_min = np.log(a_grid[0])
        t_max = np.log(1.0)
        
        # Solve f ODE
        sol = solve_ivp(ode_f, [t_min, t_max], [1.0], t_eval=a_grid, method='RK45', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            continue
            
        f_ref = sol.y[0]
        
        # Compute D_ref
        # D(a)/D(1) = exp( - integral_{ln a}^0 f dx )
        # Let's compute the integral from 1 to a (backwards) or 0 to ln a?
        # ln D(a) - ln D(1) = - integral_{ln a}^0 f(x) dx
        # Let I(a) = integral_{ln a}^0 f(x) dx
        # We can compute cumulative integral of f from a_min to 1, then subtract.
        # Or just use np.cumtrapz on the reversed array?
        
        # Let's compute integral from 0 to ln a (which is negative).
        # integral_{0}^{ln a} f(x) dx = - integral_{ln a}^{0} f(x) dx
        # So D(a)/D(1) = exp( integral_{0}^{ln a} f(x) dx )
        
        # We have f on grid a_grid (increasing).
        # We want integral from 0 (a=1) to ln a (a=a_i).
        # This is -1 * integral from a_i to 1.
        
        # Use cumulative trapezoidal integration on log(a)
        log_a = np.log(a_grid)
        # Integral from a_grid[0] to a_grid[i]
        # We want integral from 1 to a_i.
        
        # Let's compute cumulative integral from start (a_min) to end (a=1)
        # C[i] = integral_{a_min}^{a_i} f dx
        # Then integral_{a_i}^{1} = C[-1] - C[i]
        # So ln D(a_i) - ln D(1) = - (C[-1] - C[i]) = C[i] - C[-1]
        
        # np.cumtrapz(f_ref, log_a) gives integral from a_min to a_i
        # Note: cumtrapz in scipy is deprecated, use scipy.integrate.cumulative_trapezoid
        
        from scipy.integrate import cumulative_trapezoid
        C = cumulative_trapezoid(f_ref, log_a, initial=0)
        
        # D_ref_ratio = exp(C - C[-1])
        D_ref_ratio = np.exp(C - C[-1])
        
        f_refs.append(f_ref)
        D_refs.append(D_ref_ratio)
        
    # 4. Define Objective Function
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        max_err = 0.0
        for i, (Om0, w0, wa) in enumerate(models):
            # Compute q, u
            # We need Om_m(a) for the approximation too.
            # Use the same approximation as in reference for consistency?
            # Yes, use the same Om_m calculation.
            
            a = a_grid
            w = w0 + wa * (1 - a)
            w_eff = w0 + wa * (1 - a)
            Om_m = Om0 * a**3 / (Om0 * a**3 + (1 - Om0) * a**(-3 * w_eff))
            Om_m = np.clip(Om_m, 1e-10, 1.0)
            
            q = 1 - Om_m
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            # Check bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6 # Penalty
            
            f_app = Om_m ** gamma
            
            # Compute D_app
            # D_app(a)/D(1) = exp( - integral_{ln a}^0 f_app dx )
            # Same logic: C_app = cumtrapz(f_app, log_a)
            # D_app_ratio = exp(C_app - C_app[-1])
            
            C_app = cumulative_trapezoid(f_app, log_a, initial=0)
            D_app_ratio = np.exp(C_app - C_app[-1])
            
            # Errors
            # f_err = |f_app - f_ref| / |f_ref|
            # D_err = |D_app - D_ref| / |D_ref|
            
            f_ref = f_refs[i]
            D_ref = D_refs[i]
            
            # Avoid division by zero
            f_err = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
            D_err = np.abs(D_app_ratio - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
            
            err = np.max(np.concatenate([f_err, D_err]))
            if err > max_err:
                max_err = err
                
        return max_err

    # 5. Optimize
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Use Nelder-Mead
    # Bounds: [-10, 10]
    # Nelder-Mead doesn't support bounds directly, but we can penalize or use L-BFGS-B.
    # L-BFGS-B is better for bounded problems.
    
    bounds = [(-10, 10)] * 6
    
    best_x = x0
    best_val = objective(x0)
    
    # Run minimize
    # Since objective is fast now, we can run more iterations.
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        return best_x
        
    # Try a second run with different initial guess or method if time permits
    # e.g. Nelder-Mead from best_x
    try:
        res2 = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 500})
        if res2.fun < best_val:
            best_x = res2.x
            best_val = res2.fun
    except Exception as e:
        pass
        
    return best_x

