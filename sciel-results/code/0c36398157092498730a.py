"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a = np.asarray(a, dtype=float)
        Om_de = 1.0 - Om0
        # CPL dark energy density factor: a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return (Om0/a**3) / E2

    def E_of_a(a, Om0, w0, wa):
        """H(a)/H0 for CPL dark energy."""
        a = np.asarray(a, dtype=float)
        Om_de = 1.0 - Om0
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return np.sqrt(E2)

    def compute_reference(Om0, w0, wa, a_grid):
        """Compute reference f and D/D(1) using GR ODE integrated from early times."""
        n = len(a_grid)
        
        def ode(x, y):
            a = np.exp(x)
            E = E_of_a(a, Om0, w0, wa)
            Om_m = Om_of_a(a, Om0, w0, wa)
            # Growth equation in x=ln(a): D'' + 2*E(a)*D' - 1.5*Om_m(a)*D = 0
            dD = y[1]
            d2D = -2.0*E*dD + 1.5*Om_m*y[0]
            return [dD, d2D]
        
        x_min = np.log(a_grid[0])
        x_max = 0.0
        
        # Initial conditions at early times: D ~ a, D' ~ 1
        a_min = a_grid[0]
        y_init = [a_min, 1.0]
        
        # Use a dense grid for integration
        x_eval = np.linspace(x_min, x_max, 500)
        
        sol = solve_ivp(ode, [x_min, x_max], y_init, 
                       t_eval=x_eval,
                       method='RK45', rtol=1e-10, atol=1e-12)
        
        if not sol.success:
            return None, None
        
        # Interpolate to get values at a_grid
        x_grid = np.log(a_grid)
        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])
        
        # Normalize: D/D(1)
        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        
        # f = D'/D
        f_ref = dD_ref / D_ref
        
        return f_ref, D_ref

    def model_gamma(c, a, Om0, w0, wa):
        """Compute gamma from the 6-coefficient model."""
        q = 1.0 - Om_of_a(a, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma

    def model_f(c, a, Om0, w0, wa):
        """Compute f from the model."""
        gamma = model_gamma(c, a, Om0, w0, wa)
        Om_m = Om_of_a(a, Om0, w0, wa)
        # Clip gamma to [0.1, 1.2]
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_m**gamma_clipped

    def model_D(c, a, Om0, w0, wa):
        """Compute normalized D from the model via quadrature."""
        n = len(a)
        D_norm = np.zeros(n)
        D_norm[-1] = 1.0
        
        f_vals = model_f(c, a, Om0, w0, wa)
        x = np.log(a)
        
        # Cumulative trapezoidal integration from right to left
        dx = np.diff(x)
        trap = 0.5*(f_vals[:-1] + f_vals[1:]) * dx
        cum = np.zeros(n)
        if n > 1:
            cum[-2] = trap[-1]
            for i in range(n-3, -1, -1):
                cum[i] = cum[i+1] + trap[i]
        D_norm[:-1] = np.exp(-cum)
        return D_norm

    # Define validation cosmologies
    cosmologies = [
        (0.3, -0.5, 0.0),    # LCDM
        (0.3, -1.0, 0.0),    # Cosmological constant
        (0.3, -0.9, 0.2),    # CPL with positive wa
        (0.3, -0.9, -0.2),   # CPL with negative wa
        (0.25, -0.8, 0.1),   # Lower matter, moderate wa
        (0.35, -1.0, 0.0),   # Higher matter
        (0.3, -1.1, 0.3),    # More negative w0, larger wa
        (0.28, -0.7, -0.3),  # Less negative w0, negative wa
    ]
    
    a_grid = np.linspace(0.05, 1.0, 80)
    
    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        if elapsed() > DEADLINE - 5:
            break
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)
    
    if not refs:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    if elapsed() > DEADLINE - 3:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def fast_objective(c):
        max_err = 0.0
        for (Om0, w0, wa) in cosmologies:
            if (Om0, w0, wa) not in refs:
                continue
            f_ref, D_ref = refs[(Om0, w0, wa)]
            f_mod = model_f(c, a_grid, Om0, w0, wa)
            D_mod = model_D(c, a_grid, Om0, w0, wa)
            
            # Relative errors
            err_f = np.abs((f_mod - f_ref) / f_ref)
            err_D = np.abs((D_mod - D_ref) / D_ref)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        return max_err
    
    # Start from constant gamma
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_c = c0.copy()
    best_err = fast_objective(best_c)
    
    if elapsed() > DEADLINE - 10:
        return np.clip(best_c, -10, 10).tolist()
    
    # Nelder-Mead optimization
    result = minimize(fast_objective, best_c, method='Nelder-Mead', 
                      options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
    
    if result.fun < best_err:
        best_c = result.x
        best_err = result.fun
    
    if elapsed() > DEADLINE - 5:
        return np.clip(best_c, -10, 10).tolist()
    
    # Try multiple starting points
    starts = [
        np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.05, 0.0, 0.05, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.05, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.05, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.05]),
        np.array([0.55, 0.05, 0.0, 0.0, 0.0, 0.05]),
        np.array([0.5, 0.0, 0.0, 0.05, 0.05, 0.0]),
    ]
    
    for s in starts:
        if elapsed() > DEADLINE - 3:
            break
        res = minimize(fast_objective, s, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        if res.fun < best_err:
            best_c = res.x
            best_err = res.fun
    
    # Final polish with tighter tolerances
    if elapsed() < DEADLINE - 2:
        res = minimize(fast_objective, best_c, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        if res.fun < best_err:
            best_c = res.x
            best_err = res.fun
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return best_c.tolist()
# EVOLVE-BLOCK-END
