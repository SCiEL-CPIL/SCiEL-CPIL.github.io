import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import time

# Global cache for reference data
_REF_CACHE = {}

def _get_reference_data(models, a_grid):
    """
    Pre-compute reference f and D for given models and a_grid.
    Returns arrays of shape (n_models, n_a)
    """
    key = (tuple(models), tuple(a_grid))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
        
    n_models = len(models)
    n_a = len(a_grid)
    f_ref = np.zeros((n_models, n_a))
    D_ref = np.zeros((n_models, n_a))
    
    for i, (Om0, w0, wa) in enumerate(models):
        # Define E^2(a)
        def E2(a):
            # Omega_m(a) * a^3
            om_m_term = Om0
            # Omega_de(a) * a^3
            # rho_de ~ a^-3(1+w0+wa(1-1/a))
            # rho_de0 = 1 - Om0
            # rho_de(a) = rho_de0 * a^-3(1+w0+wa(1-1/a))
            # term in E^2 is rho_de(a) * a^3 = rho_de0 * a^-3*wa*(1-1/a)
            # Let's be careful.
            # E^2 = Om0 a^-3 + (1-Om0) a^-3(1+w0+wa(1-1/a))
            # E^2 = a^-3 [ Om0 + (1-Om0) a^(-wa(1-1/a)) ] ? No.
            # a^-3(1+w0+wa(1-1/a)) = a^(-3-3w0-3wa+3wa/a)
            # This is hard to vectorize simply if we just use the formula.
            # Let's use the explicit form.
            a_arr = np.atleast_1d(a)
            exp_term = -3.0 * (1.0 + w0 + wa * (1.0 - 1.0 / a_arr))
            de_term = (1.0 - Om0) * np.exp(exp_term * np.log(a_arr))
            m_term = Om0 * np.exp(-3.0 * np.log(a_arr))
            return m_term + de_term

        def beta(a):
            """
            beta = Omega_m + 3 Omega_de w
            """
            a_arr = np.atleast_1d(a)
            log_a = np.log(a_arr)
            m_term = Om0 * np.exp(-3.0 * log_a)
            exp_term = -3.0 * (1.0 + w0 + wa * (1.0 - 1.0 / a_arr))
            de_term = (1.0 - Om0) * np.exp(exp_term * log_a)
            E2_val = m_term + de_term
            om_m = m_term / E2_val
            om_de = de_term / E2_val
            w_val = w0 + wa * (1.0 - 1.0 / a_arr)
            return om_m + 3.0 * om_de * w_val

        def ode_f(x, y):
            a = np.exp(x)
            f_val = y[0]
            b = beta(a)
            return [0.5 * f_val * (1.0 - f_val) - 1.5 * f_val * b]

        # Integrate from a=1 (x=0) to a=a_min (x=ln(a_min))
        # We want f at a_grid.
        # a_grid is sorted ascending. x_grid = ln(a_grid) is sorted ascending.
        # ODE is forward in x.
        # We need f(1) as initial condition.
        # f(1) approx Om0**0.55
        f_init = Om0**0.55
        
        # We integrate from x=0 to x_min.
        # But we need values at specific x points.
        # solve_ivp can handle this.
        x_min = np.log(a_grid[0])
        x_eval = np.log(a_grid)
        
        # We need to integrate from 0 to x_min? No, x_min is negative.
        # So we integrate from x_min to 0?
        # The ODE is autonomous in x? No, beta depends on a=exp(x).
        # It is not symmetric.
        # We should integrate from x_min to 0, and reverse the result.
        # Or integrate from 0 to x_min (backwards).
        # solve_ivp supports t_span = (0, x_min) if x_min < 0.
        
        sol = solve_ivp(ode_f, [0.0, x_min], [f_init], t_eval=x_eval, method='RK45', rtol=1e-8, atol=1e-10)
        
        if sol.success:
            f_ref[i, :] = sol.y[0]
        else:
            # Fallback if integration fails
            f_ref[i, :] = f_init

        # Compute D
        # D(a) = exp( integral f dx )
        # We have f at x_eval.
        # Integrate f from x_eval[0] to x_eval[i]
        # Use cumulative trapezoid
        dx = np.diff(x_eval)
        integral = np.zeros_like(x_eval)
        integral[1:] = np.cumsum(0.5 * (f_ref[i, 1:] + f_ref[i, :-1]) * dx)
        D_ref[i, :] = np.exp(integral)
        D_ref[i, :] /= D_ref[i, -1] # Normalize D(1)=1

    _REF_CACHE[key] = (f_ref, D_ref)
    return f_ref, D_ref

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin
    
    # Define validation models
    # Typical range
    om_m_vals = [0.25, 0.30, 0.35]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.2, 0.0, 0.2]
    
    models = []
    for om in om_m_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                models.append((om, w0, wa))
                
    a_grid = np.linspace(0.05, 1.0, 50)
    
    f_ref, D_ref = _get_reference_data(models, a_grid)
    
    def objective(c):
        if time.time() > deadline:
            return 1e6 # Large penalty if time is up
            
        c0, c1, c2, c3, c4, c5 = c
        
        max_err = 0.0
        gamma_violation = False
        
        for i, (Om0, w0, wa) in enumerate(models):
            a = a_grid
            # Compute q, u
            # Need Omega_m(a)
            log_a = np.log(a)
            m_term = Om0 * np.exp(-3.0 * log_a)
            exp_term = -3.0 * (1.0 + w0 + wa * (1.0 - 1.0 / a))
            de_term = (1.0 - Om0) * np.exp(exp_term * log_a)
            E2_val = m_term + de_term
            om_m = m_term / E2_val
            
            q = 1.0 - om_m
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            # Check bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6 # Penalty for constraint violation
                
            f_app = om_m ** gamma
            
            # Compute D_app
            # D_app(a) = exp( integral f_app dx )
            dx = np.diff(np.log(a))
            integral = np.zeros_like(a)
            integral[1:] = np.cumsum(0.5 * (f_app[1:] + f_app[:-1]) * dx)
            D_app = np.exp(integral)
            D_app /= D_app[-1]
            
            # Calculate errors
            # f error
            err_f = np.abs(f_app - f_ref[i, :]) / f_ref[i, :]
            # D error
            err_D = np.abs(D_app - D_ref[i, :]) / D_ref[i, :]
            
            curr_max = np.max(np.concatenate([err_f, err_D]))
            if curr_max > max_err:
                max_err = curr_max
                
        return max_err

    # Initial guess
    # c0 ~ 0.55, others 0
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Differential Evolution
    bounds = [(-10, 10)] * 6
    best_c = x0
    best_score = objective(x0)
    
    if time.time() < deadline - 5:
        try:
            res = differential_evolution(objective, bounds, seed=42, maxiter=100, popsize=10, polish=False, tol=1e-4)
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Polish with L-BFGS-B
    if time.time() < deadline - 2:
        try:
            res2 = minimize(objective, best_c, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
            if res2.fun < best_score:
                best_score = res2.fun
                best_c = res2.x
        except Exception:
            pass
            
    return [float(x) for x in best_c]

