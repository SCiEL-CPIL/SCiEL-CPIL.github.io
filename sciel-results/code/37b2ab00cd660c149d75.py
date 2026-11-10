import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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
    
    # Pre-compute x_eval (log a)
    x_eval = np.log(a_grid)
    x_min = x_eval[0]
    
    for i, (Om0, w0, wa) in enumerate(models):
        
        def beta(a_arr):
            """
            beta = Omega_m + 3 Omega_de w
            """
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

        f_init = Om0**0.55
        
        # Integrate from x=0 to x_min (negative)
        # t_eval must be sorted in the direction of integration (descending here)
        # x_eval is ascending (x_min to 0). We need descending (0 to x_min).
        t_eval_desc = x_eval[::-1]
        
        sol = solve_ivp(ode_f, [0.0, x_min], [f_init], t_eval=t_eval_desc, method='RK45', rtol=1e-8, atol=1e-10)
        
        if sol.success:
            # sol.y[0] is in descending order of x (i.e., ascending order of a? No.)
            # x goes from 0 to x_min. a goes from 1 to a_min.
            # a_grid is ascending (a_min to 1).
            # t_eval_desc is [0, ..., x_min] which corresponds to a=[1, ..., a_min].
            # We need f_ref[i] to correspond to a_grid (a_min to 1).
            # So we reverse the solution.
            f_ref[i, :] = sol.y[0][::-1]
        else:
            # Fallback if integration fails
            f_ref[i, :] = f_init

        # Compute D
        # D(a) = exp( integral f dx )
        # We have f at x_eval (ascending).
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
    deadline = start_time + 85.0 # 5s margin
    
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
    
    # Pre-compute terms that depend only on a and model parameters, not on c
    n_models = len(models)
    n_a = len(a_grid)
    
    # Arrays to store pre-computed values
    om_m_arr = np.zeros((n_models, n_a))
    q_arr = np.zeros((n_models, n_a))
    u_arr = np.zeros((n_models, n_a))
    wa_a_arr = np.zeros((n_models, n_a))
    u_q_arr = np.zeros((n_models, n_a))
    q2_arr = np.zeros((n_models, n_a))
    
    log_a = np.log(a_grid)
    dx = np.diff(log_a)
    
    for i, (Om0, w0, wa) in enumerate(models):
        m_term = Om0 * np.exp(-3.0 * log_a)
        exp_term = -3.0 * (1.0 + w0 + wa * (1.0 - 1.0 / a_grid))
        de_term = (1.0 - Om0) * np.exp(exp_term * log_a)
        E2_val = m_term + de_term
        om_m = m_term / E2_val
        
        om_m_arr[i, :] = om_m
        q_arr[i, :] = 1.0 - om_m
        u_arr[i, :] = 1.0 + w0 + wa * (1.0 - a_grid)
        wa_a_arr[i, :] = wa * a_grid
        u_q_arr[i, :] = u_arr[i, :] * q_arr[i, :]
        q2_arr[i, :] = q_arr[i, :] ** 2
        
    def objective(c):
        if time.time() > deadline:
            return 1e6 # Large penalty if time is up
            
        c0, c1, c2, c3, c4, c5 = c
        
        # Vectorized computation of gamma for all models and a
        gamma = (c0 + c1 * q_arr + c2 * q2_arr + c3 * u_arr + c4 * wa_a_arr + c5 * u_q_arr)
        
        # Check bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6 # Penalty for constraint violation
            
        f_app = om_m_arr ** gamma
        
        # Compute D_app for all models
        f_diff = 0.5 * (f_app[:, 1:] + f_app[:, :-1]) * dx[np.newaxis, :]
        
        integral = np.zeros_like(f_app)
        integral[:, 1:] = np.cumsum(f_diff, axis=1)
        
        D_app = np.exp(integral)
        D_app = D_app / D_app[:, -1:] # Normalize D(1)=1
        
        # Calculate errors
        err_f = np.abs(f_app - f_ref) / f_ref
        err_D = np.abs(D_app - D_ref) / D_ref
        
        max_err = np.max(np.concatenate([err_f.ravel(), err_D.ravel()]))
        
        return max_err

    # Initial guess based on asymptotic matching
    # At a->0: gamma ~ 6/11. q~1, u~1+w0+wa, wa*a~0, u*q~1+w0+wa
    # At a->1: gamma ~ 0.55. q~q0, u~1+w0, wa*a~wa, u*q~(1+w0)q0
    # We can use a representative model to get a better initial guess.
    # Let's use the LambdaCDM model (Om=0.3, w0=-1, wa=0)
    # q0 = 0.7, u0 = 0, wa*a = 0, u*q = 0
    # gamma(a->1) = c0 + c1*0.7 + c2*0.49 = 0.55
    # gamma(a->0) = c0 + c1 + c2 + c3*0 + c4*0 + c5*0 = 6/11
    # This is underdetermined. We'll use a multi-start approach.
    
    x0_candidates = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0], # Seed
        [6.0/11.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Low a limit
        [0.55, -0.1, 0.1, 0.0, 0.0, 0.0], # Perturbation
        [0.50, 0.05, 0.0, 0.0, 0.0, 0.0]
    ]
    
    best_c = x0_candidates[0]
    best_score = objective(best_c)
    
    # Multi-start local optimization
    bounds = [(-10, 10)] * 6
    
    for x0 in x0_candidates:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 2:
        try:
            res2 = minimize(objective, best_c, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_c = res2.x
        except Exception:
            pass
            
    return [float(x) for x in best_c]
