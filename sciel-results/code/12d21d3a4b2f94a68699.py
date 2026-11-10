import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import time

# Global cache for reference data to avoid recomputation if construct is called repeatedly
# Note: In a single evaluation run, construct is called once. But good practice.
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
        t_eval_desc = x_eval[::-1]
        
        sol = solve_ivp(ode_f, [0.0, x_min], [f_init], t_eval=t_eval_desc, method='RK45', rtol=1e-9, atol=1e-11)
        
        if sol.success:
            f_ref[i, :] = sol.y[0][::-1]
        else:
            f_ref[i, :] = f_init

        # Compute D
        # D(a) = exp( integral f dx )
        dx = np.diff(x_eval)
        integral = np.zeros_like(x_eval)
        integral[1:] = np.cumsum(0.5 * (f_ref[i, 1:] + f_ref[i, :-1]) * dx)
        D_ref[i, :] = np.exp(integral)
        D_ref[i, :] /= D_ref[i, -1] # Normalize D(1)=1

    _REF_CACHE[key] = (f_ref, D_ref)
    return f_ref, D_ref

def construct():
    start_time = time.time()
    # Evaluator kills after 90s. We use 85s for safety.
    deadline = start_time + 85.0
    
    # Define validation models
    # We use a grid that covers the likely parameter space of the "disjoint public validation cosmologies".
    # Standard values for Om0, w0, wa.
    om_m_vals = [0.20, 0.25, 0.30, 0.35, 0.40]
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    wa_vals = [-0.4, -0.2, 0.0, 0.2, 0.4]
    
    models = []
    for om in om_m_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                # Filter out unphysical or extreme cases if any, though these are generally fine
                models.append((om, w0, wa))
                
    # a_grid: Logarithmic spacing is often better for growth factor integration accuracy
    a_grid = np.geomspace(0.05, 1.0, 40)
    
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
        
    # Flatten arrays for faster vectorized operations in objective
    # We will compute gamma as a flat array, then reshape if needed, but vectorized ops on 2D are fine.
    
    def objective(c):
        if time.time() > deadline:
            return 1e6 # Large penalty if time is up
            
        c0, c1, c2, c3, c4, c5 = c
        
        # Vectorized computation of gamma for all models and a
        gamma = (c0 + c1 * q_arr + c2 * q2_arr + c3 * u_arr + c4 * wa_a_arr + c5 * u_q_arr)
        
        # Check bounds: Gamma must lie in [0.1, 1.2]
        # Note: The task says "Gamma must lie in [0.1, 1.2] at all checked locations".
        # If violated, we return a penalty.
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6 # Penalty for constraint violation
            
        f_app = om_m_arr ** gamma
        
        # Compute D_app for all models
        # Integral of f_app dx
        f_diff = 0.5 * (f_app[:, 1:] + f_app[:, :-1]) * dx[np.newaxis, :]
        
        integral = np.zeros_like(f_app)
        integral[:, 1:] = np.cumsum(f_diff, axis=1)
        
        D_app = np.exp(integral)
        D_app = D_app / D_app[:, -1:] # Normalize D(1)=1
        
        # Calculate errors
        # f error
        err_f = np.abs(f_app - f_ref) / f_ref
        # D error
        err_D = np.abs(D_app - D_ref) / D_ref
        
        # Max error over all models and a
        max_err = np.max(np.concatenate([err_f.ravel(), err_D.ravel()]))
        
        return max_err

    # Initial guess / Bounds for Differential Evolution
    # c0 is the constant gamma, typically ~0.55
    # Other coefficients adjust for deviations.
    # We set bounds to [-10, 10] as per task, but DE works best with tighter initial bounds if possible.
    # However, the task requires coefficients in [-10, 10].
    bounds = [(-10, 10)] * 6
    
    # To improve convergence speed of DE, we can use a smaller population and mutation strategy.
    # 'best1bin' is a good default.
    
    # We will run DE with a time limit.
    # DE in scipy doesn't have a direct time limit, but we can monitor iterations or use a callback.
    # However, scipy's DE can be slow.
    # Alternative: Use `minimize` with `method='Powell'` from multiple starts, but DE is more robust globally.
    # Let's try to run DE with a moderate population size.
    
    # Since we have ~125 models * 40 points = 5000 data points, the objective is relatively fast (vectorized numpy).
    # Each objective eval takes ~ms.
    # DE with popsize=15 (default) for 6 dims -> 90 individuals.
    # maxiter=100 -> 9000 evals. Should be fast enough.
    
    try:
        # Use a seed for reproducibility in case of multiple runs, but randomness helps escape local minima.
        # We'll use a fixed seed for the initial population to be deterministic if needed, 
        # but the task implies finding the best possible.
        
        # We'll run DE.
        # To be safe with time, we can limit maxiter.
        # If we have ~80s, and each eval is ~1ms, we can afford ~10,000 evals.
        # DE with popsize=10, maxiter=100 -> 1000 * 10 = 1000 evals? No, popsize is number of individuals.
        # Total evals = popsize * maxiter.
        # Let's use popsize=15, maxiter=200. 3000 evals.
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-8,
            seed=42,
            polish=True  # Use L-BFGS-B for final polishing
        )
        
        best_c = res.x
        best_score = res.fun
        
    except Exception as e:
        # Fallback to a simpler local optimization if DE fails or times out
        print(f"DE failed: {e}")
        
        # Fallback: Powell from a few starts
        best_c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        best_score = objective(best_c)
        
        from scipy.optimize import minimize
        rng = np.random.default_rng(42)
        start_points = [best_c]
        for _ in range(10):
            p = [
                rng.uniform(0.3, 0.8),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0)
            ]
            start_points.append(p)
            
        for sp in start_points:
            if time.time() > deadline - 2.0:
                break
            try:
                res = minimize(objective, sp, method='Powell', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_c = res.x
            except Exception:
                pass

    # Final check: Ensure the returned coefficients are within [-10, 10]
    # DE respects bounds, but let's clip just in case of numerical drift
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # One final polish with a tighter local optimizer to squeeze out last bit of error
    # Using Nelder-Mead or Powell for a final local step
    from scipy.optimize import minimize
    try:
        if time.time() < deadline - 1.0:
            res_final = minimize(
                objective, 
                best_c, 
                method='Powell', 
                bounds=bounds, 
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            if res_final.fun < best_score:
                best_c = res_final.x
                best_score = res_final.fun
    except Exception:
        pass

    return [float(x) for x in best_c]
