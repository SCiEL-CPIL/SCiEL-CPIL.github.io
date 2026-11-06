from scipy.optimize import minimize, differential_evolution
import numpy as np
from scipy.integrate import solve_ivp
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin
    
    # 1. Define the grid of cosmologies and scale factors
    # We want to cover a reasonable range of parameters likely used in validation.
    om_m0_vals = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    w0_vals = np.array([-1.5, -1.0, -0.5, 0.0])
    wa_vals = np.array([-0.5, 0.0, 0.5])
    a_vals = np.geomspace(0.01, 1.0, 40)
    
    # Pre-compute reference data
    # We will store f_ref and D_ref for each point
    f_refs = []
    D_refs = []
    
    for om_m0 in om_m0_vals:
        om_de0 = 1.0 - om_m0
        for w0 in w0_vals:
            for wa in wa_vals:
                # Define H^2 and its derivative
                def E2(a):
                    term_m = om_m0 * a**(-3)
                    term_de = om_de0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
                    return term_m + term_de
                
                def E2_prime(a):
                    term_m = om_m0 * a**(-3)
                    term_de = om_de0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
                    d_term_m = -3 * term_m / a
                    d_term_de = term_de * (-3*(1+w0+wa)/a - 3*wa)
                    return d_term_m + d_term_de

                def ode(t, y):
                    a = t
                    d = y[0]
                    dp = y[1]
                    e2 = E2(a)
                    e2_p = E2_prime(a)
                    h_prime_over_h = e2_p / (2 * e2)
                    om_m_a = om_m0 * a**(-3) / e2
                    dpp = -(3 + h_prime_over_h) * dp + 1.5 * om_m_a * d
                    return [dp, dpp]
                
                a_start = 0.01
                a_end = 1.0
                y0 = [a_start, 1.0]
                
                try:
                    sol = solve_ivp(ode, [a_start, a_end], y0, method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)
                    if not sol.success:
                        continue
                    
                    D_1 = sol.y[0, -1]
                    D_norm = sol.y[0] / D_1
                    D_prime_norm = sol.y[1] / D_1
                    f_ref = D_prime_norm / D_norm
                    D_ref = D_norm
                    
                    for i, a in enumerate(a_vals):
                        f_refs.append(f_ref[i])
                        D_refs.append(D_ref[i])
                except Exception:
                    continue
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    f_refs = np.array(f_refs)
    D_refs = np.array(D_refs)
    
    # Prepare cosmology parameters for vectorized evaluation
    # We create arrays of shape (N_cosmos, N_a)
    N_cosmos = len(om_m0_vals) * len(w0_vals) * len(wa_vals)
    om_m0_arr = np.repeat(om_m0_vals, len(w0_vals)*len(wa_vals))
    w0_arr = np.tile(np.repeat(w0_vals, len(wa_vals)), len(om_m0_vals))
    wa_arr = np.tile(wa_vals, len(om_m0_vals)*len(w0_vals))
    
    # Precompute Om_m(a) for all cosmologies and a values
    # om_m_a[c, i] = Om_m(a_i) for cosmology c
    om_m_a_grid = np.zeros((N_cosmos, len(a_vals)))
    for c in range(N_cosmos):
        om_m0 = om_m0_arr[c]
        w0 = w0_arr[c]
        wa = wa_arr[c]
        om_de0 = 1.0 - om_m0
        term_m = om_m0 * a_vals**(-3)
        term_de = om_de0 * a_vals**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a_vals))
        e2 = term_m + term_de
        om_m_a_grid[c] = term_m / e2
        
    # Precompute q and u
    q_grid = 1 - om_m_a_grid
    u_grid = 1 + w0_arr[:, None] + wa_arr[:, None] * (1 - a_vals[None, :])
    wa_a_grid = wa_arr[:, None] * a_vals[None, :]
    uq_grid = u_grid * q_grid

    def objective(c):
        c = np.asarray(c)
        if np.any(c < -10) or np.any(c > 10):
            return 100.0
            
        # Vectorized gamma calculation
        gamma = c[0] + c[1]*q_grid + c[2]*q_grid**2 + c[3]*u_grid + c[4]*wa_a_grid + c[5]*uq_grid
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 100.0 + np.max(np.abs(gamma - np.clip(gamma, 0.1, 1.2)))
            
        f_app = om_m_a_grid ** gamma
        
        # Calculate D_app
        # Integral from a_i to 1
        # cum_int[c, i] = \int_{a_0}^{a_i} f_app[c, :] da
        # We need \int_{a_i}^{1} = cum_int[c, -1] - cum_int[c, i]
        
        # Use cumulative trapezoid
        # scipy.integrate.cumulative_trapezoid works on 1D. We need to apply along axis=1.
        # We can do this by looping over cosmologies or using np.apply_along_axis, but loop is faster for small N.
        # N_cosmos is ~84, N_a is 40. 84*40 is small.
        
        int_a_to_1 = np.zeros_like(f_app)
        for c in range(N_cosmos):
            # f_app[c] is 1D
            # cum_int = cumulative_trapezoid(f_app[c], a_vals, initial=0)
            # int_a_to_1[c] = cum_int[-1] - cum_int
            # Manual trapz for speed and no import overhead inside loop
            cum_int = np.zeros_like(a_vals)
            for i in range(1, len(a_vals)):
                cum_int[i] = cum_int[i-1] + 0.5 * (f_app[c, i] + f_app[c, i-1]) * (a_vals[i] - a_vals[i-1])
            int_a_to_1[c] = cum_int[-1] - cum_int
            
        D_app = np.exp(-int_a_to_1)
        
        # Get reference values
        # f_refs and D_refs are flattened. We need to reshape or slice.
        # They are stored in order: for each cosmology, for each a.
        # So f_refs[c * len(a_vals) : (c+1) * len(a_vals)]
        
        max_err = 0.0
        for c in range(N_cosmos):
            start_idx = c * len(a_vals)
            end_idx = start_idx + len(a_vals)
            
            f_ref_slice = f_refs[start_idx:end_idx]
            D_ref_slice = D_refs[start_idx:end_idx]
            
            # Avoid division by zero
            mask_f = f_ref_slice != 0
            mask_d = D_ref_slice != 0
            
            if np.any(mask_f):
                err_f = np.abs(f_app[c][mask_f] - f_ref_slice[mask_f]) / f_ref_slice[mask_f]
                max_err = max(max_err, np.max(err_f))
                
            if np.any(mask_d):
                err_D = np.abs(D_app[c][mask_d] - D_ref_slice[mask_d]) / D_ref_slice[mask_d]
                max_err = max(max_err, np.max(err_D))
                
            if time.time() > deadline:
                return max_err * 100.0
                
        return max_err * 100.0

    # 3. Optimize
    # Initial guess: constant gamma 0.55
    c0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_c = c0
    best_score = objective(c0)
    
    # Use Differential Evolution for global search
    if time.time() < deadline - 20:
        bounds = [(-10, 10)] * 6
        try:
            res = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-5, seed=42, polish=False)
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass
            
    # Polish with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_c, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res2.fun < best_score:
                best_score = res2.fun
                best_c = res2.x
        except Exception:
            pass
            
    # Ensure return type is list of floats
    return [float(x) for x in best_c]
