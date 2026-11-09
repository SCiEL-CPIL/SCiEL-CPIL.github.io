"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

# EVOLVE-BLOCK-START
def construct():
    """
    Fit six coefficients to minimize the worst-case relative error of f and D 
    across a grid of CPL cosmologies.
    """
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin before 90s kill

    # 1. Initial Least Squares Fit to get a good starting point
    # Grid for initial fit
    Om0_init = np.array([0.25, 0.30, 0.35])
    w0_init = np.array([-1.0, -0.9, -0.8])
    wa_init = np.array([-0.3, 0.0, 0.3])
    a_init = np.linspace(0.1, 0.99, 50)

    rows = []
    targets = []

    for Om0 in Om0_init:
        for w0 in w0_init:
            for wa in wa_init:
                for a in a_init:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    if E2 <= 0: continue
                    Om_a = Om0 * a_cubed / E2
                    if Om_a <= 0 or Om_a >= 1.0: continue
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    if u <= 0.05: continue
                    
                    gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    rows.append(row)
                    targets.append(gamma_target)
                    
                    if time.time() > deadline - 5.0: break
                if time.time() > deadline - 5.0: break
            if time.time() > deadline - 5.0: break
        if time.time() > deadline - 5.0: break

    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.array(rows)
    b = np.array(targets)
    try:
        coeffs_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        coeffs = np.clip(coeffs_ls, -10.0, 10.0)
    except Exception:
        coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 2. Prepare Data for Chebyshev Optimization
    # Use a denser grid for the final optimization to better estimate the max error
    Om0_opt = np.array([0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0_opt = np.array([-1.2, -1.0, -0.9, -0.8, -0.7])
    wa_opt = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    
    # Dense a-grid for integration and evaluation
    # We need to integrate from a to 1, so we sort a in descending order for cumulative trapezoid?
    # cumulative_trapezoid(y, x, initial=0) integrates from x[0] to x[i].
    # We want integral from a to 1.
    # Let x = ln(a). We want int_{ln(a)}^0 f(e^t) dt.
    # If we define g(t) = f(e^t), we want int_{x}^0 g(t) dt.
    # This is equivalent to - int_0^x g(t) dt.
    # Or, if we define the grid in reverse: a_desc from 1 to a_min.
    # Let's use a grid from 1.0 down to 0.05.
    a_grid = np.linspace(1.0, 0.05, 100)
    
    # Pre-compute cosmological quantities for all (Om0, w0, wa, a) combinations
    # To save memory and time, we process one cosmology at a time in the objective function,
    # but we can pre-compute the static parts: Om_a, q, u for each a and each cosmology.
    
    # Structure:
    # cosm_data: list of dicts for each (Om0, w0, wa)
    # Each dict has:
    #   'a': a_grid
    #   'Om_a': array of Om(a)
    #   'q': array of q
    #   'u': array of u
    #   'wa_a': array of wa*a
    #   'u_q': array of u*q
    #   'gamma_true': array of true gamma
    #   'f_true': array of true f
    #   'D_true': array of true D (normalized)
    
    cosm_data = []
    for Om0 in Om0_opt:
        for w0 in w0_opt:
            for wa in wa_opt:
                a = a_grid
                a_cubed = a**3
                de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                
                # Handle potential invalid E2
                valid_mask = E2 > 1e-10
                if not np.all(valid_mask):
                    # If some points are invalid, we might skip this cosmology or handle carefully.
                    # For simplicity and robustness, skip if too many invalid.
                    if np.sum(valid_mask) < 50:
                        continue
                        
                Om_a = np.zeros_like(a)
                Om_a[valid_mask] = Om0 * a_cubed[valid_mask] / E2[valid_mask]
                
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                
                # Check u validity
                u_valid = u > 0.05
                if not np.all(u_valid):
                    continue
                
                gamma_true = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                f_true = Om_a**gamma_true
                
                # Compute D_true
                # D(a) = exp(- integral_{ln(a)}^0 f(e^x) dx)
                # Let x = ln(a). Grid is a_grid (1.0 to 0.05).
                # x_grid = ln(a_grid) (0.0 to ln(0.05))
                x_grid = np.log(a)
                # We need integral from x to 0.
                # Let's compute cumulative integral from 0 to x.
                # integral_0^x f(e^t) dt.
                # Since x_grid is decreasing (0 to negative), we reverse it to integrate normally.
                x_rev = x_grid[::-1] # negative to 0
                f_rev = f_true[::-1]
                
                # cumulative_trapezoid gives integral from x_rev[0] to x_rev[i]
                # We want integral from 0 to x.
                # Let's integrate f(e^t) from t=0 to t=x.
                # Note: x is negative.
                # int_0^x = - int_x^0.
                # Let's just compute I(x) = int_x^0 f(e^t) dt directly.
                # Using the grid from x to 0 (which is a_grid from a to 1).
                # a_grid is 1.0, ..., 0.05.
                # x_grid is 0.0, ..., ln(0.05).
                # We want for each a_i (x_i), the integral from x_i to 0.
                # This is the area under f(e^t) from t=x_i to t=0.
                
                # We can compute the cumulative sum of trapezoids from the end (0.0) backwards.
                # Or use cumulative_trapezoid on the reversed grid (from x_min to 0).
                # x_rev goes from ln(0.05) to 0.
                # cum_int_rev[i] = int_{x_rev[0]}^{x_rev[i]} f(e^t) dt.
                # We want int_{x_i}^0 f(e^t) dt.
                # If x_i is x_rev[j], then int_{x_rev[j]}^0 = total_int - cum_int_rev[j].
                
                cum_int_rev = cumulative_trapezoid(f_rev, x_rev, initial=0.0)
                total_int = cum_int_rev[-1] # integral from x_min to 0
                
                # For each point i in original order (0 to x_min):
                # x_i corresponds to index N-1-i in rev?
                # a_grid[0]=1.0, x_grid[0]=0.
                # a_grid[-1]=0.05, x_grid[-1]=ln(0.05).
                # x_rev[0]=ln(0.05), x_rev[-1]=0.
                # Point i in original: x_i.
                # In rev, x_i is at index N-1-i.
                # cum_int_rev[N-1-i] is int_{x_min}^{x_i} f(e^t) dt.
                # We want int_{x_i}^0 f(e^t) dt = total_int - int_{x_min}^{x_i} f(e^t) dt.
                
                # Map indices:
                # idx_rev = N - 1 - np.arange(N)
                # int_x_to_0 = total_int - cum_int_rev[idx_rev]
                
                idx_rev = len(a) - 1 - np.arange(len(a))
                integral_x_to_0 = total_int - cum_int_rev[idx_rev]
                
                D_true = np.exp(-integral_x_to_0)
                
                # Store data
                cosm_data.append({
                    'Om_a': Om_a,
                    'q': q,
                    'u': u,
                    'wa_a': wa * a,
                    'u_q': u * q,
                    'f_true': f_true,
                    'D_true': D_true
                })
                
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
        if time.time() > deadline - 10.0:
            break

    if not cosm_data:
        return [float(c) for c in coeffs]

    def compute_max_err(c):
        max_err = 0.0
        c = np.array(c)
        
        for data in cosm_data:
            Om_a = data['Om_a']
            q = data['q']
            u = data['u']
            wa_a = data['wa_a']
            u_q = data['u_q']
            f_true = data['f_true']
            D_true = data['D_true']
            
            # Compute gamma approx
            gamma = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa_a + c[5]*u_q
            
            # Check constraints
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            
            # Compute f approx
            f_approx = Om_a**gamma
            
            # Error in f
            err_f = np.abs(f_approx - f_true) / np.abs(f_true)
            max_err_f = np.max(err_f)
            if max_err_f > max_err:
                max_err = max_err_f
            
            # Compute D approx
            # Need integral of f_approx from x to 0
            x_grid = np.log(data['a']) # Wait, we didn't store 'a' in data. 
            # We need 'a' or 'x_grid' to do the integration.
            # Let's add 'a' to data.
            
            # If 'a' is not in data, we can't do this. 
            # I will assume I added it in the loop above. 
            # Correction: I did not add 'a' to the dict in the code above. 
            # I must fix the data storage to include 'a'.
            
            pass # Placeholder, will fix in final code

    # Re-do data storage to include 'a'
    cosm_data = []
    for Om0 in Om0_opt:
        for w0 in w0_opt:
            for wa in wa_opt:
                a = a_grid
                a_cubed = a**3
                de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                valid_mask = E2 > 1e-10
                if np.sum(valid_mask) < 50: continue
                Om_a = np.zeros_like(a)
                Om_a[valid_mask] = Om0 * a_cubed[valid_mask] / E2[valid_mask]
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a)
                if np.any(u <= 0.05): continue
                gamma_true = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                f_true = Om_a**gamma_true
                x_grid = np.log(a)
                x_rev = x_grid[::-1]
                f_rev = f_true[::-1]
                cum_int_rev = cumulative_trapezoid(f_rev, x_rev, initial=0.0)
                total_int = cum_int_rev[-1]
                idx_rev = len(a) - 1 - np.arange(len(a))
                integral_x_to_0 = total_int - cum_int_rev[idx_rev]
                D_true = np.exp(-integral_x_to_0)
                
                cosm_data.append({
                    'a': a,
                    'x_grid': x_grid,
                    'Om_a': Om_a,
                    'q': q,
                    'u': u,
                    'wa_a': wa * a,
                    'u_q': u * q,
                    'f_true': f_true,
                    'D_true': D_true
                })
                if time.time() > deadline - 10.0: break
            if time.time() > deadline - 10.0: break
        if time.time() > deadline - 10.0: break

    if not cosm_data:
        return [float(c) for c in coeffs]

    def compute_max_err(c):
        max_err = 0.0
        c = np.array(c)
        
        for data in cosm_data:
            Om_a = data['Om_a']
            q = data['q']
            u = data['u']
            wa_a = data['wa_a']
            u_q = data['u_q']
            f_true = data['f_true']
            D_true = data['D_true']
            x_grid = data['x_grid']
            
            gamma = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa_a + c[5]*u_q
            
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6
            
            f_approx = Om_a**gamma
            
            err_f = np.abs(f_approx - f_true) / np.abs(f_true)
            max_err_f = np.max(err_f)
            if max_err_f > max_err:
                max_err = max_err_f
            
            # Compute D approx
            x_rev = x_grid[::-1]
            f_approx_rev = f_approx[::-1]
            cum_int_rev_app = cumulative_trapezoid(f_approx_rev, x_rev, initial=0.0)
            total_int_app = cum_int_rev_app[-1]
            idx_rev = len(x_grid) - 1 - np.arange(len(x_grid))
            integral_x_to_0_app = total_int_app - cum_int_rev_app[idx_rev]
            D_approx = np.exp(-integral_x_to_0_app)
            
            err_D = np.abs(D_approx - D_true) / np.abs(D_true)
            max_err_D = np.max(err_D)
            if max_err_D > max_err:
                max_err = max_err_D
                
        return max_err

    # Optimize
    try:
        result = minimize(compute_max_err, coeffs, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < compute_max_err(coeffs):
            coeffs = result.x
    except Exception:
        pass

    # Final validity check
    valid = True
    for data in cosm_data:
        gamma = coeffs[0] + coeffs[1]*data['q'] + coeffs[2]*data['q']**2 + coeffs[3]*data['u'] + coeffs[4]*data['wa_a'] + coeffs[5]*data['u_q']
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            valid = False
            break
    
    if not valid:
        return [float(c) for c in coeffs_ls] if 'coeffs_ls' in locals() else [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [float(c) for c in coeffs]
# EVOLVE-BLOCK-END
