"""Deterministic multi-start least-squares regression for growth-index coefficients."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import qmc

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a = np.asarray(a, dtype=np.float64)
        Om_de = 1.0 - Om0
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return Om0/a**3 / E2

    def compute_reference(Om0, w0, wa, a_grid):
        """Compute reference f and D/D(1) using GR ODE."""
        n = len(a_grid)
        x_grid = np.log(a_grid)

        def ode(x, y):
            a = np.exp(x)
            E2 = Om0/a**3 + (1-Om0)*a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a))
            E = np.sqrt(E2)
            Om_m = Om0/a**3 / E2
            dD = y[1]
            d2D = -2.0*E*dD + 1.5*Om_m*y[0]
            return [dD, d2D]

        x_min = x_grid[0]
        x_max = 0.0

        # Use a fine grid for integration
        x_fine = np.linspace(x_min, x_max, 1000)
        a_min = np.exp(x_min)
        y_init = [a_min, 1.0]

        sol = solve_ivp(ode, [x_min, x_max], y_init,
                       t_eval=x_fine,
                       method='DOP853', rtol=1e-13, atol=1e-15)

        if not sol.success:
            return None, None

        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])

        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        f_ref = dD_ref / D_ref

        return f_ref, D_ref

    # Define training cosmologies (dense LHS grid)
    # Om0 in [0.2, 0.4], w0 in [-1.5, 0], wa in [-0.5, 0.5]
    sampler = qmc.LatinHypercube(d=3, seed=42)
    n_samples = 30
    points = sampler.random(n_samples)
    cosmologies = []
    for p in points:
        Om0 = 0.2 + 0.2 * p[0]
        w0 = -1.5 + 1.5 * p[1]
        wa = -0.5 + 1.0 * p[2]
        cosmologies.append((Om0, w0, wa))

    # Use a moderately dense a-grid for reference (50 points for speed)
    a_grid = np.linspace(0.05, 1.0, 50)
    n_a = len(a_grid)
    x_grid = np.log(a_grid)

    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)

    if time.time() > DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Precompute arrays for fast vectorized evaluation
    valid_cosmos = [(Om0, w0, wa) for (Om0, w0, wa) in cosmologies if (Om0, w0, wa) in refs]
    n_cosmo = len(valid_cosmos)
    
    # Store reference arrays in flat form for fast max
    ref_f_flat = np.array([refs[c][0] for c in valid_cosmos]).ravel()
    ref_D_flat = np.array([refs[c][1] for c in valid_cosmos]).ravel()

    # Precompute q, u, wa*a for each cosmology to speed up model evaluation
    precomp = []
    for (Om0, w0, wa) in valid_cosmos:
        q = 1.0 - Om_of_a(a_grid, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a_grid)
        wa_a = wa * a_grid
        Om_m = Om_of_a(a_grid, Om0, w0, wa)
        precomp.append((q, u, wa_a, Om_m))

    def hard_objective_vec(c):
        """Vectorized hard max-error objective."""
        max_err = 0.0
        for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
            gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa_a + c[5]*u*q
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            f_mod = Om_m ** gamma_clipped

            # D via cumulative trapezoid
            dx = np.diff(x_grid)
            trap = 0.5*(f_mod[:-1] + f_mod[1:]) * dx
            cum = np.zeros(n_a)
            cum[-2] = trap[-1]
            for i in range(n_a-3, -1, -1):
                cum[i] = cum[i+1] + trap[i]
            D_mod = np.zeros(n_a)
            D_mod[-1] = 1.0
            D_mod[:-1] = np.exp(-cum)

            # Relative errors
            f_ref_slice = ref_f_flat[idx*n_a:(idx+1)*n_a]
            D_ref_slice = ref_D_flat[idx*n_a:(idx+1)*n_a]

            err_f = np.abs((f_mod - f_ref_slice) / f_ref_slice)
            err_D = np.abs((D_mod - D_ref_slice) / D_ref_slice)

            local_max = max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max

        return max_err

    # Multi-start least squares
    # Bounds for coefficients
    bounds = [(-10.0, 10.0)] * 6
    
    # Initial guesses
    initial_guesses = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.1, -0.05, 0.0, 0.0, 0.0],
        [0.60, -0.1, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.05, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
    ]
    
    best_c = initial_guesses[0]
    best_err = hard_objective_vec(best_c)
    
    for x0 in initial_guesses:
        if time.time() > DEADLINE - 5:
            break
            
        # Use least_squares with loss='soft_l1' to handle outliers and smoothness
        # But we want to minimize the max error, so we use a custom residual
        # that emphasizes large errors
        
        def residuals(c):
            # Compute errors for all points
            errors = []
            for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
                gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa_a + c[5]*u*q
                gamma_clipped = np.clip(gamma, 0.1, 1.2)
                f_mod = Om_m ** gamma_clipped
                
                dx = np.diff(x_grid)
                trap = 0.5*(f_mod[:-1] + f_mod[1:]) * dx
                cum = np.zeros(n_a)
                cum[-2] = trap[-1]
                for i in range(n_a-3, -1, -1):
                    cum[i] = cum[i+1] + trap[i]
                D_mod = np.zeros(n_a)
                D_mod[-1] = 1.0
                D_mod[:-1] = np.exp(-cum)
                
                f_ref_slice = ref_f_flat[idx*n_a:(idx+1)*n_a]
                D_ref_slice = ref_D_flat[idx*n_a:(idx+1)*n_a]
                
                err_f = (f_mod - f_ref_slice) / f_ref_slice
                err_D = (D_mod - D_ref_slice) / D_ref_slice
                
                errors.extend(err_f.tolist())
                errors.extend(err_D.tolist())
                
            return np.array(errors)
        
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=bounds,
                method='trf',
                loss='soft_l1',
                f_scale=0.01,  # Scale to make errors O(1)
                max_nfev=1000,
                xtol=1e-10,
                ftol=1e-10,
            )
            
            # Evaluate the hard objective for this result
            err = hard_objective_vec(result.x)
            if err < best_err:
                best_err = err
                best_c = result.x
        except Exception:
            pass

    # Final polish with Nelder-Mead on the hard objective
    if time.time() < DEADLINE - 3:
        from scipy.optimize import minimize
        res = minimize(
            hard_objective_vec,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        hard_val = hard_objective_vec(res.x)
        if hard_val < best_err:
            best_c = res.x
            best_err = hard_val

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c.tolist()
# EVOLVE-BLOCK-END
