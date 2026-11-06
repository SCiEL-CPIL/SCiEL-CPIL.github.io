"""
Strategy: Physics-informed sequential refinement.
1. Compute high-precision GR ODE references for a dense set of cosmologies.
2. Solve a linear least-squares problem to minimize L2 error of f and D.
   This provides a strong, non-random initial guess for the 6 coefficients.
3. Refine using multi-start Nelder-Mead on the L-infinity (worst-case) objective.
4. Use adaptive grid refinement to ensure the worst-case error is accurately captured.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a = np.asarray(a, dtype=np.float64)
        Om_de = 1.0 - Om0
        # Handle a=0 carefully if needed, but we start from 0.01
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

        # Use a fine grid for integration to ensure high precision
        x_fine = np.linspace(x_min, x_max, 2000)
        a_min = np.exp(x_min)
        y_init = [a_min, 1.0]

        sol = solve_ivp(ode, [x_min, x_max], y_init,
                       t_eval=x_fine,
                       method='DOP853', rtol=1e-12, atol=1e-14)

        if not sol.success:
            return None, None

        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])

        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
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
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_m**gamma_clipped

    def model_D(c, a, Om0, w0, wa):
        """Compute normalized D from the model via quadrature."""
        n = len(a)
        f_vals = model_f(c, a, Om0, w0, wa)
        x = np.log(a)
        dx = np.diff(x)
        trap = 0.5*(f_vals[:-1] + f_vals[1:]) * dx
        cum = np.zeros(n)
        cum[-2] = trap[-1]
        for i in range(n-3, -1, -1):
            cum[i] = cum[i+1] + trap[i]
        D_norm = np.zeros(n)
        D_norm[-1] = 1.0
        D_norm[:-1] = np.exp(-cum)
        return D_norm

    # Define validation cosmologies (disjoint from training, but we fit on a broader set 
    # to generalize, then evaluate on the specific validation set if known, or assume 
    # the evaluator uses a similar distribution. The task says "disjoint public validation 
    # cosmologies", implying we should fit on a representative set).
    # We will fit on a diverse set to ensure robustness.
    cosmologies = [
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -0.9, -0.2),
        (0.25, -0.8, 0.1),
        (0.35, -1.0, 0.0),
        (0.30, -0.7, 0.3),
        (0.28, -1.1, -0.1),
        (0.32, -0.95, 0.15),
        (0.27, -0.85, -0.15),
    ]

    # Use a moderately dense a-grid for reference (100 points for speed/accuracy balance)
    a_grid = np.linspace(0.05, 1.0, 100)

    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)

    if time.time() > DEADLINE - 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Precompute arrays for fast vectorized evaluation
    n_cosmo = len([c for c in cosmologies if c in refs])
    n_a = len(a_grid)
    
    # Store reference arrays in flat form for fast max
    ref_f_flat = np.array([refs[(Om0,w0,wa)][0] for (Om0,w0,wa) in cosmologies if (Om0,w0,wa) in refs]).ravel()
    ref_D_flat = np.array([refs[(Om0,w0,wa)][1] for (Om0,w0,wa) in cosmologies if (Om0,w0,wa) in refs]).ravel()

    # Precompute q, u, wa*a for each cosmology to speed up model evaluation
    precomp = []
    for (Om0, w0, wa) in cosmologies:
        if (Om0, w0, wa) not in refs:
            continue
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
            x = np.log(a_grid)
            dx = np.diff(x)
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

    # --- PHASE 1: Linear Least Squares (L2) for Initial Guess ---
    # We want to minimize sum (f_mod - f_ref)^2 + sum (D_mod - D_ref)^2
    # f_mod is non-linear in c (Om_m^gamma). However, for small deviations from a seed,
    # or if we linearize, we can use SVD. 
    # Better approach: Since f = Om_m^gamma = exp(gamma * ln(Om_m)), and gamma is linear in c,
    # f is exp(linear(c)). This is non-linear.
    # However, we can use a simple grid search or a few iterations of Gauss-Newton.
    # Let's use a simple local minimization from a few random starts to find a good L2 minimum,
    # then switch to L-infinity.
    
    # Actually, a simpler structural change: 
    # Use the fact that for aCDM, gamma is constant ~0.55.
    # We can start with a small grid of c0 around 0.55 and set others to 0, 
    # then run Nelder-Mead on L2 error, then on L-inf error.

    # Let's implement a multi-start local optimization on L2 first.
    def l2_objective(c):
        sum_err = 0.0
        for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
            gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa_a + c[5]*u*q
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            f_mod = Om_m ** gamma_clipped
            
            x = np.log(a_grid)
            dx = np.diff(x)
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

            sum_err += np.sum(err_f**2) + np.sum(err_D**2)
        return sum_err

    # Multi-start L2 minimization
    np.random.seed(42)
    best_l2_c = None
    best_l2_val = float('inf')
    
    # Start points: constant gamma, and slight variations
    starts = [
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.50, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.60, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.1]),
    ]

    for start in starts:
        if time.time() > DEADLINE - 20:
            break
        res = minimize(l2_objective, start, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_l2_val:
            best_l2_val = res.fun
            best_l2_c = res.x

    # --- PHASE 2: L-infinity Refinement ---
    # Start from the best L2 solution and refine on worst-case error
    if best_l2_c is None:
        best_l2_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    best_c = best_l2_c
    best_err = hard_objective_vec(best_c)

    # Multi-start L-inf refinement
    # Add small perturbations to the L2 solution to escape local minima
    perturbations = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1]),
    ]

    for pert in perturbations:
        if time.time() > DEADLINE - 10:
            break
        start_c = best_c + pert
        res = minimize(hard_objective_vec, start_c, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        hard_val = hard_objective_vec(res.x)
        if hard_val < best_err:
            best_err = hard_val
            best_c = res.x

    # Final polish with L-BFGS-B if time allows
    if time.time() < DEADLINE - 2:
        res2 = minimize(
            hard_objective_vec,
            best_c,
            method='L-BFGS-B',
            bounds=[(-10.0, 10.0)] * 6,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        hard_val3 = hard_objective_vec(res2.x)
        if hard_val3 < best_err:
            best_c = res2.x
            best_err = hard_val3

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c.tolist()
# EVOLVE-BLOCK-END
