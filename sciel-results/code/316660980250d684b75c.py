import numpy as np
import scipy.optimize
import scipy.integrate
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def om_m_of_a(a, om0, w0, wa):
        """Compute Omega_m(a) for CPL dark energy."""
        if a <= 0:
            return om0
        de = (1.0 - om0) * (a ** (-3.0 * (1.0 + w0))) * np.exp(-3.0 * wa * (a - 1.0))
        m = om0 * (a ** -3.0)
        return m / (m + de)

    def ode_system(x, y, om0, w0, wa):
        a = np.exp(x)
        om_m = om_m_of_a(a, om0, w0, wa)
        w_a = w0 + wa * (1.0 - a)
        dD = y[1]
        dDp = -1.5 * (1.0 + w_a) * y[1] + 1.5 * om_m * y[0]
        return [dD, dDp]

    def integrate_growth(om0, w0, wa, x_grid):
        """Integrate ODE over x_grid (x=ln a), return D and f arrays."""
        f_seed = om0 ** 0.55
        sol = scipy.integrate.solve_ivp(
            lambda x, y: ode_system(x, y, om0, w0, wa),
            [x_grid[-1], x_grid[0]],  # from most negative to 0
            [1.0, f_seed],
            method='DOP853', rtol=1e-11, atol=1e-13,
            dense_output=True, max_step=0.02
        )
        if not sol.success:
            return None, None
        y = sol.sol(x_grid)
        D = y[0]
        f = y[1] / y[0]
        # Normalize D(1)=1 (at x=0, which is the last point)
        D = D / D[-1]
        return D, f

    # ============================================================
    # STRATEGY: Global L∞ (minimax) grid fit with differential evolution
    # - Build a dense grid of cosmologies and a-values
    # - Precompute true f and D for each grid point via ODE
    # - Objective = max relative error over all grid points (matches evaluator)
    # - Differential evolution handles non-smooth L∞ objective robustly
    # ============================================================

    # Grid of cosmologies to fit (representative of validation space)
    # Om_m0 in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    om0_grid = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    w0_grid = np.array([-1.5, -1.2, -1.0, -0.8, -0.6, -0.4])
    wa_grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    # a-grid: log-spaced from 0.05 to 1.0
    a_vals = np.logspace(np.log10(0.05), 0.0, 60)
    x_vals = np.log(a_vals)

    # Build list of all (om0, w0, wa) combinations
    cosmos = []
    for om0 in om0_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                cosmos.append((om0, w0, wa))

    # Precompute true f and D for all grid points
    print(f"Fitting on {len(cosmos)} cosmologies x {len(a_vals)} a-values", flush=True)
    true_f = []
    true_D = []
    om_m_grid = []
    q_grid = []
    u_grid = []
    wa_a_grid = []
    a_grid = []

    for i, (om0, w0, wa) in enumerate(cosmos):
        if time.time() > deadline - 30.0:
            break
        D, f = integrate_growth(om0, w0, wa, x_vals)
        if D is None:
            continue
        om_m = np.array([om_m_of_a(a, om0, w0, wa) for a in a_vals])
        q = 1.0 - om_m
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        wa_a = wa * a_vals
        true_f.append(f)
        true_D.append(D)
        om_m_grid.append(om_m)
        q_grid.append(q)
        u_grid.append(u)
        wa_a_grid.append(wa_a)
        a_grid.append(a_vals)

    n_samples = len(true_f)
    if n_samples == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Flatten arrays for vectorized evaluation
    om_m_flat = np.concatenate(om_m_grid)
    q_flat = np.concatenate(q_grid)
    u_flat = np.concatenate(u_grid)
    wa_a_flat = np.concatenate(wa_a_grid)
    a_flat = np.concatenate(a_grid)
    f_flat = np.concatenate(true_f)
    D_flat = np.concatenate(true_D)
    # For D integration, we need per-cosmology grouping
    # Store offsets
    offsets = np.cumsum([len(true_f[i]) for i in range(n_samples)])
    offsets = np.insert(offsets, 0, 0)

    # Objective: L∞ max relative error
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        # Check bounds
        if np.any(np.abs(coeffs) > 10.0):
            return 1e10

        # Compute gamma for all points
        gamma = c0 + c1*q_flat + c2*q_flat**2 + c3*u_flat + c4*wa_a_flat + c5*u_flat*q_flat

        # Check gamma bounds [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10

        # Compute f_approx
        f_approx = om_m_flat ** gamma

        # Relative error in f
        err_f = np.abs(f_approx - f_flat) / np.abs(f_flat)

        # Compute D_approx via integration per cosmology
        # D(a)/D(1) = exp(-∫_{ln a}^{0} f(e^x) dx)
        # We have x_vals (same for all), and f_approx is ordered by cosmology
        # For each cosmology, integrate f_approx from x to 0
        max_err_D = 0.0
        for i in range(n_samples):
            f_i = f_approx[offsets[i]:offsets[i+1]]
            # Integrate from x to 0: ∫_{x_vals[j]}^{0} f dx
            # x_vals goes from log(0.05) to 0
            # Use trapezoidal rule
            dx = np.diff(x_vals)
            integral = np.zeros(len(x_vals))
            for j in range(1, len(x_vals)):
                integral[j] = integral[j-1] + 0.5*(f_i[j-1] + f_i[j]) * dx[j-1]
            # integral[j] = ∫_{x_vals[0]}^{x_vals[j]} f dx
            # We want ∫_{x_vals[j]}^{0} f dx = integral[-1] - integral[j]
            D_approx_i = np.exp(-(integral[-1] - integral))
            err_D_i = np.abs(D_approx_i - true_D[offsets[i]:offsets[i+1]])
            # Relative error (D should be positive)
            err_D_i = np.abs(D_approx_i - D_flat[offsets[i]:offsets[i+1]]) / np.maximum(np.abs(D_flat[offsets[i]:offsets[i+1]]), 1e-10)
            max_err_D = max(max_err_D, np.max(err_D_i))

        # Overall L∞ error
        max_err_f = np.max(err_f)
        max_err = max(max_err_f, max_err_D)
        return max_err

    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_coeffs = x0.copy()
    best_obj = objective(x0)
    print(f"Initial L∞ error: {best_obj:.6f}", flush=True)

    # Differential evolution
    try:
        result = scipy.optimize.differential_evolution(
            objective,
            bounds=[(-10, 10)] * 6,
            seed=42,
            maxiter=200,
            tol=1e-8,
            popsize=15,
            polish=True,
            init='latinhypercube',
            update_interval=10,
            callback=None
        )
        if result.fun < best_obj:
            best_coeffs = result.x
            best_obj = result.fun
        print(f"After DE: L∞ error = {best_obj:.6f}", flush=True)
    except Exception as e:
        print(f"DE failed: {e}", flush=True)

    # Polish with Nelder-Mead if time remains
    if time.time() < deadline - 10.0:
        try:
            result2 = scipy.optimize.minimize(objective, best_coeffs, method='Nelder-Mead',
                                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result2.fun < best_obj:
                best_coeffs = result2.x
                best_obj = result2.fun
            print(f"After NM polish: L∞ error = {best_obj:.6f}", flush=True)
        except Exception as e:
            print(f"NM failed: {e}", flush=True)

    # Final validation: check gamma bounds on a broader grid
    c = np.clip(best_coeffs, -10, 10)
    test_cosmos = [(0.20, -1.5, -1.0), (0.25, -1.2, -0.5), (0.30, -1.0, 0.0),
                   (0.35, -0.8, 0.5), (0.40, -0.5, 1.0), (0.22, -1.4, -0.8),
                   (0.38, -0.6, 0.7), (0.28, -1.1, -0.3)]
    test_a = np.logspace(np.log10(0.03), 0.0, 200)
    valid = True
    for (om0, w0, wa) in test_cosmos:
        om_m_t = np.array([om_m_of_a(a, om0, w0, wa) for a in test_a])
        q_t = 1.0 - om_m_t
        u_t = 1.0 + w0 + wa * (1.0 - test_a)
        gamma_t = c[0] + c[1]*q_t + c[2]*q_t**2 + c[3]*u_t + c[4]*wa*test_a + c[5]*u_t*q_t
        if np.any(gamma_t < 0.1) or np.any(gamma_t > 1.2):
            valid = False
            break

    if not valid:
        # Fallback to constant gamma
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return c.tolist()
# EVOLVE-BLOCK-END
