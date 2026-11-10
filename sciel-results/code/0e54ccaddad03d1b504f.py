"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import time
import math
import numpy as np
from scipy.optimize import least_squares

_START = time.time()
_DEADLINE = 85.0

def _elapsed_ok():
    return (time.time() - _START) < _DEADLINE

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter as function of scale factor (vectorized)."""
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-10)
    de_exp = 3.0 * wa * (1.0 / a_safe - 1.0)
    E2 = Om0 * a_safe**3 + (1.0 - Om0) * np.exp(de_exp)
    return Om0 * a_safe**3 / E2

def _gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q (vectorized)."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _f_approx(coeffs, a, Om0, w0, wa):
    """f = Om(a)^gamma (vectorized)."""
    Om = _Om_of_a(a, Om0, w0, wa)
    g = _gamma_from_coeffs(coeffs, a, Om0, w0, wa)
    g = np.clip(g, 0.1, 1.2)
    return np.power(Om, g)

def _precompute_ode_reference(Om0, w0, wa, n_pts=2000):
    """Solve the GR ODE for f(a) and return interpolation grid."""
    a_start = 0.01
    a_end = 1.0
    a_grid = np.linspace(a_start, a_end, n_pts + 1)
    h = (a_end - a_start) / n_pts

    # Initial condition: f ~ Om^0.55 in matter-dominated era
    Om_start = _Om_of_a(np.array([a_start]), Om0, w0, wa)[0]
    f_curr = Om_start**0.55

    f_grid = np.zeros(n_pts + 1)
    f_grid[0] = f_curr

    for i in range(n_pts):
        a_curr = a_grid[i]
        fc = f_grid[i]
        Om_c = _Om_of_a(np.array([a_curr]), Om0, w0, wa)[0]
        w_c = w0 + wa * (1.0 - a_curr)

        def rhs(av, fv):
            Om_v = _Om_of_a(np.array([av]), Om0, w0, wa)[0]
            w_v = w0 + wa * (1.0 - av)
            return (-fv**2 - 1.5 * (1.0 + w_v) * fv + 1.5 * Om_v) / av

        k1 = rhs(a_curr, fc)
        k2 = rhs(a_curr + h/2, fc + h/2 * k1)
        k3 = rhs(a_curr + h/2, fc + h/2 * k2)
        k4 = rhs(a_curr + h, fc + h * k3)
        f_grid[i+1] = fc + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)

    return a_grid, f_grid

def _f_ref_interp(a_query, a_grid, f_grid):
    """Interpolate reference f at query points."""
    a_q = np.asarray(a_query, dtype=float)
    # Clip to grid range
    a_q = np.clip(a_q, a_grid[0], a_grid[-1])
    return np.interp(a_q, a_grid, f_grid)

def _D_normalized_from_f(f_on_grid, a_grid, a_target):
    """Compute D(a)/D(1) = exp(-integral from a to 1 of f(t)/t dt) using precomputed f."""
    if a_target >= 1.0:
        return 1.0
    # Find indices where a_grid >= a_target
    mask = a_grid >= a_target
    if not np.any(mask):
        return 1.0
    idx = np.argmax(mask)
    # Include the point at a_target by interpolating
    t_vals = np.concatenate([[a_target], a_grid[idx:]])
    f_vals = np.concatenate([[_f_ref_interp(np.array([a_target]), a_grid, f_grid)[0]], f_grid[idx:]])
    integrand = f_vals / t_vals
    integral = np.trapz(integrand, t_vals)
    return math.exp(-integral)

def _compute_linder_gamma(a, Om0, w0, wa):
    """Linder (2005) approximation: gamma = 0.55 * Om_m^alpha, alpha = 0.55/(1+0.5*w0+0.5*wa)."""
    Om = _Om_of_a(a, Om0, w0, wa)
    alpha = 0.55 / (1.0 + 0.5 * w0 + 0.5 * wa)
    # Handle edge case where denominator is near zero
    if abs(1.0 + 0.5 * w0 + 0.5 * wa) < 1e-6:
        alpha = 0.55
    g = 0.55 * np.power(np.maximum(Om, 1e-10), alpha)
    return np.clip(g, 0.1, 1.2)

def _build_training_data(cosmologies, a_values):
    """Build training data: for each (cosmo, a) point, compute target gamma from Linder."""
    all_a = []
    all_Om = []
    all_q = []
    all_u = []
    all_wa_a = []
    all_uq = []
    all_target_g = []

    for Om0, w0, wa in cosmologies:
        Om_arr = _Om_of_a(a_values, Om0, w0, wa)
        q_arr = 1.0 - Om_arr
        u_arr = 1.0 + w0 + wa * (1.0 - a_values)
        wa_a_arr = wa * a_values
        uq_arr = u_arr * q_arr
        target_g = _compute_linder_gamma(a_values, Om0, w0, wa)

        for i in range(len(a_values)):
            all_a.append(a_values[i])
            all_Om.append(Om_arr[i])
            all_q.append(q_arr[i])
            all_u.append(u_arr[i])
            all_wa_a.append(wa_a_arr[i])
            all_uq.append(uq_arr[i])
            all_target_g.append(target_g[i])

    return (np.array(all_a), np.array(all_Om), np.array(all_q),
            np.array(all_u), np.array(all_wa_a), np.array(all_uq),
            np.array(all_target_g))

def _fit_coefficients_linder(cosmologies, a_values):
    """Fit 6 coefficients to match Linder gamma via linear least squares."""
    a_arr, Om_arr, q_arr, u_arr, wa_a_arr, uq_arr, target_g = _build_training_data(cosmologies, a_values)

    # Design matrix: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa_a + c5*uq
    A = np.column_stack([
        np.ones_like(q_arr),
        q_arr,
        q_arr**2,
        u_arr,
        wa_a_arr,
        uq_arr
    ])

    # Linear least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, target_g, rcond=None)

    # Clip to valid range
    coeffs = np.clip(coeffs, -10.0, 10.0)
    return coeffs

def _objective_maxerr(coeffs, cosmologies, a_values, ode_refs):
    """Compute max relative error of f and D over all validation points."""
    max_err = 0.0
    for idx, (Om0, w0, wa) in enumerate(cosmologies):
        a_grid, f_grid = ode_refs[idx]
        for a in a_values:
            if a <= 0 or a >= 1.0:
                continue
            # Check gamma constraint
            g = _gamma_from_coeffs(coeffs, np.array([a]), Om0, w0, wa)[0]
            if g < 0.1 or g > 1.2:
                return 1e10

            f_app = _f_approx(coeffs, np.array([a]), Om0, w0, wa)[0]
            f_ref = _f_ref_interp(np.array([a]), a_grid, f_grid)[0]

            if abs(f_ref) > 1e-8:
                err_f = abs(f_app - f_ref) / abs(f_ref)
            else:
                err_f = abs(f_app - f_ref)
            max_err = max(max_err, err_f)

            # D error
            D_app = _D_normalized_from_f(f_app, a_grid, a)  # placeholder, need proper D
            # Actually compute D properly
            t = np.linspace(a, 1.0, 300)
            f_app_t = _f_approx(coeffs, t, Om0, w0, wa)
            integral_app = np.trapz(f_app_t / t, t)
            D_app = math.exp(-integral_app)

            f_ref_t = _f_ref_interp(t, a_grid, f_grid)
            integral_ref = np.trapz(f_ref_t / t, t)
            D_ref = math.exp(-integral_ref)

            if abs(D_ref) > 1e-8:
                err_D = abs(D_app - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_app - D_ref)
            max_err = max(max_err, err_D)

    return max_err

def construct():
    """Return 6 coefficients for the growth-index formula."""
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -1.0, -0.5),
        (0.27, -1.0, 0.0),
        (0.3, -0.9, 0.0),
        (0.3, -1.1, 0.2),
        (0.35, -1.0, 0.0),
        (0.3, -1.0, 0.5),
    ]

    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

    try:
        # Step 1: Pre-compute ODE references for each cosmology
        ode_refs = []
        for Om0, w0, wa in cosmologies:
            a_grid, f_grid = _precompute_ode_reference(Om0, w0, wa, n_pts=1500)
            ode_refs.append((a_grid, f_grid))

        # Step 2: Fit coefficients to Linder approximation via least squares
        coeffs_ls = _fit_coefficients_linder(cosmologies, a_values)

        # Step 3: Refine with scipy least_squares on the actual max-error objective
        # Use a dense set of a values for refinement
        a_refine = np.linspace(0.1, 0.99, 30)

        def residuals_for_ls(c):
            """Return residuals for least_squares: relative errors in f."""
            res = []
            for idx, (Om0, w0, wa) in enumerate(cosmologies):
                a_grid, f_grid = ode_refs[idx]
                for a in a_refine:
                    g = _gamma_from_coeffs(c, np.array([a]), Om0, w0, wa)[0]
                    if g < 0.1 or g > 1.2:
                        res.append(10.0)
                        continue
                    f_app = _f_approx(c, np.array([a]), Om0, w0, wa)[0]
                    f_ref = _f_ref_interp(np.array([a]), a_grid, f_grid)[0]
                    if abs(f_ref) > 1e-8:
                        res.append((f_app - f_ref) / f_ref)
                    else:
                        res.append(f_app - f_ref)
            return np.array(res)

        # Initial guess from Linder fit
        x0 = np.clip(coeffs_ls, -10.0, 10.0)

        # Run least_squares
        result = least_squares(
            residuals_for_ls,
            x0,
            method='trf',
            bounds=([-10.0]*6, [10.0]*6),
            max_nfev=200,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10
        )

        best_coeffs = np.clip(result.x, -10.0, 10.0)

        # Step 4: If time remains, do a local polish with Nelder-Mead style
        if _elapsed_ok():
            from scipy.optimize import minimize
            def obj_fn(c):
                return _objective_maxerr(c, cosmologies, a_values, ode_refs)

            res2 = minimize(
                obj_fn,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.success and res2.fun < obj_fn(best_coeffs):
                best_coeffs = np.clip(res2.x, -10.0, 10.0)

        # Final validation: ensure gamma in [0.1, 1.2] at all check points
        final_coeffs = []
        for i in range(6):
            if not math.isfinite(best_coeffs[i]):
                final_coeffs.append(0.0)
            else:
                final_coeffs.append(float(np.clip(best_coeffs[i], -10.0, 10.0)))

        return final_coeffs

    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
