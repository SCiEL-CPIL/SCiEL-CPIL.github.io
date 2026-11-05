import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85

    x0 = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

    cosmologies = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.0),
        (0.30, -1.2, 0.5),
        (0.30, -0.8, -0.3),
        (0.30, -1.0, 0.3),
        (0.30, -1.1, -0.2),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -0.95, 0.1),
        (0.30, -1.05, -0.1),
        (0.30, -0.9, 0.2),
        (0.30, -1.1, 0.2),
    ]

    a_vals = np.linspace(0.1, 1.0, 50)
    x_vals = np.log(a_vals)

    def compute_Om_a(Om0, w0, wa, a):
        a = np.asarray(a, dtype=float)
        OmL0 = 1.0 - Om0
        exponent = 3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (a - 1.0)
        rho_L_ratio = np.exp(exponent)
        rho_m_ratio = Om0 * a**(-3.0)
        rho_L = OmL0 * rho_L_ratio
        E2 = rho_m_ratio + rho_L
        Om_a = rho_m_ratio / E2
        return Om_a

    def compute_dlnOm_dlna(Om0, w0, wa, a):
        a = np.asarray(a, dtype=float)
        da = 1e-6
        Om_plus = compute_Om_a(Om0, w0, wa, a + da)
        Om_minus = compute_Om_a(Om0, w0, wa, a - da)
        Om_mid = compute_Om_a(Om0, w0, wa, a)
        dlnOm_da = (Om_plus - Om_minus) / (2.0 * da) / Om_mid
        return a * dlnOm_da

    def solve_growth_ODE(Om0, w0, wa):
        def ode_func(x, y):
            D, Dp = y
            a = np.exp(x)
            Om = compute_Om_a(Om0, w0, wa, a)
            dlnOm_dx = compute_dlnOm_dlna(Om0, w0, wa, a)
            Dpp = -1.5 * Om * D - (1.5 + 0.5 * dlnOm_dx) * Dp
            return [Dp, Dpp]

        x0_val = x_vals[0]
        a0 = np.exp(x0_val)
        y0 = [a0, a0]

        sol = solve_ivp(ode_func, [x0_val, x_vals[-1]], y0, t_eval=x_vals, rtol=1e-10, atol=1e-12)

        if not sol.success:
            return None, None

        D = sol.y[0]
        Dp = sol.y[1]
        D_norm = D / D[-1]
        f = Dp / D
        return f, D_norm

    true_data = {}
    for (Om0, w0, wa) in cosmologies:
        if time.time() > deadline:
            break
        f_true, D_true = solve_growth_ODE(Om0, w0, wa)
        if f_true is not None:
            true_data[(Om0, w0, wa)] = (f_true, D_true)

    if not true_data:
        return [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]

    def evaluate_coeffs(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0

        for (Om0, w0, wa) in cosmologies:
            if (Om0, w0, wa) not in true_data:
                continue
            f_true, D_true = true_data[(Om0, w0, wa)]

            Om_a = compute_Om_a(Om0, w0, wa, a_vals)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a_vals)

            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q

            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e6

            f_approx = Om_a ** gamma

            integral_to_end = cumulative_trapezoid(f_approx[::-1], x_vals[::-1], initial=0.0)[::-1]
            D_approx = np.exp(-integral_to_end)

            err_f = np.abs(f_approx - f_true) / np.abs(f_true)
            err_D = np.abs(D_approx - D_true) / np.abs(D_true)

            max_err = max(max_err, np.max(err_f), np.max(err_D))

        return max_err

    best_x = x0
    best_val = evaluate_coeffs(x0)

    if time.time() < deadline - 10:
        result = minimize(evaluate_coeffs, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun

    if time.time() < deadline - 5:
        bounds = [(-10, 10)] * 6
        result = minimize(evaluate_coeffs, best_x, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10})
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun

    best_x = np.clip(best_x, -10, 10)
    return list(best_x)

