import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0

    # Design grid: dense enough to capture worst-case behavior across the CPL parameter space
    w0_vals = np.linspace(-1.2, 0.5, 14)
    w1_vals = np.linspace(-1.5, 1.5, 14)
    z_vals = np.concatenate([
        np.linspace(0.01, 0.1, 5),
        np.linspace(0.2, 1.0, 10),
        np.linspace(1.5, 3.0, 10),
        np.linspace(3.5, 6.0, 5)
    ])

    # Precompute exact integrals for all (w0, w1, z) combinations
    exact_ints = []
    model_params = []

    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    us = np.linspace(0, 1, 10001)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    w0_arr = np.array([p[0] for p in model_params])
    w1_arr = np.array([p[1] for p in model_params])
    z_arr = np.array([p[2] for p in model_params])

    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        u = 1.0 / (1.0 + np.exp(-theta))
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        return u, w

    def compute_rel_errors(params):
        u, w = params_to_nodes_weights(params)
        t = z_arr[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return rel_errors

    def objective(params):
        return np.max(compute_rel_errors(params))

    def objective_smooth(params, temp=0.005):
        rel_errors = compute_rel_errors(params)
        max_err = np.max(rel_errors)
        if max_err > 0:
            scaled = (rel_errors - max_err) / temp
            sum_exp = np.sum(np.exp(scaled))
            if sum_exp > 0:
                return max_err + temp * np.log(sum_exp)
        return max_err

    def objective_l2(params):
        rel_errors = compute_rel_errors(params)
        return np.mean(rel_errors**2)

    # Initial guess: Gauss-Legendre nodes and weights mapped to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights

    u0_safe = np.clip(u0, 1e-9, 1 - 1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    p0 = np.concatenate([theta0, alpha0])

    # Multiple starting points for global search
    np.random.seed(42)
    start_points = [p0.copy()]
    for _ in range(5):
        pert = p0 + np.random.randn(16) * 0.15
        start_points.append(pert)

    best_params = p0
    best_score = objective(p0)

    # Optimization strategy:
    # 1. L-BFGS-B on smooth objective from multiple starts
    # 2. Powell refinement on raw max objective
    # 3. Nelder-Mead for local refinement
    for sp in start_points:
        if time.time() - start_time > time_limit - 15:
            break
        try:
            # Step 1: L-BFGS-B on smooth objective
            res_l2 = scipy.optimize.minimize(
                objective_l2, sp, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if objective(res_l2.x) < best_score:
                best_params = res_l2.x
                best_score = objective(res_l2.x)

            # Step 2: L-BFGS-B on soft-max objective
            res_sm = scipy.optimize.minimize(
                lambda p: objective_smooth(p, temp=0.003), best_params,
                method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if objective(res_sm.x) < best_score:
                best_params = res_sm.x
                best_score = objective(res_sm.x)

            # Step 3: Powell on raw max objective
            res_pw = scipy.optimize.minimize(
                objective, best_params, method='Powell',
                options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-14}
            )
            if res_pw.fun < best_score:
                best_params = res_pw.x
                best_score = res_pw.fun

            # Step 4: Nelder-Mead for local refinement
            if time.time() - start_time < time_limit - 10:
                res_nm = scipy.optimize.minimize(
                    objective, best_params, method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-14}
                )
                if res_nm.fun < best_score:
                    best_params = res_nm.x
                    best_score = res_nm.fun

        except Exception:
            pass

    # Final refinement with tighter tolerance
    if time.time() - start_time < time_limit - 5:
        try:
            res_final = scipy.optimize.minimize(
                objective, best_params, method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-16}
            )
            if res_final.fun < best_score:
                best_params = res_final.x
                best_score = res_final.fun
        except Exception:
            pass

    u_final, w_final = params_to_nodes_weights(best_params)
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    w_norm = w_sorted / np.sum(w_sorted)

    # Ensure nodes are strictly in (0,1) and strictly increasing
    u_sorted = np.clip(u_sorted, 1e-10, 1 - 1e-10)
    # Ensure strict monotonicity
    for i in range(1, 8):
        if u_sorted[i] <= u_sorted[i-1]:
            u_sorted[i] = u_sorted[i-1] + 1e-10

    return {"nodes": u_sorted, "weights": w_norm}
