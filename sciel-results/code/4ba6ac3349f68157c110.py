import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    time_limit = 610.0  # Safety margin before 630s kill

    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2*Y

    bin_idx_m1 = (V_m1 + 9).flatten()
    bin_idx_0 = V_0.flatten()
    bin_idx_1 = (V_1).flatten()
    bin_idx_2 = (V_2).flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def compute_ratio_from_logP(logP_flat, smooth_beta=0.0):
        z = logP_flat - np.max(logP_flat)
        P_flat = np.exp(z)
        sum_P = np.sum(P_flat)
        if sum_P < 1e-300:
            return 0.0
        P_flat = P_flat / sum_P

        counts_m1 = np.bincount(bin_idx_m1, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(bin_idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P_flat, minlength=num_bins_2)

        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))

        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)

        if smooth_beta > 1e-6:
            H_vals = np.array([H_0, H_1, H_2])
            exp_H = np.exp(H_vals / smooth_beta)
            M = smooth_beta * np.log(np.sum(exp_H))
        else:
            M = max(H_0, H_1, H_2)

        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def objective(logP_flat):
        r = compute_ratio_from_logP(logP_flat)
        return -r

    def objective_smooth(logP_flat):
        return compute_ratio_from_logP(logP_flat, smooth_beta=1.0) * -1

    best_score = 0.0
    best_P = np.ones((10, 10)) / 100.0

    # 4-param exponential family: logP = a0 + a1*x + a2*y + a3*x*y
    # This allows asymmetric diagonal mass distributions
    def logP_from_params_4(params):
        a0, a1, a2, a3 = params
        x_vals = np.arange(n).reshape(1, -1)
        y_vals = np.arange(n).reshape(-1, 1)
        logP = a0 + a1 * x_vals + a2 * y_vals + a3 * x_vals * y_vals
        return logP.flatten()

    def objective_params_4(params):
        logP = logP_from_params_4(params)
        return objective(logP)

    def objective_params_4_smooth(params):
        logP = logP_from_params_4(params)
        return objective_smooth(logP)

    # 3-param quadratic Gaussian for diversity
    def logP_from_params_3(params):
        alpha, beta, gamma = params
        x_vals = np.arange(n).reshape(1, -1)
        y_vals = np.arange(n).reshape(-1, 1)
        logP = -alpha * x_vals**2 - beta * y_vals**2 - gamma * x_vals * y_vals
        return logP.flatten()

    # Grid search over 4-param exponential family
    param_inits_4 = []
    for a1 in np.linspace(-2.0, 2.0, 10):
        for a2 in np.linspace(-2.0, 2.0, 10):
            for a3 in np.linspace(-1.5, 1.5, 10):
                param_inits_4.append((0.0, a1, a2, a3))

    scored_params_4 = []
    for params in param_inits_4:
        if time.time() - start_time > time_limit - 120:
            break
        logP = logP_from_params_4(params)
        score = -objective(logP)
        scored_params_4.append((score, params))

    scored_params_4.sort(key=lambda t: t[0], reverse=True)

    # Grid search over 3-param quadratic
    param_inits_3 = []
    for alpha in np.linspace(-1.5, 1.5, 8):
        for beta in np.linspace(-1.5, 1.5, 8):
            for gamma in np.linspace(-2.0, 2.0, 8):
                param_inits_3.append((alpha, beta, gamma))

    scored_params_3 = []
    for params in param_inits_3:
        if time.time() - start_time > time_limit - 100:
            break
        logP = logP_from_params_3(params)
        score = -objective(logP)
        scored_params_3.append((score, params))

    scored_params_3.sort(key=lambda t: t[0], reverse=True)

    # Differential evolution on 4-param space (global search)
    if time.time() - start_time < time_limit - 60:
        bounds_4 = [(-10, 10), (-3, 3), (-3, 3), (-2, 2)]
        try:
            de_res = differential_evolution(
                objective_params_4_smooth, bounds_4,
                seed=42, maxiter=50, popsize=20, tol=1e-8,
                polish=True, init='sobol'
            )
            logP_de = logP_from_params_4(de_res.x)
            score_de = -objective(logP_de)
            if score_de > best_score:
                best_score = score_de
                z = logP_de - np.max(logP_de)
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # L-BFGS-B refinement from top parametric points (4-param)
    init_points = []
    init_points.append(np.zeros(100))  # uniform

    for score, params in scored_params_4[:15]:
        init_points.append(logP_from_params_4(params))

    for score, params in scored_params_3[:10]:
        init_points.append(logP_from_params_3(params))

    # Rank-1 structured
    for la in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        for lb in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            logP = np.zeros(100)
            for y in range(10):
                for x in range(10):
                    logP[y*10+x] = la*x + lb*y
            init_points.append(logP)

    # Diagonal concentration
    for c in range(-9, 10):
        logP = np.full(100, -20.0)
        for y in range(10):
            x = y + c
            if 0 <= x < 10:
                logP[y*10+x] = 0.0
        init_points.append(logP)

    # Random
    rng = np.random.default_rng(12345)
    for _ in range(15):
        init_points.append(rng.normal(0, 1.5, 100))

    for i, init in enumerate(init_points):
        if time.time() - start_time > time_limit - 30:
            break
        try:
            res = minimize(objective_smooth, init, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            score = -objective(res.x)
            if score > best_score:
                best_score = score
                z = res.x - np.max(res.x)
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # L-BFGS-B on 4-param space from top grid points
    if time.time() - start_time < time_limit - 20:
        for score, params in scored_params_4[:8]:
            if time.time() - start_time > time_limit - 20:
                break
            try:
                res = minimize(objective_params_4_smooth, np.array(params),
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                logP_opt = logP_from_params_4(res.x)
                score_opt = -objective(logP_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    z = logP_opt - np.max(logP_opt)
                    P_flat = np.exp(z)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
            except Exception:
                pass

    # High-precision polish
    if time.time() - start_time < time_limit - 10:
        z = np.log(np.clip(best_P, 1e-300, None)).flatten()
        for trial in range(10):
            if time.time() - start_time > time_limit - 10:
                break
            jitter = rng.normal(0, 0.03, 100)
            try:
                res = minimize(objective_smooth, z + jitter, method='L-BFGS-B',
                               options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-13})
                score = -objective(res.x)
                if score > best_score:
                    best_score = score
                    z2 = res.x - np.max(res.x)
                    P_flat = np.exp(z2)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
                    z = z2
            except Exception:
                pass

    # Final exact-evaluation polish
    if time.time() - start_time < time_limit - 5:
        for trial in range(4):
            if time.time() - start_time > time_limit - 5:
                break
            jitter = rng.normal(0, 0.01, 100)
            try:
                res = minimize(objective, z + jitter, method='L-BFGS-B',
                               options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-14})
                score = -res.fun
                if score > best_score:
                    best_score = score
                    z2 = res.x - np.max(res.x)
                    P_flat = np.exp(z2)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
                    z = z2
            except Exception:
                pass

    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s

    return best_P
