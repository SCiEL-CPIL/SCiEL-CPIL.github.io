"""Optimizer for BBOB f19 (Griewank-Rosenbrock) using CMA-ES + Nelder-Mead finisher."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead local finisher, deadline-guarded."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / (dim ** 1.7 + 10)) * (1 - 1 / mu_eff)
        c_mu = (2 / (dim ** 1.7 + 10)) * min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 10))
        c_1 = max(c_1, 1e-8)
        c_mu = max(c_mu, 1e-8)

        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_used = 0
        best_x = x_mean.copy()
        best_f = float('inf')

        def eval_point(x):
            nonlocal calls_used, best_x, best_f
            fx = float(func(np.clip(x, -5, 5)))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5).copy()
            return fx

        eval_point(x_mean)

        # Nelder-Mead local finisher
        def nelder_mead(x0, max_iter=100):
            nonlocal calls_used, best_x, best_f
            n = len(x0)
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            simplex = [x0.copy()]
            fvals = [float(func(np.clip(x0, -5, 5)))]
            calls_used += 1
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = x0.copy()

            if time.time() > deadline:
                return x0, fvals[0]

            for i in range(1, n + 1):
                xi = x0.copy()
                step = 0.05 if x0[i - 1] >= 0 else -0.05
                xi[i - 1] += step
                simplex.append(xi)
                fvals.append(float(func(np.clip(xi, -5, 5))))
                calls_used += 1
                if fvals[-1] < best_f:
                    best_f = fvals[-1]
                    best_x = xi.copy()
                if time.time() > deadline:
                    return simplex[int(np.argmin(fvals))], min(fvals)

            for _ in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = fvals[-1]

                x_r = centroid + alpha * (centroid - worst)
                f_r = float(func(np.clip(x_r, -5, 5)))
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fvals[-1]:
                    if f_r < fvals[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        f_e = float(func(np.clip(x_e, -5, 5)))
                        calls_used += 1
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fvals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    if f_r < worst_f:
                        x_c = centroid + 0.5 * (x_r - centroid)
                        f_c = float(func(np.clip(x_c, -5, 5)))
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fvals[i] = float(func(np.clip(simplex[i], -5, 5)))
                            calls_used += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                            if time.time() > deadline:
                                break

            i_min = int(np.argmin(fvals))
            return simplex[i_min], fvals[i_min]

        # CMA-ES main loop
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Sample offspring
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            z = rng.standard_normal((lam, dim))
            offspring = x_mean + sigma * (z @ L.T)
            offspring = np.clip(offspring, -5, 5)

            fvals = np.array([eval_point(x) for x in offspring])
            if calls_used >= budget:
                break

            # Sort
            idx = np.argsort(fvals)
            offspring = offspring[idx]
            fvals = fvals[idx]

            # Weighted mean
            y_w = np.sum(w[:, None] * offspring[:mu], axis=0)

            # Update evolution paths
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            h_sigma = (np.sqrt((1 - np.exp(-2 / c_c))) / (1 - np.exp(-1 / c_c))) * \
                      (np.linalg.norm(C_inv_sqrt @ y_w) /
                       (sigma * np.sqrt(mu_eff)) *
                       (1 + 2 * max(0, (mu_eff - 1) / (dim + mu_eff - 1))))

            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma

            # Update covariance
            delta_h = (1 - h_sigma) * (1 - c_c) - c_c
            C = (1 - c_1 - c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) + (1 - c_c) * delta_h * C) + \
                c_mu * np.sum(w[i] * np.outer(offspring[i] - x_mean, offspring[i] - x_mean) for i in range(mu))

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update mean
            x_mean = x_mean + y_w

            # Convergence check
            if np.std(fvals[:mu]) < 1e-10:
                break

        # Final Nelder-Mead polish from best point
        if calls_used < budget and time.time() < deadline:
            nelder_mead(best_x, max_iter=80)

        return best_x, best_f
# EVOLVE-BLOCK-END
