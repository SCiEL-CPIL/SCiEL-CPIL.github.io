"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES with Nelder-Mead polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        call_count = 0
        rng = np.random.default_rng(42)

        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead implementation for polishing
        def nelder_mead(start_x, max_iter=20):
            nonlocal call_count, deadline
            dim = self.dim
            simplex = [start_x.copy()]
            f_simplex = [evaluate(start_x)]

            for i in range(dim):
                x_i = start_x.copy()
                step = 0.5 if start_x[i] < 4.0 else -0.5
                x_i[i] += step
                simplex.append(x_i)
                f_simplex.append(evaluate(x_i))

            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            it = 0
            while it < max_iter:
                if time.time() > deadline or call_count >= budget:
                    break

                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                centroid = np.sum(simplex[:-1], axis=0) / dim
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = evaluate(x_r)

                if f_r < f_simplex[0]:
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1], f_simplex[-1] = x_e, f_e
                    else:
                        simplex[-1], f_simplex[-1] = x_r, f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1], f_simplex[-1] = x_r, f_r
                else:
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = evaluate(x_c)
                        if f_c < f_r:
                            simplex[-1], f_simplex[-1] = x_c, f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                f_simplex[i] = evaluate(simplex[i])
                            order = np.argsort(f_simplex)
                            simplex = [simplex[i] for i in order]
                            f_simplex = [f_simplex[i] for i in order]
                            break
                    else:
                        x_c = centroid - rho * (centroid - x_worst)
                        f_c = evaluate(x_c)
                        if f_c < f_worst:
                            simplex[-1], f_simplex[-1] = x_c, f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                f_simplex[i] = evaluate(simplex[i])
                            order = np.argsort(f_simplex)
                            simplex = [simplex[i] for i in order]
                            f_simplex = [f_simplex[i] for i in order]
                            break

                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                    break
                it += 1
            return simplex[0], f_simplex[0]

        # CMA-ES Parameters
        sigma = 1.5
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))

        mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Initial evaluation
        f_mean = evaluate(mean)

        # Main CMA-ES loop
        while call_count < budget - 10 and time.time() < deadline:
            # Generate population
            Y, X, F = [], [], []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            for i in range(lambda_pop):
                Z = rng.standard_normal(dim)
                y = L @ Z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                f = evaluate(x)
                Y.append(Z)
                X.append(x)
                F.append(f)
                if call_count >= budget or time.time() > deadline:
                    break

            if len(X) < mu:
                break

            sort_idx = np.argsort(F)
            X = [X[i] for i in sort_idx]
            Y = [Y[i] for i in sort_idx]
            F = [F[i] for i in sort_idx]

            old_mean = mean.copy()
            mean = np.sum([w * X[i] for i, w in enumerate(weights)], axis=0)

            Y_norm = [Y[i] / sigma for i in range(mu)]
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                C_update += weights[i] * np.outer(Y_norm[i], Y_norm[i])

            pc = (1 - c_sigma) * pc + c_sigma * np.sqrt(mu_eff) * (mean - old_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C) + c_mu * C_update
            C = (C + C.T) / 2
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * (mean - old_mean) / sigma

            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

            if sigma < 1e-10:
                break
            if sigma > 5.0:
                sigma = 5.0

        # Polish with Nelder-Mead on the best point found
        if best_x is not None and call_count < budget - 10 and time.time() < deadline:
            nelder_mead(best_x.copy(), max_iter=15)

        return best_x, best_f

# EVOLVE-BLOCK-END
