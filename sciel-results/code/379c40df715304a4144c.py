"""CMA-ES + Nelder-Mead hybrid optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid CMA-ES + Nelder-Mead + (1+1)-ES for BBOB f11.
    CMA-ES handles the discus-shaped ill-conditioned landscape via covariance adaptation.
    Nelder-Mead polishes the local region. (1+1)-ES does final refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Coarse exploration ---
        points = []
        values = []

        # Evaluate center
        center = np.zeros(dim)
        fc = func(np.clip(center, -5, 5))
        points.append(center)
        values.append(fc)
        evals = 1

        # Axis-aligned probes to find shift direction
        for i in range(dim):
            for d in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1.0, -1.0]:
                    if evals >= budget or not time_ok():
                        break
                    p = np.zeros(dim)
                    p[i] = sign * d
                    fp = func(np.clip(p, -5, 5))
                    points.append(p)
                    values.append(fp)
                    evals += 1
                if evals >= budget or not time_ok():
                    break
            if evals >= budget or not time_ok():
                break

        # Random points for diversity
        explore_budget = min(60, budget // 5)
        if evals < explore_budget:
            for _ in range(explore_budget - evals):
                if evals >= budget or not time_ok():
                    break
                p = rng.uniform(-4.5, 4.5, dim)
                fp = func(np.clip(p, -5, 5))
                points.append(p)
                values.append(fp)
                evals += 1

        if evals >= budget or not time_ok():
            best_idx = int(np.argmin(values))
            return points[best_idx], values[best_idx]

        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx], dtype=np.float64)
        best_f = values[best_idx]

        # --- Phase 2: CMA-ES ---
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_pop // 2

        sigma = 1.0
        cc = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        cs = (lambda_pop / 2 + 1) / (dim + lambda_pop / 2 + 5)
        cmu = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        damps = 1 + 2 * max(0, np.sqrt((lambda_pop / 2 - 1) / (dim + 1)) - 1) + cs

        x_mean = best_x.copy()
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        try:
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
            C_sqrt = L.T
            C_inv_sqrt = L_inv
        except:
            pass

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x_cma = x_mean.copy()
        best_f_cma = best_f

        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        gen = 0
        cma_budget = int(budget * 0.6)  # Use 60% of budget for CMA-ES

        while evals < cma_budget and time_ok():
            gen += 1

            offspring = []
            offspring_f = []

            for i in range(lambda_pop):
                if evals >= cma_budget or not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            off_order = np.argsort(offspring_f)
            sorted_offspring = [offspring[j] for j in off_order]
            sorted_offspring_f = [offspring_f[j] for j in off_order]

            if sorted_offspring_f[0] < best_f_cma:
                best_f_cma = sorted_offspring_f[0]
                best_x_cma = sorted_offspring[0].copy()

            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_offspring[i]

            x_diff = x_new - x_mean
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (x_diff / sigma)

            hsig = np.sqrt((1 + (1 - 2 * cs) ** 2) / (1 - cs) - 1) * (2 - cs)
            hs = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (dim + 1)) * hsig else 0.0

            p_c = (1 - cc) * p_c + hs * np.sqrt(cc * (2 - cc) * mu_eff) * C_inv_sqrt @ (x_diff / sigma)

            delta_hsig = (1 - hs) * cc * (2 - cc)
            C = (1 + cmu * (mu_eff + 2) - cmu) * C + \
                cmu * np.outer(p_c, p_c) * (1 + delta_hsig - cc * (2 - cc)) + \
                cmu / mu_eff * sum(weights[i] * np.outer(sorted_offspring[i] - x_mean, sorted_offspring[i] - x_mean) for i in range(mu))

            C = (C + C.T) / 2 + 1e-12 * np.eye(dim)

            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * gen)) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x_mean = x_new

            try:
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
                C_sqrt = L.T
                C_inv_sqrt = L_inv
            except:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = max(sigma, 0.1)

        if best_f_cma < best_f:
            best_f = best_f_cma
            best_x = best_x_cma

        # --- Phase 3: Nelder-Mead refinement ---
        if evals < budget and time_ok():
            step = max(sigma, 0.3)
            simplex = [best_x.copy() for _ in range(dim + 1)]
            for i in range(dim):
                simplex[i + 1][i] += step

            simplex_f = []
            for s in simplex:
                if evals >= budget or not time_ok():
                    break
                f = func(np.clip(s, -5, 5))
                simplex_f.append(f)
                evals += 1

            if len(simplex_f) == dim + 1:
                min_idx = int(np.argmin(simplex_f))
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5

                nm_budget = budget - 30  # Reserve 30 for (1+1)-ES
                stagnation = 0
                max_stagnation = 20

                while evals < nm_budget and time_ok():
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    if simplex_f[0] < best_f:
                        best_f = simplex_f[0]
                        best_x = simplex[0].copy()

                    centroid = np.mean(simplex[:-1], axis=0)

                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    evals += 1

                    if f_r < simplex_f[0]:
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        evals += 1

                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                        stagnation = 0
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                        stagnation = 0
                    else:
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        evals += 1

                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                            stagnation = 0
                        else:
                            for i in range(1, dim + 1):
                                if evals >= nm_budget or not time_ok():
                                    break
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                                evals += 1
                            stagnation += 1

                    if stagnation >= max_stagnation:
                        break

        # --- Phase 4: (1+1)-ES final polish ---
        if evals < budget and time_ok():
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = 0.05
            no_improve = 0
            max_no_improve = 30

            while evals < budget and time_ok():
                z = rng.standard_normal(dim)
                y = x_local + sigma_local * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1

                if fy < f_local:
                    x_local = y
                    f_local = fy
                    sigma_local = max(sigma_local * 1.05, 1e-6)
                    no_improve = 0
                else:
                    sigma_local = max(sigma_local * 0.95, 1e-6)
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
