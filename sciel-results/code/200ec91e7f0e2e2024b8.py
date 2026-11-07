import numpy as np
import time
import math


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0

        def time_left():
            return deadline - time.time()

        def do_eval(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def _ensure_pd(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            min_eig = np.min(eigvals)
            if min_eig < 1e-14:
                C = C + (1e-14 - min_eig) * np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
            C = eigvecs @ np.diag(np.clip(eigvals, 1e-14, None)) @ eigvecs.T
            return C

        def run_cmaes(mean_init, sigma_init, max_evals):
            nonlocal eval_count, best_x, best_f
            if time_left() < 5.0 or max_evals <= 0 or eval_count >= budget:
                return

            mean = np.clip(mean_init.copy(), -5, 5)
            sigma = sigma_init
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            local_evals = 0
            no_improve = 0
            best_local_f = best_f

            do_eval(mean)
            local_evals += 1

            lam = max(4, int(4 + 2 * math.log(dim)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()
            mueff = 1.0 / weights.sum()

            c1 = (2 * (mueff - 2 + 2 / dim)) / ((dim + 2) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mueff)
            cs = (lam / 2 + 1) / (dim + lam / 2 + 3)
            ds = 1 + 2 * max(0, math.sqrt((lam / dim - 1) / (dim + 1)) - 1) + cs
            chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            while local_evals < max_evals and time_left() > 5.0 and eval_count < budget:
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * z
                    x = np.clip(x, -5, 5)
                    offspring.append(x)

                values = []
                for x in offspring:
                    if local_evals >= max_evals or time_left() < 2.0 or eval_count >= budget:
                        break
                    fx = do_eval(x)
                    values.append(fx)
                    local_evals += 1

                if len(values) < mu:
                    break

                idx = np.argsort(values)
                x_sorted = [offspring[i] for i in idx]
                f_sorted = [values[i] for i in idx]

                if f_sorted[0] < best_local_f - 1e-10:
                    best_local_f = f_sorted[0]
                    no_improve = 0
                else:
                    no_improve += 1

                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (x_sorted[i] - mean) / sigma
                old_mean = mean.copy()
                mean = mean + sigma * y_w
                mean = np.clip(mean, -5, 5)

                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * y_w
                hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (local_evals / (cs * (dim + 1))))) < \
                       (1.4 + 2 / (dim + 1)) * chi_n
                sigma = sigma * math.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-15, 50.0)

                delta_hsig = (1 - hsig) * cs * (2 - cs)
                pc = (1 - cmu) * pc + delta_hsig * (y_w + delta_hsig * ps)
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (x_sorted[i] - old_mean) / sigma
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * (2 - c1 - delta_hsig) + cmu) * C - c1 * rank1 - cmu * rankmu
                C = _ensure_pd(C)

                if sigma < 1e-12 or no_improve > 30:
                    break

        # Phase 1: Broad exploration
        run_cmaes(np.zeros(dim), 2.5, min(budget, 150))

        # Phase 2: Medium exploration around best found
        if time_left() > 10.0 and eval_count < budget:
            run_cmaes(best_x.copy(), 1.0, min(budget - eval_count, 100))

        # Phase 3: Finer refinement
        if time_left() > 10.0 and eval_count < budget:
            run_cmaes(best_x.copy(), 0.3, min(budget - eval_count, 80))

        # Phase 4: Very fine CMA-ES
        if time_left() > 5.0 and eval_count < budget:
            run_cmaes(best_x.copy(), 0.05, min(budget - eval_count, 50))

        # Phase 5: Final fine-tuning with even smaller sigma
        if time_left() > 3.0 and eval_count < budget:
            run_cmaes(best_x.copy(), 0.01, min(budget - eval_count, 30))

        return best_x, best_f
