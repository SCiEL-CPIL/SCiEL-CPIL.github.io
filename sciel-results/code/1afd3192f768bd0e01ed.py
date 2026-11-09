"""Combined DE + CMA-ES optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE for global search + CMA-ES for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            fx = func(x)
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_cmaes(x0, sigma0, max_calls, rng_local):
            nonlocal calls_made
            lambda_ = 20
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 2) / (dim + mueff)) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            gen = 0
            local_calls = 0

            while local_calls + lambda_ <= max_calls and time.time() < deadline and calls_made < budget:
                B = rng_local.standard_normal((lambda_, dim))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)

                Y = x_mean[None, :] + sigma * (B @ L)
                Y = np.clip(Y, -5, 5)

                F = []
                for y in Y:
                    if calls_made >= budget:
                        break
                    F.append(evaluate(y))
                    local_calls += 1

                if len(F) < lambda_:
                    return

                F = np.array(F)
                gen += 1
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]

                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.eye(dim)

                step = (L_inv @ (x_mean - x_old).T).T[0]
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-16:
                    ps = (1 - c_sigma) * ps + c_sigma * step_norm / chi_n * (step / step_norm)
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 50.0)

                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * (x_mean - x_old) / max(sigma, 1e-10)

                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank_1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (Y_sorted[i] - x_old) / max(sigma, 1e-10)
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * (dim + 1 - c_1) - c_mu - c_c * (1 - c_c) * delta_hsig) * C \
                    + c_1 * (rank_1 - C) + c_mu * rank_mu

                C = (C + C.T) / 2
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C = C + 1e-10 * np.eye(dim)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)

                if sigma < 1e-10 or (gen > 50 and np.ptp(F_sorted[:mu]) < 1e-8):
                    break

            return x_mean, f_best

        # Phase 1: Differential Evolution for global search
        rng = np.random.default_rng(42)
        pop_size = 12
        F_mut = 0.8
        CR = 0.5
        de_budget = int(budget * 0.7)  # 70% for DE, 30% for CMA-ES

        # Initialize population
        pop = []
        for _ in range(pop_size):
            x = rng.uniform(-5, 5, dim)
            f = evaluate(x)
            pop.append([x, f])

        pop.sort(key=lambda item: item[1])

        # DE loop
        while calls_made < de_budget and time.time() < deadline:
            x_best_pop = pop[0][0]
            new_pop = []

            for i in range(pop_size):
                # Select distinct indices
                indices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(indices, 3, replace=False)

                # Mutation: best/1/bin
                v = x_best_pop + F_mut * (pop[r1][0] - pop[r2][0])

                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][0][j]

                trial = np.clip(trial, -5, 5)

                if calls_made >= budget:
                    new_pop.append(pop[i])
                    continue

                f_trial = evaluate(trial)

                # Selection
                if f_trial <= pop[i][1]:
                    new_pop.append([trial, f_trial])
                else:
                    new_pop.append(pop[i])

            pop = new_pop
            pop.sort(key=lambda item: item[1])

        # Phase 2: CMA-ES local refinement around best point
        if x_best is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining >= 20:  # Need at least one full CMA-ES generation
                rng_cma = np.random.default_rng(123)
                run_cmaes(x_best.copy(), 0.5, remaining, rng_cma)

        return x_best, f_best

# EVOLVE-BLOCK-END
