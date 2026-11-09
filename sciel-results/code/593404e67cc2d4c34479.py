"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(x, -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_left, max_iter=200):
            nonlocal calls, best_x, best_f, deadline

            n = self.dim
            # Population size
            lam = 4 + int(3 * np.log(n))
            # Weights
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / weights.sum()

            # Strategy parameters
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            # State
            mean = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)  # Initialize covariance matrix!
            B = np.eye(n)
            D = np.ones(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            rng = np.random.default_rng()

            # Initial population
            pop = np.array([mean + sigma * (B @ (D * rng.standard_normal(n))) for _ in range(lam)])
            pop_f = np.array([eval_func(x) for x in pop])

            it = 0
            while calls < budget_left and it < max_iter:
                if time.time() > deadline:
                    break
                it += 1

                # Sort
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                # Update mean
                old_mean = mean.copy()
                mean = weights[:mu] @ pop[:mu]

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D @ (B.T @ (mean - old_mean) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / D @ (B.T @ (mean - old_mean) / sigma)

                # Update covariance matrix
                delta_d = cc * hsig + (1 - cc)
                delta_d = min(delta_d, 1.0)
                rank_one = np.outer(pc, pc)
                rank_mu = sum(weights[i] * np.outer(pop[i] - old_mean, pop[i] - old_mean) for i in range(mu))
                C = (1 + c1 * delta_d - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Re-orthonormalize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 1e3)

                # Generate new population
                for i in range(lam):
                    z = rng.standard_normal(n)
                    pop[i] = mean + sigma * (B @ (D * z))
                    pop_f[i] = eval_func(pop[i])

                # Convergence check
                if np.linalg.norm(mean - old_mean) / max(sigma, 1e-12) < 1e-10:
                    break

            return mean, sigma, it

        # Main loop: run CMA-ES
        rng = np.random.default_rng(42)

        # Initial best guess: try origin and a few random points
        init_points = [np.zeros(self.dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-self.bound, self.bound, self.dim))

        for p in init_points:
            eval_func(p)

        # Run CMA-ES starting from the current best
        x0 = best_x.copy()
        sigma0 = 2.0  # Start with a moderate sigma

        x0, sigma0, _ = cma_es(x0, sigma0, self.budget)

        # If we have budget left, do a local refinement with Nelder-Mead
        if calls < self.budget and time.time() < deadline:
            x0 = best_x.copy()
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x0
            step = 0.1 * max(sigma0, 0.1)
            for i in range(1, self.dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += step

            simplex_f = np.array([eval_func(x_i) for x_i in simplex])

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            nm_iter = 0
            max_nm_iter = 100

            while calls < self.budget and nm_iter < max_nm_iter:
                if time.time() > deadline:
                    break
                nm_iter += 1

                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                if abs(simplex_f[0] - simplex_f[1]) < 1e-14:
                    break

                centroid = np.mean(simplex[:self.dim], axis=0)
                x_r = centroid - alpha * (centroid - simplex[-1])
                f_r = eval_func(x_r)

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    if f_r < simplex_f[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = eval_func(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            for i in range(1, self.dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex_f[i] = eval_func(simplex[i])
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                        f_c = eval_func(x_c)
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            for i in range(1, self.dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex_f[i] = eval_func(simplex[i])

        return best_x, best_f
# EVOLVE-BLOCK-END
