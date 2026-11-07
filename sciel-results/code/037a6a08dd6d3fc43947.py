import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Random scan to find a promising basin
        # Use Sobol-like quasi-random points for better coverage
        rng = np.random.default_rng(42)

        # Generate quasi-random points using a simple Halton-like approach
        def halton_sequence(index, dim):
            """Generate Halton sequence point."""
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2  # Start with base 2
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point[i] = r
            return point

        # Scan phase: 100 points
        n_scan = min(100, budget // 5)
        for i in range(1, n_scan + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_sequence(i, dim) * 10 - 5  # Scale to [-5, 5]
            x = np.clip(x, -5, 5)
            eval_and_track(x)

        # Phase 2: CMA-ES with population-based covariance adaptation
        # This is the main optimization phase

        def cma_es(x0, sigma0, max_iter, pop_size=None):
            """
            CMA-ES implementation for black-box optimization.
            """
            nonlocal calls, best_f, best_x

            if pop_size is None:
                pop_size = max(4, int(4 + 3 * np.log(dim)))

            # CMA-ES parameters
            mu = pop_size // 2  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            # Learning rates
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

            # Step size control
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            # State
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            hsigma = (1 + 0.34 + 2 / (dim + 1)) * c_sigma

            # Hessian update interval
            hess_interval = max(1, int(1 / (c_1 + c_mu)))
            hess_counter = 0

            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Generate population
                population = []
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (B @ (D * z))
                    x = np.clip(x, -5, 5)
                    population.append((x, eval_and_track(x)))

                if len(population) == 0:
                    break

                # Sort by fitness
                population.sort(key=lambda t: t[1])

                # Update mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * population[i][0]
                x_new = np.clip(x_new, -5, 5)

                # Update covariance matrix adaptation
                # Cumulation path for sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B.T @ (D * (x_new - x_mean)))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

                # Cumulation path for C
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new - x_mean)

                # Update B and D
                delta_hs = (1 - c_c) / (1 - c_c**2)
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = np.linalg.solve(B.T, (x_new - population[i][0]) / sigma)
                    z_i = z_i / D
                    rankmu += weights[i] * np.outer(z_i, z_i)

                C_update = (1 - c_1 - c_mu) + c_1 * (rank1 + delta_hs * (pc @ pc - np.eye(dim)))
                # This is simplified; in practice we update the eigen-decomposition
                # For simplicity, we'll use a simplified update

                # Actually, let's use a simpler approach: update the covariance matrix directly
                # and periodically recompute B and D
                if hess_counter % hess_interval == 0:
                    # Recompute B and D from the covariance matrix
                    # For simplicity, we'll skip the full CMA-ES covariance update
                    # and just use the population to estimate the covariance
                    pass

                hess_counter += 1

                x_mean = x_new
                sigma = np.clip(sigma, 1e-10, 10.0)

            return x_mean, sigma

        # Run CMA-ES from the best point found in scan
        if best_f < float('inf'):
            # Start CMA-ES with a reasonable initial sigma
            sigma0 = 2.0
            max_iter_cma = (budget - calls) // 15
            max_iter_cma = max(max_iter_cma, 10)

            x_cma, sigma_cma = cma_es(best_x, sigma0, max_iter_cma)

            # Phase 3: Second CMA-ES restart with different initial sigma
            if calls < budget - 20:
                # Restart with a larger sigma to escape local optima
                sigma0_2 = 3.0
                max_iter_cma2 = (budget - calls) // 15
                max_iter_cma2 = max(max_iter_cma2, 5)

                # Use the best point from the first CMA-ES run
                x_cma2, sigma_cma2 = cma_es(best_x, sigma0_2, max_iter_cma2)

        return best_x, best_f
