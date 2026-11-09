"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start for Rastrigin rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Number of multi-starts
        n_starts = 3
        # Budget per start
        budget_per_start = max(1, (budget - n_starts) // n_starts)

        best_x = None
        best_fx = np.inf

        for start_idx in range(n_starts):
            if time.time() > deadline - 10:
                break

            # Initialize center
            if start_idx == 0:
                center = np.zeros(dim)
            else:
                center = rng.uniform(-4, 4, dim)

            # CMA-ES parameters
            lambda_ = 4 + int(3 * np.log(dim))  # population size
            mu = lambda_ // 2  # number of selected individuals

            # Initial sigma
            sigma = 2.0

            # CMA-ES state
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)  # sqrt of inverse covariance
            C_sqrt = np.eye(dim)      # sqrt of covariance
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Learning rates
            c_sigma = (mu + 0.5) / (dim + 4)
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))

            # Expected norm of standard normal
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Population
            x_best = center.copy()
            fx_best = func(x_best)
            calls_used = 1

            # Main CMA-ES loop
            gen = 0
            while calls_used < budget_per_start:
                if time.time() > deadline - 5:
                    break

                # Generate population
                pop = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    # Apply covariance transformation
                    # For efficiency, use eigendecomposition or Cholesky
                    # Here we use a simplified approach
                    y = C_sqrt @ z
                    x_i = x_best + sigma * y
                    x_i = np.clip(x_i, -5, 5)
                    pop.append(x_i)

                # Evaluate population
                fvals = []
                for x_i in pop:
                    if time.time() > deadline - 5:
                        fvals.append(np.inf)
                        continue
                    fi = func(x_i)
                    fvals.append(fi)
                    calls_used += 1

                fvals = np.array(fvals)
                pop = np.array(pop)

                # Sort by fitness
                idx = np.argsort(fvals)
                pop = pop[idx]
                fvals = fvals[idx]

                # Update best
                if fvals[0] < fx_best:
                    fx_best = fvals[0]
                    x_best = pop[0].copy()

                if fx_best < best_fx:
                    best_fx = fx_best
                    best_x = x_best.copy()

                # Weighted mean
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                m = np.sum(weights[:, None] * pop[:mu], axis=0)

                # Update step size control
                d_sigma = 1 + 2 * max(0, (mu / dim - 1) / (dim + 2)) + c_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu) * (C_sqrt @ (m - x_best) / sigma)
                norm_ps = np.linalg.norm(p_sigma)

                if norm_ps / (1 + 0.3 * (1 - c_sigma)) > 1.4 * chi_n:
                    sigma *= np.exp(c_sigma * d_sigma * (norm_ps / chi_n - 1))

                # Update evolution path
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu) * (m - x_best) / sigma

                # Update covariance matrix
                delta_c = (1 + 2 * max(0, np.sqrt(c_c / (1 - c_c)) - 1)) * (np.linalg.norm(p_c) ** 2 / dim - 1)
                rank1 = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = C_sqrt @ (pop[i] - m) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)

                C = (1 + c_1 * (delta_c - 1) + c_mu) * C + c_1 * rank1 - c_mu * rank_mu

                # Ensure C is positive definite
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    # Fallback: reset covariance
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)

                # Update center
                x_best = m

                # Check for convergence
                if np.linalg.norm(m - x_best) < 1e-10:
                    break

                gen += 1

                # If we've made good progress, consider stopping early
                if fx_best < best_fx + 1e-6 and gen > 50:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
