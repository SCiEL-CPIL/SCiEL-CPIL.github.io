"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Sobol initialization and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        start_time = time.time()
        deadline = start_time + 285.0

        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Phase 1: Sobol sequence for initial global coverage
        from scipy.stats import qmc

        n_sobol = min(25, budget)
        sobol_sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sobol_sampler.random(n_sobol)

        for i in range(n_sobol):
            if calls_made >= budget or time.time() >= deadline:
                break

            x = -5.0 + 10.0 * sobol_points[i]
            x = np.clip(x, -5.0, 5.0)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if calls_made >= budget or best_x is None:
            return best_x, best_f

        # Phase 2: CMA-ES
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda
        # Initial sigma: roughly 1/3 of the box width per dimension
        sigma = 1.5

        # CMA-ES state
        mean = best_x.copy()
        C = np.eye(dim)  # covariance matrix
        # Precompute C^{-1/2} for sampling
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # Learning rates
        mu = pop_size // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Max iterations for CMA phase
        max_cma_iters = (budget - calls_made) // pop_size + 10

        for it in range(max_cma_iters):
            if calls_made >= budget or time.time() >= deadline - 10:
                break

            # Check if we should stop (convergence)
            if sigma < 1e-8 or (it > 5 and np.all(np.abs(mean - best_x) < 1e-4)):
                break

            # Sample population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5.0, 5.0)
                offspring.append((y, i))

            # Evaluate offspring
            evals = []
            for y, i in offspring:
                if calls_made >= budget or time.time() >= deadline - 5:
                    break
                fy = func(y)
                calls_made += 1
                evals.append((fy, y, i))

            if len(evals) < pop_size:
                break

            # Sort by fitness
            evals.sort(key=lambda x: x[0])
            selected = evals[:mu]

            # Update mean
            new_mean = sum(weights[i] * selected[i][1] for i in range(mu))
            new_mean = np.clip(new_mean, -5.0, 5.0)

            # Update covariance matrix
            # Compute C^{1/2} for path updates
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

            # Update evolution paths
            y_weighted = sum(weights[i] * (selected[i][1] - mean) for i in range(mu))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt.T @ y_weighted)
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) / chi_n < 1.4 + 2 / (dim + 1)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_weighted

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(selected[i][1] - mean, selected[i][1] - mean) for i in range(mu))

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Renormalize to ensure positive semi-definite
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Update C_inv_sqrt
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = min(sigma, 5.0)  # Cap sigma

            # Update mean
            mean = new_mean

            # Track best
            for fy, y, i in selected:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        # Phase 3: Final polish with (1+1)-ES using small sigma
        if calls_made < budget and best_x is not None and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            max_no_improve = 15

            while calls_made < budget and time.time() < deadline - 2:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    sigma = min(sigma * 1.1, 0.5)
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.8, 0.001)

                if no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
