"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart for Rastrigin, tuned for early progress."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial exploration with structured + random points
        # Use a small grid to cover the space, plus random points
        n_grid = 3  # 3^5 = 243 grid points, but we'll sample a subset
        # Actually, let's use a smarter approach: random + a few structured
        # First, try center and a few random points
        n_initial = 15
        for i in range(n_initial):
            if call_count >= budget:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i < 5:
                # Try points along each axis
                x = np.zeros(dim)
                x[i % dim] = 2.0 if i % 2 == 0 else -2.0
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            fx = call_func(x)
            if fx is None:
                break

        # Phase 2: CMA-ES with restarts
        lambda_ = 10
        mu = 3  # Lighter population for faster iterations
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        def run_cmaes(x_start, sigma_start):
            nonlocal best_x, best_f, call_count, deadline

            x = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            stagnation_count = 0
            last_best_f = best_f
            no_improve_generations = 0

            while call_count < budget and time.time() < deadline:
                # Generate offspring
                offspring = []
                offspring_f = []

                for k in range(lambda_):
                    if call_count >= budget or time.time() >= deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = call_func(y)
                    if fy is None:
                        break
                    offspring.append(y)
                    offspring_f.append(fy)

                if len(offspring) < mu:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(offspring_f)
                selected_x = [offspring[i] for i in sorted_idx[:mu]]
                selected_f = [offspring_f[i] for i in sorted_idx[:mu]]

                # Check for improvement
                if best_f < last_best_f - 1e-10:
                    no_improve_generations = 0
                    last_best_f = best_f
                else:
                    no_improve_generations += 1

                # Stagnation restart: if no improvement for 20 generations,
                # restart with larger sigma to escape local minima
                if no_improve_generations > 20:
                    # Increase sigma to explore more broadly
                    sigma = min(sigma * 2.0, 5.0)
                    x = best_x.copy()
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    no_improve_generations = 0
                    last_best_f = best_f
                    continue

                # Weighted mean
                m = sum(w * xk for w, xk in zip(weights, selected_x))

                # Update evolution paths
                diff = m - x
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff

                # Sigma adaptation
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (no_improve_generations + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-4)

                # Covariance matrix update
                delta_d = (1 + c_mu * (mueff - 1)) * diff
                rank_mu = np.outer(delta_d, delta_d)

                h_c = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (no_improve_generations + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0

                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * np.eye(dim)) + c_mu * rank_mu

                # Update C_inv_sqrt periodically
                if no_improve_generations % 3 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                # Update current point
                x = m

        # Run CMA-ES from the best initial point
        # Start with sigma=1.5 which is good for Rastrigin's periodicity
        run_cmaes(best_x, 1.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
