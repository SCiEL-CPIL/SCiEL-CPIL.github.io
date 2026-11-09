"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f1 (Sphere) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = 4 * int(np.floor(3 + 3 * np.log(n)))  # lambda
        if pop_size < 2:
            pop_size = 2

        # Initial point: center of the box
        x = np.zeros(n)
        fvals = [func(x)]
        best_x = x.copy()
        best_f = fvals[0]

        sigma = 2.0  # Initial step size
        C = np.eye(n)  # Covariance matrix
        ps = np.zeros(n)  # Evolution path for sigma
        pc = np.zeros(n)  # Evolution path for C

        # Adaptation parameters
        mu = pop_size // 2  # Number of weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - weights[1]) / ((n + 2) ** 2 + mu_eff))

        # Boundary handling
        lb = -5.0
        ub = 5.0

        calls_made = 1
        remaining = budget - 1

        while remaining > 0:
            if time.time() >= deadline - 5.0:
                break

            # Generate population
            offspring = []
            fvals_gen = []

            for i in range(pop_size):
                if remaining <= 0:
                    break
                if time.time() >= deadline - 5.0:
                    break

                # Sample from multivariate normal
                z = np.random.multivariate_normal(np.zeros(n), C)
                y = x + sigma * z
                y = np.clip(y, lb, ub)

                fy = func(y)
                calls_made += 1
                remaining -= 1

                offspring.append(y)
                fvals_gen.append(fy)

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) == 0:
                break

            # Sort by fitness
            indices = np.argsort(fvals_gen)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_fvals = [fvals_gen[i] for i in indices]

            # Update mean
            for i in range(min(mu, len(sorted_offspring))):
                x += weights[i] * (sorted_offspring[i] - x)

            # Update evolution paths
            # Standardized selected points
            std_points = []
            for i in range(min(mu, len(sorted_offspring))):
                std_points.append((sorted_offspring[i] - x) / (sigma + 1e-30))

            # Sigma evolution path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 np.sum(weights[:min(mu, len(std_points))], axis=0) * \
                 np.mean(std_points[:min(mu, len(std_points))], axis=0) if len(std_points) > 0 else ps

            # Check for stagnation in sigma
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < \
                         (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff) else 0.0

            # Update covariance matrix
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                 np.mean(std_points[:min(mu, len(std_points))], axis=0) if len(std_points) > 0 else pc

            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1) * C + \
                c_1 * (np.outer(pc, pc) - C) * delta_h + \
                c_mu * np.sum([weights[i] ** 2 for i in range(min(mu, len(std_points)))] *
                               [np.outer(p, p) for p in std_points[:min(mu, len(std_points))]], axis=0)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
