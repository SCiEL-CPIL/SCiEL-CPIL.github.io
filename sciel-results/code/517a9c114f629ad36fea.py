"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f6 (Attractive Sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # lambda = 4 + 3*ln(n)
        # For dim=5: lambda = 4 + 3*1.609 ≈ 8.8 -> 8
        pop_size = 8

        mu = pop_size // 2  # number of selected individuals
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        sigma = 2.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # Precompute sqrt of C
        D, V = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))
        sqrtC = V @ np.diag(D) @ V.T
        inv_sqrtC = V @ np.diag(1.0 / np.maximum(D, 1e-20)) @ V.T

        best_x = x.copy()
        best_fx = fx

        # Track best value for scoring
        calls_made = 1

        while calls_made < budget:
            # Check time deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            fitness = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x + sigma * (sqrtC @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                fitness.append(fy)

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            # Sort by fitness
            indices = np.argsort(fitness)
            offspring = [offspring[i] for i in indices]
            fitness = [fitness[i] for i in indices]

            # Compute mean of selected individuals
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * offspring[i]

            # Update step-size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (inv_sqrtC @ (y_mean - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - pop_size)))
                    < (1 + 2 / (dim + 1)) * np.sqrt(mu_eff + 1 - mu_eff))
            pc = (1 - cc) * pc + hsig * cc * (y_mean - x) / sigma

            # Update covariance matrix
            delta = np.zeros(dim)
            for i in range(mu):
                delta += weights[i] * (offspring[i] - x) / sigma
            C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu * np.outer(delta, delta))

            # Ensure symmetry
            C = (C + C.T) / 2.0

            # Update inverse sqrt
            D, V = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            sqrtC = V @ np.diag(D) @ V.T
            inv_sqrtC = V @ np.diag(1.0 / np.maximum(D, 1e-20)) @ V.T

            # Update sigma
            target = np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma *= target
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = y_mean
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
