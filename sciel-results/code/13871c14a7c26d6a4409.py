"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES optimizer for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * dim)  # population size
        # For dim=5: lambda = 4 * (1+15) = 64
        # But we have limited budget, so use smaller population
        # Standard: mu = lambda // 2
        mu = lambda_ // 2

        # Step size initialization
        sigma = 2.0

        # Initial point
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)

        # Covariance matrix
        C = np.eye(dim)
        # Square root of C
        Csqr = np.eye(dim)
        Csqr_inv = np.eye(dim)

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        # Standard formulas for CMA-ES
        mu_eff = sum(1.0 / i for i in range(1, mu + 1)) / mu
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = c_sigma / (1 + c_c)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Weights for the mu best points
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Recalculate learning rates with correct mu_eff
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = c_sigma / (1 + c_c)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Normalization constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best solution
        x_best = x.copy()
        fx_best = fx

        calls_made = 1

        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate lambda candidates
            offspring = []
            fitnesses = []

            for i in range(lambda_):
                if time.time() > deadline:
                    break

                # Sample from multivariate normal using current covariance
                z = rng.standard_normal(dim)
                y = Csqr @ z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)

                fxi = func(xi)
                calls_made += 1
                offspring.append(xi)
                fitnesses.append(fxi)

                if fxi < fx_best:
                    fx_best = fxi
                    x_best = xi.copy()

            if len(offspring) == 0:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_fitnesses = [fitnesses[i] for i in indices]

            # Update mean using weighted combination of mu best
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * sorted_offspring[i]

            x = x + y_w

            # Update sigma (step size)
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += w[i] * (sorted_offspring[i] - x) / sigma

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made // lambda_ + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance matrix
            y_c = np.zeros(dim)
            for i in range(mu):
                y_c += w[i] * (sorted_offspring[i] - x) / sigma

            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            # Rank-one and rank-mu update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + \
                c_mu * sum(w[i] * (sorted_offspring[i] - x) @ (sorted_offspring[i] - x) / sigma ** 2 for i in range(mu))

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                Csqr, _ = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                Csqr_inv = np.linalg.inv(Csqr)
            except np.linalg.LinAlgError:
                # Fallback: reinitialize
                C = np.eye(dim)
                Csqr = np.eye(dim)
                Csqr_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

            # If no improvement in a while, restart with smaller sigma
            # This is a simple adaptation

        return x_best, fx_best
# EVOLVE-BLOCK-END
