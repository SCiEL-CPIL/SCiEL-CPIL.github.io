"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 * int(np.ceil(3.0 / dim))  # population size
        mu = lambda_ // 2  # number of best solutions to update weights

        # Initial parameters
        x = np.zeros(dim)
        sigma = 2.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        pc = np.zeros(dim)  # evolution path for C
        ps = np.zeros(dim)  # evolution path for sigma

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Track best
        best_x = x.copy()
        best_f = func(x)
        call_count = 1

        # Initial population
        for gen in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                Y[i] = x + sigma * (np.linalg.cholesky(C) @ np.random.standard_normal(dim))
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = func(Y[i])
                call_count += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()

            if call_count >= budget:
                break

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]

            # Update mean
            x_new = np.sum(weights[:, None] * Y[:mu], axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / np.linalg.norm(np.sqrt(np.diag(C))) * (np.linalg.inv(np.sqrt(np.diag(C))) @ (x_new - x))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) < (1 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x) / np.linalg.norm(np.sqrt(np.diag(C)))

            # Update covariance
            delta = (1 + 2 * max(0, hsig - 1)) * cc * (2 - cc) if hsig else 0
            C = (1 + c1 + cmu + delta - c1 - cmu) * C + c1 * (np.outer(pc, pc) - C) + cmu * np.sum(weights[:, None, None] * np.outer(Y[:mu] - x, Y[:mu] - x), axis=0)

            # Update sigma
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt((dim + 1) / (dim - 1)) - 1))
            sigma = np.clip(sigma, 1e-8, 1e8)

            x = x_new

            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(np.diag(C)) < 1e-10):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
