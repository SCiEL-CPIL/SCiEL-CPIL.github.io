"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(123)

        # Initialize at origin (minimizer is in [-4,4]^5, so origin is reasonable)
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_pop = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
        sigma = 1.0  # initial step size
        c1 = (2 + 2 / (n + 1)) / ((n + 2) ** 2 + 2)  # covariance adaptation rate
        cmu = min(0.03 * (lambda_pop / 2 - 1), 0.16) / ((n + 4) ** 2 + 8)  # mutation adaptation rate
        c_sigma = (mu_sigma + 5) / (n + mu_sigma + 5)  # sigma adaptation rate
        mu_sigma = int(np.floor(lambda_pop / 2))  # number of weights for sigma
        mu = int(np.floor(lambda_pop / 2))  # number of weights for recombination
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))  # expected norm of N(0,I)

        # Weights
        w = np.log(mu + 1) - np.arange(mu)
        w /= w.sum()
        mu_eff = 1 / (w ** 2).sum()
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)

        # Covariance matrix
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        # Precompute eigendecomposition for efficient updates
        evals, evecs = np.linalg.eigh(C)
        C_inv_sqrt = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T

        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        d_cor = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Initial population
        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_pop, n))
            for j in range(lambda_pop):
                ys[j] = x + sigma * (C_inv_sqrt @ rng.standard_normal(n))
                ys[j] = np.clip(ys[j], -5, 5)

            # Evaluate population
            fvals = np.zeros(lambda_pop)
            for j in range(lambda_pop):
                fvals[j] = func(ys[j])

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_fx:
                best_fx = fvals[min_idx]
                best_x = ys[min_idx].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = ys[order]
            fvals_sorted = fvals[order]

            # Mean of best mu individuals
            y_mean = np.zeros(n)
            for j in range(mu):
                y_mean += w[j] * ys_sorted[j]

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (y_mean - x) / sigma)

            # Update x
            x = y_mean

            # Update p_c
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (i + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mu_eff) * (y_mean - x) / sigma

            # Update C
            delta_h = (1 - hsig) * c1 * (1 - c1)
            C = (1 + c1 * (cmu + delta_h - c1)) * C
            C -= c1 * (1 + cmu + delta_h - c1) * np.outer(p_c, p_c)
            for j in range(mu):
                z = C_inv_sqrt @ (ys_sorted[j] - y_mean)
                C += cmu * w[j] * np.outer(z, z)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-10)
            C = evecs @ np.diag(evals) @ evecs.T
            C_inv_sqrt = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T

            # Update x to be the weighted mean of top individuals
            x = y_mean

        return best_x, best_fx
# EVOLVE-BLOCK-END
