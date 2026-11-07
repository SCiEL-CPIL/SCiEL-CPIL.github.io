"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """(mu, lambda)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def count_func(x):
            self._call_count += 1
            fx = func(x)
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Use (mu, lambda)-ES with mu=4, lambda=20 for dim=5
        mu = 4
        lam = 20
        x = rng.uniform(-4, 4, self.dim)
        fx = count_func(x)
        self._best_f = fx
        self._best_x = x.copy()
        sigma = 1.0

        # Weighted recombination weights
        weights = np.log(mu / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Update rate for sigma
        cs = (mueff + 1.5) / (self.dim + mueff + 3)
        ds = 1 + 2 * max(0, (mueff - 1) / (self.dim + 1)) + cs

        # Update rate for covariance
        c1 = (2 * (mueff - 2 + 1 / mueff)) / ((self.dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2) ** 2 + 2 * mueff))

        # Path accumulators
        ps = np.zeros(self.dim)
        pc = np.zeros((self.dim, self.dim))
        C = np.eye(self.dim)
        chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        while self._call_count < self.budget:
            # Generate offspring
            offspring = []
            for i in range(lam):
                # Sample from multivariate normal with covariance C and std sigma
                # Use Cholesky decomposition for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(self.dim)
                z = rng.standard_normal(self.dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = count_func(y)
                offspring.append((fy, y))
                if self._call_count >= self.budget:
                    break

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update mean using weighted recombination of best mu
            x_new = np.zeros(self.dim)
            for i in range(min(mu, len(offspring))):
                x_new += weights[i] * offspring[i][1]
            x_new = np.clip(x_new, -5, 5)

            # Update path for sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (x_new - x))

            # Check if ps is within confidence interval
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (self._call_count / (cs * self.dim))) < (1.4 + 2 / (self.dim + 1)) * chiN:
                pc = (1 - cs) * pc + cs * (C @ (x_new - x) / sigma - ps)

            # Update covariance matrix
            rank_one = np.outer(x_new - x, x_new - x)
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(min(mu, len(offspring))):
                yi = offspring[i][1] - x
                rank_mu += weights[i] * np.outer(yi, yi)

            c_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
            sigma *= np.exp((cs / ds) * (np.linalg.norm(ps) / chiN - 1))

            C = (1 + c1 + cmu - c_sigma) * C + c_sigma * pc @ pc + c1 * rank_one + cmu * rank_mu

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = x_new

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
