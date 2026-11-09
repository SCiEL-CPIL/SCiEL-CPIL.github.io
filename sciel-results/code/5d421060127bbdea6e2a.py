"""Rewritten optimizer using CMA-ES for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for continuous box-constrained optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initialize parameters
        x0 = np.zeros(n)  # start at origin, likely near center of search space
        f0 = func(x0)
        calls = 1

        # CMA-ES parameters
        # lambda: number of samples per generation
        # For dim=5, lambda=4*floor(1.5+3/dim) = 4*floor(2.1) = 4*2 = 8
        # But for small dims, we can use a smaller population
        lam = max(4, 4 * int(1.5 + 3.0 / n))
        # Ensure we don't exceed budget
        if lam * 2 > budget:
            lam = max(2, budget // 2)

        # Step size
        sigma = 2.0

        # Weights: log-weighted
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_eigenvalues = np.ones(n)
        C_eigenvectors = np.eye(n)
        dcor = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best solution
        best_x = x0.copy()
        best_f = f0

        # Initial eigen decomposition
        C_eigenvalues, C_eigenvectors = np.linalg.eigh(C)
        C_eigenvalues = np.clip(C_eigenvalues, 1e-20, None)

        while calls < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lam, n))
            F = np.zeros(lam)

            for i in range(lam):
                z = np.random.standard_normal(n)
                # Transform using eigen decomposition: C^{1/2} = V * diag(sqrt(lambda)) * V^T
                y = x0 + sigma * (C_eigenvectors * (np.sqrt(C_eigenvalues) * (C_eigenvectors.T @ z)))
                y = np.clip(y, -5, 5)
                F[i] = func(y)
                Y[i] = y
                calls += 1

                # Update best
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()

                # Check time limit inside loop
                if time.time() > deadline:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Mean of top mu individuals
            m = Y[:mu].T @ weights
            x0 = x0 + m

            # Clip to box
            x0 = np.clip(x0, -5, 5)

            # Compute difference vector
            d = Y[:mu].T @ weights - m

            # Update p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (C_eigenvectors @ (C_eigenvalues ** -0.5)) @ (C_eigenvectors.T @ d)
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * max(1, calls / lam)))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update sigma
            sigma *= np.exp(cs / dcor * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update p_c
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * d

            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(Y[i] - x0, Y[i] - x0)

            C = (1 + c1 * hsig * (2 - cc - 1) + cmu - c1) * C + c1 * rank_one + cmu * rank_mu

            # Ensure positive semi-definite
            C = (C + C.T) / 2

            # Eigen decomposition (do less frequently for speed)
            if calls % (2 * lam) < lam:
                C_eigenvalues, C_eigenvectors = np.linalg.eigh(C)
                C_eigenvalues = np.clip(C_eigenvalues, 1e-20, None)

        return best_x, best_f
# EVOLVE-BLOCK-END
