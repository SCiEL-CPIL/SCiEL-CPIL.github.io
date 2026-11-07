"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive sigma for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3) ** 2 + mueff))
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial point and sigma
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, dim)
        sigma = 2.0

        # Best tracking
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # CMA-ES state
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Precompute sqrt of C for sampling
        def sample_C(sigma_val):
            # Sample z ~ N(0,1), then y = B @ (D * z)
            z = rng.standard_normal(dim)
            return B @ (D * z) * sigma_val

        for gen in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                y = x_mean + sample_C(sigma)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                calls += 1
                if calls >= budget:
                    break

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = func(y)
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) == 0:
                break

            # Sort by fitness
            sorted_idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in sorted_idx]
            sorted_f = [offspring_f[i] for i in sorted_idx]

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_offspring[i]
            x_mean = x_new

            # Update covariance
            v = np.zeros((dim, mu))
            for i in range(mu):
                v[:, i] = (sorted_offspring[i] - x_mean) / sigma

            delta_hs = np.sqrt((1 + 2 * max(0, gen / (ds * (dim + 1)))) * (1 - np.exp(-2 * gen / (ds * (dim + 1)))))
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (C - np.diag(np.diag(C))) * 0) + cmu * (v @ np.diag(weights) @ v.T)

            # Symmetrize
            C = (C + C.T) / 2

            # Update eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)

            # Update p_sigma and p_c
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D * (x_mean - x_new) / sigma))
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < 1.4 + 2 / (dim + 1)

            p_c = (1 - cs) * p_c + h_sig * np.sqrt(cs * (2 - cs) * mueff) * (x_mean - x_new) / sigma

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Ensure C is positive definite
            if gen % 10 == 0:
                try:
                    eigvals_check, _ = np.linalg.eigh(C)
                    if np.min(eigvals_check) < 1e-10:
                        C = C + 1e-8 * np.eye(dim)
                except:
                    C = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
