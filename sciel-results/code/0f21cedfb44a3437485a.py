"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initialize with a few random restarts to find a good starting point
        rng = np.random.default_rng(42)

        # First, do a quick random search to find a decent starting point
        x_best = None
        f_best = np.inf
        for i in range(min(20, budget)):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()
            if time.time() > deadline:
                break

        if x_best is None:
            x_best = np.zeros(dim)
            f_best = func(x_best)

        # CMA-ES parameters
        n = dim
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = x_best.copy()
        sigma = 1.0  # initial step size
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        # Precompute C^(1/2) for sampling
        C_half = C.copy()

        calls_made = 20  # already made 20 calls in random search
        remaining = budget - calls_made

        # Main CMA-ES loop
        while remaining > 0 and time.time() < deadline:
            # Generate population
            Y = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                Y[i] = x + sigma * (C_half @ z)
                # Clip to bounds
                Y[i] = np.clip(Y[i], -5, 5)

            # Evaluate population
            F = np.zeros(lambda_)
            for i in range(lambda_):
                F[i] = func(Y[i])
                calls_made += 1
                remaining -= 1
                if time.time() > deadline:
                    # Fill remaining with best known
                    for j in range(i + 1, lambda_):
                        F[j] = F[i]
                    break

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update mean
            x_old = x.copy()
            x_new = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x = x_new

            # Update step-size control (CSA)
            h_sigma = (
                np.sqrt((1 - (1 - c_sigma) ** (2 / budget)) / (1 - (1 - c_sigma) ** (-2 / budget)))
                * chi_n
                * np.linalg.norm(C_half @ np.linalg.inv(C_half) @ (x - x_old) / sigma)
            ) if sigma > 0 else 0
            # Simplified h_sigma
            h_sigma = chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 / max(remaining, 1)))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / h_sigma * np.linalg.norm(x - x_old) / sigma if h_sigma > 0 else p_sigma

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)
            h_c = (
                np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (1 / max(remaining, 1))) / chi_n
                < 1.4 + 2 / (n + 1)
            )
            C = (
                (1 - c_1 - c_mu + (1 - h_c) * c_c * (2 - c_c)) * C
                + c_1 * (np.outer(p_c, p_c) - C)
                + c_mu / sigma ** 2 * np.sum(w[:, None, None] * np.outer(Y_sorted[:mu] - x_old, Y_sorted[:mu] - x_old), axis=0)
            )

            # Symmetrize and update C_half
            C = (C + C.T) / 2
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2
            C_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Update best
            if F_sorted[0] < f_best:
                f_best = F_sorted[0]
                x_best = Y_sorted[0].copy()

            # Check convergence
            if np.std(F_sorted[:mu]) < 1e-10 and np.linalg.norm(x - x_old) < 1e-10:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
