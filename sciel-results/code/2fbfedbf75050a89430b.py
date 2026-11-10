"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290.0
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        mu = max(1, dim // 2)
        cmu = 2 / ((dim + 1.3) ** 2 + mu)
        c1 = (2 - cmu) / ((dim + 1.9) ** 3 + mu)
        cs = (mu + 1.8) / (dim + mu + 5)
        c_sigma = min(1.0, 0.3 * (1 + cmu))
        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initial point: try center first, then random
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1

        # If center is bad, try a random point
        rng = np.random.default_rng(42)
        x_alt = rng.uniform(-4, 4, dim)
        f_alt = func(x_alt)
        calls += 1

        if f_alt < f0:
            x0, f0 = x_alt, f_alt

        best_x = x0.copy()
        best_f = f0

        # CMA-ES state
        x = x0.copy()
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = 1.0 / chi * (1 - c1)

        # Weighted recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        n_iter = 0
        while calls < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            for i in range(mu):
                z = rng.standard_normal(dim)
                y = x + sigma * z
                fy = func(y)
                calls += 1
                offspring.append((fy, y))

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Weighted mean
            m = np.zeros(dim)
            for i in range(mu):
                m += w[i] * offspring[i][1]

            # Step size adaptation
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi * (C @ (m - x) / sigma)
            sigma *= np.exp(c_sigma / chi * (np.linalg.norm(p_sigma) / h_sigma - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Covariance matrix adaptation
            if np.linalg.norm(p_sigma) / h_sigma < 1.4 + 2 / (dim + 1):
                h1 = 1.0
            else:
                h1 = 0.0

            p_c = (1 - cs) * p_c + h1 * np.sqrt(cs * (2 - cs) * mueff) * (m - x) / sigma
            C = (1 + c1 * h1 * (1 - h1) - c1) * C + c1 * (np.outer(p_c, p_c) - (1 - h1) * C)

            # Update mean
            x = m

            n_iter += 1
            if n_iter > 200:
                break

        # Final polish: local search around best point
        if calls < budget and time.time() < deadline - 5:
            # Use Nelder-Mead style or fine-grained local search
            polish_sigma = 0.1
            for _ in range(min(50, budget - calls)):
                if time.time() > deadline - 2:
                    break
                for _ in range(5):
                    y = np.clip(best_x + polish_sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        polish_sigma *= 0.5
                    else:
                        polish_sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
