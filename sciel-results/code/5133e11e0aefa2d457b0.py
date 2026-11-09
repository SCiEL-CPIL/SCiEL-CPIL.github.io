"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, cumulative step-size control,
    starting near origin to exploit locality of f14 minimum."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 4  # population size
        mu = lam // 2  # number of parents
        # Weights
        w = np.zeros(mu)
        w[0] = np.log(mu + 0.5) - np.mean(np.log(np.arange(1, mu + 1)))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Start near origin
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        sigma = 2.0
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        hs = np.zeros(dim)

        # Precompute for eigendecomposition update
        for gen in range(budget - call_count):
            # Time guard
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = np.clip(x + sigma * (B @ (D * z)), -5, 5)
                fy = func(y)
                call_count += 1
                offspring.append((fy, y))

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_fx:
                best_fx = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Save old mean
            x_old = x.copy()

            # New mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * offspring[i][1]
            x = x_new

            # Cumulation path for sigma
            y_sigma = (x - x_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * B.T @ y_sigma

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance matrix every 1/(c1+cmu) generations
            if (gen + 1) % max(1, int(1 / (c1 + cmu))) == 0 or gen % 20 == 0:
                # Compute delta_h
                delta_h = (np.linalg.norm(ps) / chi_n) > (1.4 + 2 / (dim + 1))

                # Rank-mu update
                delta_sigma = np.zeros((dim, dim))
                for i in range(mu):
                    delta_sigma += w[i] * np.outer(offspring[i][1] - x_old, offspring[i][1] - x_old)

                # Rank-one update
                delta_1 = np.outer(ps, ps)

                # Update B and D via eigenvalue tracking
                # For simplicity, do a full eigendecomposition periodically
                C = B @ np.diag(D ** 2) @ B.T
                C_new = (1 + c1 * delta_h + cmu * delta_h) * C + \
                        c1 * (delta_1 - (1 - delta_h) * C) + \
                        cmu * delta_sigma

                # Ensure positive semi-definite
                C_new = (C_new + C_new.T) / 2

                # Eigendecomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C_new)
                    eigvals = np.maximum(eigvals, 1e-12)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except:
                    # Fallback: keep old B, D
                    pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
