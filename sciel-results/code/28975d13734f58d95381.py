"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Initialize population-based CMA-ES
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        x = np.zeros(n)  # start at origin
        sigma = 2.0
        C = np.eye(n)  # covariance matrix
        B = np.eye(n)  # eigenvectors
        d = np.ones(n)  # eigenvalues
        step_size = 1.0

        # Cumulative step length adaptation
        ps = np.zeros(n)
        cs = (np.sum(np.arange(lam + 1)) / 2 - 1) / (n + 2) + 0.5
        cm = (2 + np.sqrt(2 * max(0, lam - 2)) / (n + 2)) / 2
        damps = (n + 2 + 2 * min(1, 1 / (lam - n - 1))) / (n + 4 + 2 * lam / n)

        # Track best
        best_x = x.copy()
        best_fx = func(x)
        calls_used = 1

        for gen in range(1, budget):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate population
            Y = np.zeros((lam, n))
            F = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget or time.time() > deadline:
                    break
                zi = rng.standard_normal(n)
                # Transform using C = B * diag(d) * B^T
                wi = B @ (d * (B.T @ zi))
                yi = x + sigma * wi
                yi = np.clip(yi, -5, 5)
                F[i] = func(yi)
                Y[i] = yi
                calls_used += 1

            if calls_used >= budget or time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(F)
            mu = lam // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)

            # Update mean
            y_old = x.copy()
            x_new = np.sum(weights[:, None] * Y[idx[:mu]], axis=0)
            x = x_new

            # Update step size (CSA)
            z = B.T @ (d ** (-0.5)) @ (B.T @ (x - y_old) / sigma)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (x - y_old)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - gen + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs) ** (2 * (calls_used - gen + 1)))
            sigma *= np.exp(cm / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used - gen + 1))) - np.sqrt(1 - (1 - cs) ** 2)))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance (rank-mu update)
            if hsig:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (x - y_old)
                C = (1 + 2 * cm * cs * hsig * np.outer(ps, ps)) * C + cm * (hsig - 1) * C
            else:
                C = (1 + 2 * cm * cs * hsig * np.outer(ps, ps)) * C + cm * (hsig - 1) * C

            # Rank-mu update
            yw = (x - y_old) / sigma
            H = np.sum(weights[:, None, None] * np.outer(Y[idx[:mu]] - y_old, Y[idx[:mu]] - y_old) / sigma**2, axis=0)
            C = (1 + cm * (hsig - 1) + 2 * cm * mueff) * C + cm * (2 - hsig - (1 + 2 * cm * cs * hsig) * mueff) * np.outer(yw, yw) + cm * (2 - hsig) * H

            # Ensure positive definiteness
            C = 0.5 * (C + C.T)
            try:
                B, d = np.linalg.eigh(C)
                d = np.clip(d, 1e-12, None)
            except np.linalg.LinAlgError:
                B = np.eye(n)
                d = np.ones(n)

            # Track best
            min_idx = np.argmin(F)
            if F[min_idx] < best_fx:
                best_fx = F[min_idx]
                best_x = Y[idx[min_idx]].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
