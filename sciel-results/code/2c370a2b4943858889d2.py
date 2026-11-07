"""CMA-ES optimizer for BBOB-style Sphere function f1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10, adapted for box-constrained search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size
        lam = 10
        mu = lam // 2  # number of selected individuals

        # Weights: log-linear, normalized
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        cs = (mueff + 1) / (n + mueff + 3)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = (2 - mueff / (n + 1) + 2 * mueff / (n + 1)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # Initial state
        x_mean = rng.uniform(-4, 4, n)
        sigma = 2.0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        best_x = x_mean.copy()
        best_f = func(best_x)

        calls_made = 1

        for gen in range(1, budget + 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            x_pop = np.zeros((lam, n))
            for i in range(lam):
                xi = x_mean + sigma * (B @ (D * rng.standard_normal(n)))
                xi = np.clip(xi, -5, 5)
                x_pop[i] = xi
                calls_made += 1
                if time.time() > deadline:
                    break

            # Evaluate
            f_pop = np.zeros(lam)
            for i in range(lam):
                if i >= calls_made - 1:
                    break
                f_pop[i] = func(x_pop[i])
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = x_pop[i].copy()

            # Sort by fitness
            order = np.argsort(f_pop)
            x_selected = x_pop[order[:mu]]
            f_selected = f_pop[order[:mu]]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * x_selected, axis=0)
            x_mean = np.clip(x_mean, -5, 5)

            # Update evolution paths
            inv_sigma = 1.0 / sigma if sigma > 1e-12 else 1.0
            d = (x_mean - x_old) * inv_sigma
            pc = (1 - cs) * pc + cs * d
            ncs = np.sqrt((1 - (1 - cs) ** (2 * gen)) / (1 - (1 - cs) ** 2))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs)) * ncs * d
            hsig = (np.linalg.norm(ps) / ncs / np.sqrt(1 - (1 - cs) ** (2 * gen))) < (1.4 + 2 / (n + 1)) * (1 - cs)

            # Update covariance
            if hsig:
                pc = (1 - cs) * pc + cs * d
            else:
                pc = (1 - cs) * pc

            delta_hs = (1 - hsig) * cs * (2 - cs)
            C = ((1 - c1 - cmu + delta_hs) * C
                 + c1 * (np.outer(pc, pc) - (1 - (1 - cs) ** 2) * C)
                 + cmu * np.sum(weights[:, None, None] * np.outer(x_selected - x_old, x_selected - x_old) / sigma ** 2, axis=0))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
