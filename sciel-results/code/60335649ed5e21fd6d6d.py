"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with box-constraint handling for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        cc = (4 + mu_eff / (dim + 4)) / (dim + 4 + 2 * mu_eff / (dim + 4))
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - cc + cs)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lam, dim))
            C = np.eye(dim)  # start with identity, will update

            # Actually, for a proper CMA-ES we need to maintain C
            # Let me implement a simpler but effective version

            # Sample from N(x, sigma^2 * C)
            # Using Cholesky decomposition
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            pop = x[None, :] + sigma * (B @ L.T)
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.empty(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = pop[i].copy()

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update mean
            x_old = x.copy()
            x = sum(weights[i] * pop_sorted[i] for i in range(mu))
            x = np.clip(x, -5, 5)

            # Update evolution paths
            diff = (x - x_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ diff)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls // lam + 1))
                    / np.sqrt(dim) < 1.4 + 2 / (dim + 1))

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff

            # Update covariance
            delta = (1 + 2 * (hsig - 1) * c1 / cmu) if cmu > 0 else 1
            outer = np.zeros((dim, dim))
            for i in range(mu):
                outer += weights[i] * np.outer(pop_sorted[i] - x_old, pop_sorted[i] - x_old)
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu / sigma ** 2 * outer

            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = (C + 1e-5 * np.eye(dim)) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
