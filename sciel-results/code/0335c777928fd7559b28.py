"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = int(4 + 3 * np.log(n))
        # Weighting
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Initial population: spread out to explore the box
        x = np.zeros(n)
        # Try a few random starting points and pick the best
        best_x = None
        best_fx = np.inf
        for _ in range(min(10, budget // 10)):
            x_try = rng.uniform(-4, 4, n)
            fx_try = func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()
            if time.time() > deadline:
                break

        x = best_x.copy()
        fx = best_fx
        sigma = 1.0

        # CMA-ES state
        C = np.eye(n)
        C_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Track best
        global_best_x = x.copy()
        global_best_fx = fx
        calls_made = 10  # approximate; we made up to 10 calls in initialization

        for gen in range(budget // lam + 1):
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(n)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < global_best_fx:
                    global_best_fx = fy
                    global_best_x = y.copy()
                population.append((y, fy))

            if len(population) < mu:
                break

            # Sort by fitness
            population.sort(key=lambda t: t[1])

            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * np.array([p[0] for p in population[:mu]]), axis=0)

            # Update covariance matrix adaptation
            y_weighted = (x - x_old) / sigma
            pc = (1 - c_c) * pc + c_c * y_weighted
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x - x_old)

            delta_sigma = np.linalg.norm(ps) / chi_n - 1
            sigma *= np.exp(c_sigma / d_sigma * delta_sigma)
            sigma = np.clip(sigma, 1e-10, 10.0)

            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < \
                      (1.4 + 2 / (n + 1)) * chi_n

            C = (1 + c_1 * (2 - h_sigma) - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) - h_sigma * (1 - c_c) * (1 - c_c) * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(np.array([p[0] for p in population[:mu]] - x_old) / sigma,
                                                                  np.array([p[0] for p in population[:mu]] - x_old) / sigma), axis=0)

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Cholesky decomposition with fallback
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            # Check for convergence
            if np.all(np.abs(np.diag(C)) < 1e-12):
                break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
