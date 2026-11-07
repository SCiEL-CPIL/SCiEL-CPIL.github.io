"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of selected parents
        lam = max(2 * mu, dim + 1)  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial point: try a few random starts, pick best
        rng = np.random.default_rng(42)
        starts = 5
        best_x = None
        best_f = np.inf
        for _ in range(starts):
            x0 = rng.uniform(-4, 4, dim)
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        x = best_x.copy()
        f_best = best_f

        # CMA-ES state
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        calls_made = starts

        for gen in range(budget - calls_made):
            if time.time() > deadline:
                break

            # Generate population
            B = rng.standard_normal((lam, dim))
            # Apply C transform via eigenvalue decomposition (small dim)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            Y = B @ sqrt_C.T
            pop = x + sigma * Y
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            pop_f = np.array([func(p) for p in pop])
            calls_made += lam

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_sorted = pop[idx]
            pop_f_sorted = pop_f[idx]

            # Update best
            if pop_f_sorted[0] < f_best:
                f_best = pop_f_sorted[0]
                x_best = pop_sorted[0].copy()

            # Weighted mean
            m = np.average(pop_sorted[:mu], axis=0, weights=weights)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrt_C.T @ (m - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (dim + 1))

            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma

            # Update mean
            x = m

            # Update covariance
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) - np.diag((pc ** 2) * c_c * (2 - c_c))) \
                + c_mu * sum(w * np.outer(pop_sorted[i] - m, pop_sorted[i] - m) for i, w in enumerate(weights))

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

        return x, f_best
# EVOLVE-BLOCK-END
