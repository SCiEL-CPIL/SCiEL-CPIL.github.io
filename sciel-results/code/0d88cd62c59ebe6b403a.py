"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with boundary clipping, budget tracking, and wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 320.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initialize: start at 0 for symmetry, sigma = 2.0 (covers most of [-5,5])
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        d = np.ones(dim)

        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        sigma0 = 2.0
        x0 = np.zeros(dim)

        best_x = x.copy()
        best_f = func(x)
        budget_remaining = budget - 1  # one call already made

        for gen in range(10 ** 6):
            if budget_remaining <= 0 or time.time() >= deadline:
                break

            # Evaluate population
            pop = x + sigma * (rng.standard_normal((lam, dim)) @ d[:, None])
            pop = np.clip(pop, -5, 5)
            fits = np.array([func(p) for p in pop])
            budget_remaining -= lam

            # Update best
            idx = np.argmin(fits)
            if fits[idx] < best_f:
                best_f = fits[idx]
                best_x = pop[idx].copy()

            # Sort
            sorted_idx = np.argsort(fits)
            y_sorted = pop[sorted_idx]
            f_sorted = fits[sorted_idx]

            # Mean update
            m_new = x + sigma * np.sum(weights[:, None] * (y_sorted[:mu] - x) / sigma, axis=0)
            # More numerically stable
            m_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Covariance matrix update
            yw = (m_new - x) / sigma
            pc = (1 - cs) * pc + cs * yw if 'pc' in dir() else cs * yw
            pc = cs * yw + (1 - cs) * pc

            c1_update = c1 * (np.outer(yw, yw) - C)
            cmu_update = cmu * np.sum(weights[:, None] * np.outer((y_sorted[:mu] - x) / sigma, (y_sorted[:mu] - x) / sigma), axis=0)
            C = (1 + c1_update + cmu_update) * C

            # Symmetrize
            C = (C + C.T) / 2

            # Eigen decomposition for normalization
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                d = eigvecs / np.sqrt(eigvals)
                C = np.eye(dim)  # C is implicitly identity after transform, we use d
            except np.linalg.LinAlgError:
                d = np.eye(dim)

            # Sigma update
            sigma = sigma * np.exp(cs / chi * (np.linalg.norm(pc) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
