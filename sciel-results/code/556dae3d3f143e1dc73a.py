"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and covariance matrix."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        rng = np.random.default_rng(42)

        # Initialize
        x = np.zeros(d)  # Start at origin, which is often good for rotated functions
        fx = func(x)

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(d)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(np.sum(weights[:mu]) ** 2 / np.sum(weights ** 2))
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((d + 2) ** 2 + mueff))

        sigma = 2.0
        ps = np.zeros(d)
        pc = np.zeros(d)
        C = np.eye(d)
        Csqrt = np.eye(d)
        invsqrt_C = np.eye(d)

        # Precompute Cholesky factor of C
        def update_chol(C):
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Symmetrize and retry
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                    return L
                except np.linalg.LinAlgError:
                    # Eigenvalue decomposition fallback
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                    return L

        Csqrt = update_chol(C)

        # Track best
        best_x = x.copy()
        best_fx = fx

        # Population initialization
        pop = []
        for i in range(lambda_):
            y = np.clip(x + sigma * Csqrt @ rng.standard_normal(d), -5, 5)
            pop.append(y)
        fvals = [func(y) for y in pop]

        calls = 1 + lambda_

        for gen in range(1, self.budget - calls + 1):
            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = [pop[i] for i in order]
            f_sorted = [fvals[i] for i in order]

            # Update best
            if f_sorted[0] < best_fx:
                best_fx = f_sorted[0]
                best_x = y_sorted[0].copy()

            # Mean update
            m_new = np.sum(np.array([y_sorted[i] for i in range(mu)]) * weights[:, None], axis=0)

            # Step size control (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Csqrt.T @ (m_new - x))
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (d + 1)) * np.trace(Csqrt) / d else 0.0
            sigma *= np.exp(cs / d * (np.linalg.norm(ps) / np.sqrt(d) - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Covariance matrix update
            pc = (1 - cc) * pc + hsig * cc * (m_new - x) / sigma
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((d, d))
            for i in range(mu):
                zi = (y_sorted[i] - x) / sigma
                rankmu += weights[i] * np.outer(zi, zi)
            C = (1 + c1 * hsig * (cc - cc - 1) + c1 + cmu) * C + c1 * (rank1 - (1 - hsig * cc * (cc - cc - 1)) * C) + cmu * rankmu
            C = (C + C.T) / 2  # Symmetrize

            # Update square root
            Csqrt = update_chol(C)

            # Move mean
            x = m_new

            # Generate next population
            pop = []
            fvals = []
            for i in range(lambda_):
                y = np.clip(x + sigma * Csqrt @ rng.standard_normal(d), -5, 5)
                pop.append(y)
                fy = func(y)
                fvals.append(fy)
                calls += 1

            if calls >= self.budget:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
