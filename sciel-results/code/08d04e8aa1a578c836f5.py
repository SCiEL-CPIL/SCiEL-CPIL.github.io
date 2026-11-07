"""Two-phase optimizer: LHS global sweep + CMA-ES local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Latin Hypercube Sampling sweep followed by CMA-ES local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        # Phase 1: Latin Hypercube Sampling for global coverage
        n_lhs = min(50, budget // 5)
        remaining = budget - n_lhs

        # Generate LHS points
        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, n))
        for d in range(n):
            perm = rng.permutation(n_lhs)
            # Spread points within each stratum
            lhs_points[:, d] = (perm + rng.random(n_lhs)) / n_lhs * 10.0 - 5.0

        # Evaluate LHS points
        best_x = None
        best_f = np.inf
        for i in range(n_lhs):
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES starting from best LHS point
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))
        # Ensure pop_size is reasonable
        pop_size = min(pop_size, remaining // 2) if remaining >= 8 else 1

        # CMA-ES state
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(n)
        Cs = np.zeros((n, n))

        # Adaptation rates
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

        # Memory
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Cholesky factor
        try:
            chol_C = np.linalg.cholesky(C)
        except:
            chol_C = np.eye(n)

        # Track best
        cma_best_x = best_x.copy()
        cma_best_f = best_f

        # CMA-ES loop
        for gen in range(remaining // pop_size + 1):
            if time.time() > deadline - 5:
                break
            if remaining < pop_size:
                # Use remaining budget for a few more evaluations
                while remaining > 0 and time.time() < deadline - 2:
                    y = mean + sigma * rng.multivariate_normal(np.zeros(n), C)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    remaining -= 1
                    if fy < cma_best_f:
                        cma_best_f = fy
                        cma_best_x = y.copy()
                break

            # Generate population
            ys = np.zeros((pop_size, n))
            fs = np.zeros(pop_size)
            for i in range(pop_size):
                z = rng.standard_normal(n)
                ys[i] = mean + sigma * (chol_C @ z)
                ys[i] = np.clip(ys[i], -5, 5)
                fs[i] = func(ys[i])
                remaining -= 1

            if time.time() > deadline - 5:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Update best
            if fs_sorted[0] < cma_best_f:
                cma_best_f = fs_sorted[0]
                cma_best_x = ys_sorted[0].copy()

            # Update mean
            new_mean = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Update step-size control vector
            inv_chol_C = np.linalg.inv(chol_C)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_chol_C @ (new_mean - mean) / sigma)

            # Update covariance adaptation matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

            # Update covariance matrix
            delta_hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) > (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff))
            c_1_eff = c_1 + (2 * (c_1 - c_1) if delta_hs else 0)  # simplified
            rank_1 = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                yi = inv_chol_C @ (ys_sorted[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(yi, yi)

            C = (1 + c_1_eff + c_mu - c_1) * C + c_1 * rank_1 + c_mu * rank_mu

            # Ensure C is positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                chol_C = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except:
                C = np.eye(n)
                chol_C = np.eye(n)

            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update mean
            mean = new_mean

            # Check convergence
            if sigma < 1e-8:
                break

        return cma_best_x, cma_best_f
# EVOLVE-BLOCK-END
