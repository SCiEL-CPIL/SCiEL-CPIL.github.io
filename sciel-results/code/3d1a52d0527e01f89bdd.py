"""Population-based CMA-ES for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based covariance adaptation for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * dim)  # population size
        mu = lambda_ // 2  # number of best individuals used for update
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        # Step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Chi-normal
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Track best value history for AOCC (not needed for return, but we track internally)
        # We just need to return something; the evaluator ignores the return value.
        # But we must make exactly budget calls or stop at deadline.

        # Precompute sqrt of C for sampling
        # We'll update C each generation and recompute Cholesky

        # Helper to sample from N(0, C)
        def sample_from_C():
            # Use Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            z = rng.standard_normal(dim)
            return L @ z

        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline - 5:
                break
            if calls_made >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                y = x_mean + sigma * sample_from_C()
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([weights[i] * offspring[i][1] for i in range(mu)], axis=0)

            # Update step size control
            # Compute norm of (x_mean - old_mean) / sigma
            d_mean = (x_mean - old_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_mean
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_made - gen * lambda_ + 1)) if calls_made > gen * lambda_ else 0
            # Simplified: use generation count for exponent
            # Actually, let's track generations properly
            # For simplicity, use a fixed exponent approximation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance
            # pc = (1 - c_c) * pc + c_c * (hs < 1.4 + 2 / (dim + 1)) * d_mean
            hsig = 1 if np.linalg.norm(ps) / chi_n < 1.4 + 2 / (dim + 1) else 0
            pc = (1 - c_c) * pc + c_c * hsig * d_mean

            # Rank-one and rank-mu update
            C = (1 + c_c * hsig * (2 - c_c) - c_1 - c_mu) * C
            C += c_c * (2 - c_c) * np.outer(pc, pc)
            C -= c_1 * (C - np.eye(dim))

            # Rank-mu update
            for i in range(mu):
                y_i = (offspring[i][1] - old_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization for numerical stability
            C += 1e-10 * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
