"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with covariance adaptation for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s margin within 300s limit

        d = self.dim
        rng = np.random.default_rng(42)

        # Initialize at center of box
        x = np.zeros(d)
        fx = func(x)

        # CMA-ES parameters
        sigma = 2.0
        # Initial covariance: identity
        C = np.eye(d)
        # Eigen decomposition for sampling
        evals, evecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        evals = np.maximum(evals, 1e-12)

        # Adaptation parameters
        lambda_ = max(4, int(4 + 3 * np.log(d)))  # population size
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

        # State vectors
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        best_x = x.copy()
        best_f = fx

        calls_used = 1

        while calls_used < self.budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Check if sigma is too small
            if sigma < 1e-12:
                break

            # Generate population
            # Update evolution paths and sample
            # First, get Cholesky-like factor from eigen decomposition
            # C = evecs @ diag(evals) @ evecs.T
            # Sample: z ~ N(0, I), y = evecs @ (sqrt(evals) * z)
            sqrt_evals = np.sqrt(evals)

            ys = np.zeros((lambda_, d))
            for i in range(lambda_):
                z = rng.standard_normal(d)
                ys[i] = evecs @ (sqrt_evals * z)

            # Generate candidates
            candidates = np.zeros((lambda_, d))
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                y = np.clip(x + sigma * ys[i], -5, 5)
                fvals[i] = func(y)
                candidates[i] = y
                calls_used += 1
                if calls_used >= self.budget:
                    break
                if time.time() > deadline:
                    break

            if calls_used >= self.budget or time.time() > deadline:
                # Update best
                for i in range(lambda_):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = candidates[i].copy()
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            sorted_candidates = candidates[indices]
            sorted_fvals = fvals[indices]
            sorted_ys = ys[indices]

            # Update best
            if sorted_fvals[0] < best_f:
                best_f = sorted_fvals[0]
                best_x = sorted_candidates[0].copy()

            # Update mean
            x_new = np.zeros(d)
            for i in range(mu):
                x_new += w[i] * sorted_candidates[i]
            x = x_new

            # Update evolution path for covariance
            y_weighted = np.zeros(d)
            for i in range(mu):
                y_weighted += w[i] * sorted_ys[i]

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_weighted / sigma)

            # Update ps and sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_weighted
            sigma *= np.exp((1 / (d_sigma * lambda_)) * (np.linalg.norm(ps) / chi_d - 1))

            # Update covariance matrix
            # Delta h
            hsig = (1 - (1 - c_sigma) ** (2 * (calls_used - 1))) * \
                   (1 + c_sigma / c_c) * np.sqrt(
                       (1 - (1 - c_c) ** (2 * (calls_used - 1))) /
                       (1 - (1 - c_c) ** (2 * (calls_used - 1))))

            # Actually, let's use a simpler rank-mu update
            # C = (1 + c_1 + c_mu - c_1) * C + c_1 * (pc @ pc.T - C) + c_mu * sum(w_i * y_i @ y_i.T)
            # Simplified: C = (1 - c_1 - c_mu) * C + c_1 * pc @ pc.T + c_mu * outer_sum
            outer_sum = np.zeros((d, d))
            for i in range(mu):
                outer_sum += w[i] * np.outer(sorted_ys[i], sorted_ys[i])

            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * outer_sum

            # Ensure symmetry
            C = (C + C.T) / 2

            # Re-diagonalize periodically or every step
            # For efficiency, do it every step since d=5
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-12)

            # Periodically re-center if stuck
            if calls_used % 100 == 0:
                # Small random perturbation
                if sigma < 0.1:
                    x = np.clip(x + 0.5 * rng.standard_normal(d), -5, 5)
                    fx = func(x)
                    calls_used += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    if calls_used >= self.budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
