"""New design: CMA-ES style population-based optimizer for f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    Population-based continuous optimizer with a full covariance matrix,
    path-based step-size control, and elitist selection. Designed for the
    multimodal Lunacek bi-Rastrigin function on [-5, 5]^5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0  # safety margin before the 330s wall-clock kill

        d = self.dim
        n = self.budget
        rng = np.random.default_rng(12345)

        # Population size: lambda ~ 4 + 3 ln d  (standard CMA-ES heuristic)
        lam = max(4, int(4 + 3 * np.log(d)))
        # Cap population so we can afford at least a few generations within budget
        if lam > n // 3:
            lam = max(4, n // 3)
        mu = lam // 2  # number of parents

        # Weights for recombination (mu-weighted)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Initial step size and center
        sigma = 2.0
        x_mean = rng.uniform(-5, 5, d)

        # Covariance matrix C and its square root (Cholesky)
        C = np.eye(d)
        C_root = np.eye(d)

        # Path variables for step-size and covariance adaptation
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Adaptation parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Track best
        x_best = x_mean.copy()
        f_best = float('inf')

        calls_made = 0

        def _time_ok():
            return (time.time() - t0) < deadline

        # Initial evaluation
        f_val = func(x_best)
        calls_made += 1
        f_best = f_val

        while calls_made < n and _time_ok():
            # Generate population
            Y = []
            for _ in range(lam):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (C_root @ z)
                Y.append(np.clip(y, -5, 5))
            Y = np.array(Y)  # shape (lam, d)

            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if not _time_ok():
                    break
                F[i] = func(Y[i])
                calls_made += 1
            else:
                # all evaluated
                pass
            # If we broke out early, fill remaining with inf (shouldn't happen often)
            if calls_made < n and not _time_ok():
                break

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < f_best:
                f_best = F_sorted[0]
                x_best = Y_sorted[0].copy()

            # Weighted mean of parents
            y_w = np.sum(w[:, None] * Y_sorted[:mu], axis=0)

            # Old mean for path update
            x_mean_old = x_mean.copy()
            x_mean = y_w

            # Update sigma paths
            C_root_inv = np.linalg.inv(C_root)
            z_mean = C_root_inv @ (y_w - x_mean_old) / sigma

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_mean
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made / n + 1)) <
                       (1.4 + 2 / (d + 1)) * chi_n)
            h_sigma = float(h_sigma)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * z_mean

            # Step size adaptation (CSA)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)

            # Covariance matrix update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                zi = C_root_inv @ (Y_sorted[i] - x_mean_old) / sigma
                rank_mu += w[i] * np.outer(zi, zi)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu

            # Ensure C is symmetric positive definite and update Cholesky
            C = (C + C.T) / 2
            try:
                C_root = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal perturbation and retry
                C += 1e-6 * np.eye(d)
                try:
                    C_root = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(d)
                    C_root = np.eye(d)

            # Keep mean within bounds (soft)
            x_mean = np.clip(x_mean, -5, 5)

        return x_best, f_best
# EVOLVE-BLOCK-END
