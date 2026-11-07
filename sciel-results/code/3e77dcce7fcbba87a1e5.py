"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = max(4, 4 + int(3 * np.log(n)))
        # Step size initialization
        sigma = 2.0
        # Mean (starting point)
        m = rng.uniform(-5, 5, n)
        f_m = func(m)
        budget -= 1

        # Covariance matrix (start as identity)
        C = np.eye(n)
        # Eigen-decomposition cache for C
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive definite
        eigvals = np.maximum(eigvals, 1e-10)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Path accumulators for step-size adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Parameters
        c_sigma = (lambda_ + 2) / (n + lambda_ + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
        c_1 = 2 / ((n + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 - 1 + 1.0 / n) / ((n + 2) + lambda_ / 2)))

        # Weights for ranking-based mean update
        w = np.log(np.arange(1, lam + 1) - 0.5)
        w = w / np.sum(w)
        mu = int(np.floor(lam / 2))
        w = w[:mu]
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        best_x = m.copy()
        best_f = f_m

        while budget > 0:
            # Generate lambda candidates
            # Use Cholesky-like decomposition via eigen
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Generate candidates
            candidates = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                y = m + sigma * (sqrt_C @ z)
                y = np.clip(y, -5, 5)
                candidates[i] = y

            # Evaluate candidates (with budget check)
            f_values = np.full(lam, np.inf)
            for i in range(lam):
                if budget <= 0:
                    break
                f_values[i] = func(candidates[i])
                budget -= 1
                if f_values[i] < best_f:
                    best_f = f_values[i]
                    best_x = candidates[i].copy()

            # Sort by fitness
            order = np.argsort(f_values)
            y_sorted = candidates[order]
            f_sorted = f_values[order]

            # Only use the valid ones for adaptation
            valid = np.isfinite(f_sorted)
            if not np.any(valid):
                break

            # Use top mu valid candidates
            valid_idx = np.where(valid)[0]
            if len(valid_idx) < mu:
                mu_eff_actual = len(valid_idx)
                use_idx = valid_idx[:mu_eff_actual]
            else:
                use_idx = valid_idx[:mu]

            # Update mean
            m_new = m + sigma * np.sum(w[:len(use_idx)] * (y_sorted[use_idx] - m), axis=0)
            m_new = np.clip(m_new, -5, 5)

            # Update paths
            # p_sigma update
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (m_new - m) / sigma)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (budget + 1)) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # p_c update
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (m_new - m) / sigma)

            # Update covariance
            hsig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 2)) / chi_n - 1
            delta_hsig = (1.0 if hsig > 1.0 / (1 - c_c) else 0.0)
            C = (1 + c_1 * (mu_eff + 2 - 1) * hsig - c_1) * C + c_1 * (
                np.outer(p_c, p_c) + delta_hsig * C
            )
            # Rank-mu update
            diff = y_sorted[use_idx] - m
            C = C + c_mu * np.sum([w[i] * np.outer(diff[i], diff[i]) for i in range(len(use_idx))], axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2.0
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            m = m_new
            # Evaluate new mean if budget allows
            if budget > 0:
                f_m = func(m)
                budget -= 1
                if f_m < best_f:
                    best_f = f_m
                    best_x = m.copy()

            # Termination check: if sigma is very small, stop
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
