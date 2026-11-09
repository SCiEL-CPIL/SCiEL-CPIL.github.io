"""CMA-ES based optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # CMA-ES parameters - use larger population for better exploration
        lam = 6
        mu = 3

        # Standard CMA-ES weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(i + 1) - np.log(mu + 0.5)
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)

        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation - more aggressive for faster convergence
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation - faster adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial state - start with wider spread to explore the box
        x_mean = np.zeros(n)
        sigma = 2.0  # Larger initial sigma to explore [-5, 5]^5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Precompute chi_n
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        # Evaluate initial point
        best_x = x_mean.copy()
        best_f = func(x_mean)
        call_count = 1

        # Track progress for restart detection
        last_best_f = best_f
        no_improvement_count = 0
        restart_threshold = 50  # Restart if no improvement for 50 calls

        # Main loop
        while call_count < budget:
            # Check for stagnation and restart if needed
            if best_f < last_best_f - 1e-6:
                last_best_f = best_f
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count > restart_threshold and call_count < budget - lam:
                # Restart from best point with smaller sigma
                x_mean = best_x.copy()
                sigma = max(0.5, sigma * 0.5)
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                no_improvement_count = 0
                continue

            # Compute C^(1/2) via eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-15)
            sqrt_eigvals = np.sqrt(eigvals)
            C_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

            # Sample lambda points
            pop = np.zeros((lam, n))
            for j in range(lam):
                z = np.random.standard_normal(n)
                pop[j] = x_mean + sigma * C_sqrt @ z

            # Evaluate population
            fvals = np.zeros(lam)
            for j in range(lam):
                x_j = np.clip(pop[j], -5, 5)
                fvals[j] = func(x_j)
                call_count += 1

                if fvals[j] < best_f:
                    best_f = fvals[j]
                    best_x = x_j.copy()

                if call_count >= budget:
                    break

            if call_count >= budget:
                break

            # Sort and select
            sorted_idx = np.argsort(fvals)
            idx = sorted_idx[:mu]

            # New mean
            x_mean_new = np.zeros(n)
            for i in range(mu):
                x_mean_new += weights[i] * pop[idx[i]]

            diff = x_mean_new - x_mean

            # C^(-1/2)
            inv_sqrt_eigvals = 1.0 / sqrt_eigvals
            C_inv_sqrt = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ diff)

            ps_norm = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma)**2) < \
                         (1.4 + 2/(n+1)) * np.sqrt(mu_eff + 1) / np.sqrt(mu_eff - 1) else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma

            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (p_c @ p_c.T + (1 - h_sigma * (2 - c_c)) * np.eye(n))

            # Rank-mu update
            for i in range(mu):
                y_i = C_inv_sqrt @ (pop[idx[i]] - x_mean) / sigma
                C += c_mu * weights[i] * (y_i @ y_i.T)

            x_mean = x_mean_new

            # Symmetrize
            C = (C + C.T) / 2

            # Numerical stability check
            eigvals_check = np.linalg.eigvalsh(C)
            if np.any(eigvals_check < 1e-12) or np.any(eigvals_check > 1e12):
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = max(0.5, sigma * 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
