"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive step size for BBOB-style f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        sigma = 2.0
        mu = max(1, n // 2)  # number of selected individuals
        lambda_ = 4 * n  # population size
        # Weights
        weights = np.zeros(lambda_)
        log_w = np.log(np.arange(1, lambda_ + 1))
        weights = log_w / np.sum(log_w)
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = 1.0
        C = np.eye(n)

        calls = 1  # already called func once

        for gen in range(budget - 1):
            if calls >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])
            selected = offspring[:min(mu, len(offspring))]

            # Update best
            if selected[0][1] < best_f:
                best_f = selected[0][1]
                best_x = selected[0][0].copy()

            # Compute mean of selected
            w_sum = np.sum(weights[:len(selected)])
            x_new = np.zeros(n)
            for i, (yi, _) in enumerate(selected):
                x_new += weights[i] * yi
            x_new /= w_sum

            # Update evolution path ps
            inv_sqrt_C = np.linalg.inv(np.sqrtm(C + 1e-10 * np.eye(n)))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ (x_new - x)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * inv_sqrt_C @ (x_new - x)

            # Update covariance matrix
            delta_mu = (1 / c_c - 1) if np.linalg.norm(pc) ** 2 / (1 - (1 - c_c) ** (2 * (gen + 1))) > (1 + 4 / n) * (2 - c_c) else 0
            C = (1 + c_1 + c_mu - c_1) * C
            C += c_1 * (np.outer(pc, pc) - C)
            rank_mu = np.zeros((n, n))
            for i, (yi, _) in enumerate(selected):
                ydiff = inv_sqrt_C @ (yi - x)
                rank_mu += weights[i] * np.outer(ydiff, ydiff)
            C += c_mu * rank_mu
            C -= c_mu * (1 / c_c - 1) * delta_mu * np.outer(pc, pc)

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
