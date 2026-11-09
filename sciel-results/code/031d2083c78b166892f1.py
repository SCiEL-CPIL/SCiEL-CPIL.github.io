"""Rewritten optimizer for BBOB f11 (Discus) using CMA-ES with adaptive population."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES-like optimizer with adaptive population size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Random restarts to find a good starting point
        n_restarts = 10
        for _ in range(n_restarts):
            if calls_made >= self.budget:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        if calls_made >= self.budget:
            return best_x, best_f

        # Phase 2: CMA-ES optimization from the best point
        x = best_x.copy()
        sigma = 0.5
        n = self.dim
        # Population size: lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(n)) if n > 1 else 4
        mu = lam // 2  # number of parents

        # Covariance matrix
        C = np.eye(n)
        # Evolution path for covariance
        pc = np.zeros(n)
        # Evolution path for step-size
        ps = np.zeros(n)

        # Adaptation parameters
        cc = (4 + n / (n + 4)) / ((n + 2) + 4 + n / (n + 4))
        cs = (n + 1.7) / (n + n / 2 + 5)
        cmu = min(1 - 2 / ((n + 1.3) ** 2 + n), 0.5 * (cc + 2)) / (2 + cc)

        # Remaining budget for CMA-ES
        remaining = self.budget - calls_made

        while calls_made < self.budget:
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lam):
                if calls_made >= self.budget:
                    break
                # Sample from multivariate normal
                z = rng.multivariate_normal(np.zeros(n), C, 1).flatten()
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_x = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_x[0].copy()

            # Update mean
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)

            new_mean = np.sum([w[i] * sorted_x[i] for i in range(mu)], axis=0)

            # Update step-size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * \
                 np.linalg.inv(np.sqrtm(C)) @ (new_mean - x) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made)))) < \
                   (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1))
            sigma *= np.exp(cs / n * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (new_mean - x) / sigma
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + cmu * (2 - cmu + delta_hsig)) * C - cmu * (
                    np.outer(pc, pc) - C)
            # Ensure C is positive semi-definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = new_mean
            x = np.clip(x, -5, 5)

            # Check for convergence
            if sigma < 1e-10 or np.linalg.norm(new_mean - x) < 1e-15:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
