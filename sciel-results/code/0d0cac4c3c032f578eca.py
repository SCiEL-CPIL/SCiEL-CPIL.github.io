"""Rewritten optimizer: CMA-ES with initial random sampling and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with random initialization for BBOB f3 (Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        rng = np.random.default_rng(12345)
        dim = self.dim

        # Phase 1: Random sampling to find a good starting point
        n_init = 50
        best_x = None
        best_f = np.inf
        candidates = []

        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort candidates and pick the best as starting point for CMA-ES
        candidates.sort(key=lambda t: t[0])
        x_mean = candidates[0][1].copy()
        f_mean = candidates[0][0]

        # Phase 2: CMA-ES
        # Population size
        n = max(4 + int(3 * np.log(dim)), 8)  # population size
        # Initial sigma based on spread of initial samples
        # Use a moderate initial sigma
        sigma = 1.0

        # CMA-ES parameters
        cc = (4 + n / dim) / (dim + 4 + 2 * n / dim)
        cs = (n + 2 + 2 / (n + 1)) / (dim + n + 5)
        c1 = (2 - mu_eff / dim + 2 / dim) / ((dim + 1.3) ** 2 + mu_eff) if False else (2 - mu_eff / dim + 2 / dim) / ((dim + 1.3) ** 2 + mu_eff)
        cmu = (2 * (mu_eff / 2 - 1) / dim + 2 / dim) / ((dim + 1.3) ** 2 + mu_eff) if False else (2 * (mu_eff / 2 - 1) / dim + 2 / dim) / ((dim + 1.3) ** 2 + mu_eff)

        # Actually let me compute mu_eff properly
        # Top half of population is used for recombination
        mu = n // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Recompute c1 and cmu with proper mu_eff
        c1 = (2 - mu_eff / dim + 2 / dim) / ((dim + 1.3) ** 2 + mu_eff)
        cmu = (2 * (mu_eff / 2 - 1) / dim + 2 / dim) / ((dim + 1.3) ** 2 + mu_eff)

        # Covariance matrix
        C = np.eye(dim)
        # Evolution path for covariance
        pc = np.zeros(dim)
        # Evolution path for step size
        ps = np.zeros(dim)

        # Cholesky factor of C (initially identity)
        # We'll use eigendecomposition for efficiency in 5D
        # Actually for dim=5, we can just use Cholesky

        calls_used = n_init
        budget_remaining = self.budget - calls_used

        for gen in range(budget_remaining // n + 1):
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break

            # Generate population
            offspring = []
            for i in range(n):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break

                # Sample from N(x_mean, sigma^2 * C)
                # Use eigendecomposition for sampling
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                z = rng.standard_normal(dim)
                # Transform: eigvecs @ diag(sigma * sqrt(eigvals)) @ z
                x_child = x_mean + sigma * (eigvecs * np.sqrt(eigvals)) @ z
                x_child = np.clip(x_child, -5, 5)

                f_child = func(x_child)
                calls_used += 1
                offspring.append((f_child, x_child.copy()))

            if len(offspring) == 0:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Recombination: weighted average of top mu solutions
            x_new = np.zeros(dim)
            for i in range(min(mu, len(offspring))):
                x_new += weights[i] * offspring[i][1]

            # Update evolution paths
            # Normalized step
            if sigma > 1e-12:
                step = (x_new - x_mean) / sigma
            else:
                step = np.zeros(dim)

            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * step

            # Update pc
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** (gen + 1)):
                pc = (1 - cc) * pc + cc * step

            # Update covariance matrix
            # Rank-one update
            C = (1 + c1 * cc * (np.outer(pc, pc) - C) + c1 * (2 - cc)) * C
            # Rank-mu update
            for i in range(min(mu, len(offspring))):
                diff = (offspring[i][1] - x_mean) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update x_mean
            x_mean = x_new

            # Update sigma
            sigma *= np.exp(cs / (dim + 1) * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
