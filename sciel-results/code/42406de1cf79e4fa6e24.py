"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES optimizer for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with diverse random samples
        n_global = min(80, budget // 4)
        best_x = None
        best_f = np.inf
        evaluated = 0

        # Sample diverse points across the domain
        for i in range(n_global):
            if time.time() > deadline:
                break
            # Mix of uniform and structured sampling
            if i < dim:
                # Coordinate-wise exploration
                x = rng.uniform(-5, 5, dim)
            elif i < 2 * dim:
                # Sparse perturbations around origin
                x = np.zeros(dim)
                idx = i - dim
                x[idx] = rng.uniform(-5, 5)
            else:
                x = rng.uniform(-5, 5, dim)

            x = np.clip(x, -5, 5)
            fx = func(x)
            evaluated += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES local refinement from best point
        # Use remaining budget for CMA-ES
        remaining = budget - evaluated
        if remaining > 20 and best_x is not None and time.time() < deadline:
            best_x, best_f = self._cmaes(func, best_x, best_f, remaining, rng, deadline)

        # Phase 3: If we have budget left, try a few more restarts
        remaining2 = budget - evaluated - (budget - evaluated - remaining)
        # Actually let's just check remaining budget
        used = 0
        # Recalculate: we used n_global + (budget - evaluated) in CMA-ES
        # Let's be more careful
        # We'll do a final polish if time permits
        if best_x is not None and time.time() < deadline:
            # Quick local search with smaller steps
            sigma = 0.01
            for _ in range(min(50, budget - evaluated)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                evaluated += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.7

        return best_x, best_f

    def _cmaes(self, func, x0, f0, budget, rng, deadline):
        """CMA-ES implementation."""
        dim = self.dim
        x = x0.copy()
        f = f0
        sigma = 1.0  # Initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (1 + 2 / (dim + 1)) / (dim + 4)
        ds = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        cc = (4 + dim / (dim + 4)) / ((dim + 2) + 4 + dim / (dim + 4))
        cmu = min(0.04 + 2 / ((dim + 1.3) ** 2 - 2), 0.3 - cc)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Population size
        lambda_ = max(4, int(4 + np.floor(3 * np.log(dim))))
        lambda_ = min(lambda_, budget - 1)

        weights = np.log(np.arange(1, lambda_ + 1))
        weights = np.exp(weights - np.max(weights))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)

        # Generate initial population
        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            offspring_f = []
            for j in range(lambda_):
                z = rng.standard_normal(dim)
                # Use Cholesky of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                y = sigma * (L @ z)
                x_off = np.clip(x + y, -5, 5)
                f_off = func(x_off)
                offspring.append(x_off)
                offspring_f.append(f_off)

            # Sort by fitness
            sorted_idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in sorted_idx]
            offspring_f = [offspring_f[i] for i in sorted_idx]

            # Update mean
            x_new = np.zeros(dim)
            for j in range(mueff):
                x_new += weights[j] * offspring[j]
            x = x_new

            # Update covariance
            v = np.zeros((dim, dim))
            for j in range(mueff):
                v += np.outer(offspring[j] - x_new, offspring[j] - x_new)
            C = (1 - cmu) * C + cmu / sigma ** 2 * v

            # Update step size
            ps = (1 - cs) * ps + cs / chi * (L.T @ (x_new - x) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (i + 1)) < (1.4 + 2 / (dim + 1)) * chi)
            pc = (1 - cc) * pc + cc * hsig * (x_new - x) / sigma
            C = (1 - cmu * (2 - cmu) + (cc * (2 - cc) - cmu) * (1 - hsig)) * C + cmu * (2 - cmu + cc * (2 - cc)) * np.outer(pc, pc)

            # Update best
            if offspring_f[0] < f:
                f = offspring_f[0]

            # Ensure C is PSD
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)

        return x, f
# EVOLVE-BLOCK-END
