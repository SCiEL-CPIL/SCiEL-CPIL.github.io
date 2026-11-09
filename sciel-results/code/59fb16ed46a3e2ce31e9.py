"""Two-phase optimizer: coarse random search then CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search then CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        # Counters
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(42)

        def now():
            return time.time()

        # ---- Phase 1: Coarse random search to find a good basin ----
        # Use about 20% of budget for exploration
        explore_budget = max(50, int(0.2 * budget))
        best_x = None
        best_f = np.inf

        # Generate diverse random points
        n_explore = explore_budget
        for i in range(n_explore):
            if calls[0] >= budget or now() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also try origin and some structured points
        structured_starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
        ]
        for s in structured_starts:
            if calls[0] >= budget or now() > deadline:
                break
            x = np.clip(s, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        # ---- Phase 2: CMA-ES refinement from best point ----
        remaining = budget - calls[0]
        if remaining > 10:
            best_x, best_f = self._run_cma_es(
                func, best_x, best_f, remaining, deadline
            )

        return best_x, best_f

    def _run_cma_es(self, func, x0, f0, budget, deadline):
        """Run a simplified CMA-ES with population-based exploration."""
        dim = self.dim
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(123)

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~4-9 for dim=5
        sigma = 1.0
        x = np.clip(x0, -5, 5).copy()
        f_best = f0
        x_best = x.copy()

        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Step sizes for covariance update
        c_sigma = (pop_size + 2) / ((dim + 1.3) ** 2 + pop_size)
        c_c = (4 + dim / 4) / ((dim + 1.3) ** 3 + pop_size + 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
        # Eigenvalue decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        # Prevent negative eigenvalues
        eigvals = np.maximum(eigvals, 1e-12)
        D = np.sqrt(eigvals)
        inv_D = 1.0 / D

        # Memory vectors
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        used = 0
        stagnation_counter = 0
        best_f_this_run = f_best

        while used < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []
            for i in range(pop_size):
                # Sample from multivariate normal using current C and sigma
                z = rng.standard_normal(dim)
                y = x + sigma * (eigvecs @ (inv_D * (eigvecs.T @ z)))
                y = np.clip(y, -5, 5)
                try:
                    fy = safe_func(y)
                except StopIteration:
                    break
                used += 1
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) == 0:
                break

            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]

            # Update best
            if offspring_f[0] < f_best:
                f_best = offspring_f[0]
                x_best = offspring[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Compute weighted mean (top half)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            weights /= weights.sum()
            x_new = sum(weights[i] * offspring[i] for i in range(mu))

            # Update sigma (CSA - Cumulative Step Adaptation)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                (eigvecs.T @ (D * (eigvecs.T @ (x_new - x)))) / sigma
            )
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-9, 10.0)

            # Update covariance matrix
            v = (x_new - x) / sigma
            h_sigma = (
                np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (used + 1)))
                < (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
            )
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Rank-mu update
            C = (1 + c_c * (2 - c_c) - c_c) * C + c_c * np.outer(v, v)
            for i in range(mu):
                vi = (offspring[i] - x) / sigma
                C += c_c * weights[i] * np.outer(vi, vi)

            # Update eigendecomposition periodically (expensive)
            if used % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                inv_D = 1.0 / D

            x = np.clip(x_new, -5, 5)

            # Early stop if stagnating for too long
            if stagnation_counter > 20 and sigma < 1e-6:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
