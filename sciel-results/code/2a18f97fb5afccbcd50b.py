"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: CMA-ES with restarts for sphere function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        # Phase 1: Initial random sampling to get a good starting point
        n_init = min(20, budget)
        init_points = rng.uniform(-5, 5, (n_init, dim))
        init_fvals = np.array([func(init_points[i]) for i in range(n_init)])
        calls_made += n_init
        init_best_idx = np.argmin(init_fvals)
        best_x = init_points[init_best_idx].copy()
        best_f = init_fvals[init_best_idx]

        # Phase 2: CMA-ES with restarts
        # We'll do multiple CMA-ES runs with different starting points
        # Each run uses a fraction of the remaining budget

        remaining_budget = budget - calls_made

        # Try CMA-ES from the best initial point
        if remaining_budget > 30 and not check_deadline():
            x, fx = self._cmaes(func, best_x, best_f, remaining_budget, dim, rng, check_deadline)
            if fx < best_f:
                best_f = fx
                best_x = x

        # If we have budget left, try another CMA-ES from a different random point
        if remaining_budget > 60 and not check_deadline():
            x2 = rng.uniform(-5, 5, dim)
            fx2 = func(x2)
            calls_made += 1
            if fx2 < best_f:
                best_f = fx2
                best_x = x2.copy()

            remaining2 = budget - calls_made
            if remaining2 > 30 and not check_deadline():
                x, fx = self._cmaes(func, best_x, best_f, remaining2, dim, rng, check_deadline)
                if fx < best_f:
                    best_f = fx
                    best_x = x

        return best_x, best_f

    def _cmaes(self, func, x0, f0, budget, dim, rng, check_deadline):
        """Simple CMA-ES implementation."""
        x = x0.copy()
        f = f0
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        cs = np.zeros(dim)
        mu = max(1, dim // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs_ = (2 + mueff / dim) / (dim + 2 + mueff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs_
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        no_improve = 0
        max_no_improve = 5 * dim

        for _ in range(budget):
            if check_deadline():
                break

            # Generate lambda candidates
            lam = max(4, 4 + int(3 * dim))
            ys = rng.standard_normal((lam, dim))
            ys = ys / np.linalg.norm(ys, axis=1, keepdims=True)
            # Sort by norm to get diverse step sizes
            norms = np.linalg.norm(ys, axis=1)
            ys = ys * (sigma * norms)

            # Actually, for CMA-ES we should use the covariance matrix
            # Let me simplify: use a basic version with diagonal covariance
            # For sphere function, diagonal CMA-ES is sufficient

            # Sample from N(x, sigma^2 * C)
            L = np.linalg.cholesky(C)
            ys = rng.standard_normal((lam, dim))
            ys = ys @ L.T * sigma

            # Evaluate candidates
            best_y = None
            best_fy = np.inf
            for i in range(lam):
                y = np.clip(x + ys[i], -5, 5)
                fy = func(y)
                if fy < best_fy:
                    best_fy = fy
                    best_y = y.copy()

            # Update
            if best_fy < f:
                x = best_y
                f = best_fy
                no_improve = 0
                sigma *= 1.2  # increase step size on success
            else:
                no_improve += 1
                sigma *= 0.8  # decrease step size on failure

            # Simple covariance update based on direction
            if best_fy < f0:  # made progress
                direction = (x - x0) / (np.linalg.norm(x - x0) + 1e-10)
                C = 0.9 * C + 0.1 * np.outer(direction, direction)
                C = (C + C.T) / 2  # symmetrize

            # Adapt sigma based on success rate (simplified)
            if no_improve > max_no_improve:
                break

            # Check if sigma is too small
            if sigma < 1e-8:
                break

        return x, f
# EVOLVE-BLOCK-END
