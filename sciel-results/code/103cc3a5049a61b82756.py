"""CMA-ES based optimizer for Lunacek bi-Rastrigin f24 with adaptive restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Population-based CMA-ES with LHS initialization and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, f, x0, sigma, budget_remaining, deadline, dim):
        """Run a simple CMA-ES implementation with the given initial point and step size."""
        lambda_ = max(4, int(4 + 2 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        # Weights (interpolated)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Step size control parameters
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # State
        mean = np.array(x0, dtype=float)
        C_inv = np.eye(dim)
        C = np.eye(dim)
        sigma = float(sigma)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_used = 0

        # Evaluate initial point
        f_mean = f(mean)
        calls_used += 1

        # Track best
        best_x = mean.copy()
        best_f = f_mean

        # CMA-ES iterations
        max_iter = budget_remaining // (lambda_ + 1)
        for it in range(max_iter):
            if calls_used >= budget_remaining or time.time() > deadline:
                break

            # Generate population
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Sample directions
            z = np.random.randn(lambda_, dim)
            pop = mean + sigma * (z @ L.T)

            # Evaluate population
            fvals = np.array([f(x) for x in pop])
            calls_used += lambda_

            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            y = z[order]  # sorted standard normals
            f_sorted = fvals[order]

            # Update step size
            y_sigma = (y[:mu] @ weights[:, None])
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update mean
            mean = pop[order[:mu]] @ weights[:, None]

            # Update covariance
            y_c = (y[:mu] @ weights[:, None])
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * (pop[order[:mu]] - mean) @ (pop[order[:mu]] - mean).T / mu_eff

            # Regularize C
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(dim)

            # Update inverse
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)

            # Check convergence
            if sigma < 1e-8:
                break

        return best_x, best_f, calls_used

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0
        calls = [0]

        def f(x):
            if calls[0] >= budget or time.time() > deadline:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        rng = np.random.default_rng(42)
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: LHS initialization + evaluate
        n_lhs = 8
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                perm = rng.permutation(n_lhs)
                interval = perm[i]
                point[d] = -5.0 + (interval + rng.random()) * (10.0 / n_lhs)
            lhs_points.append(point)

        # Add strategic points
        strategic = [np.zeros(dim), np.full(dim, 2.5), np.full(dim, -2.5)]
        all_starts = lhs_points + strategic

        for x0 in all_starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            fx0 = f(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        # Phase 2: CMA-ES from the best starting point
        remaining = budget - calls[0]
        if remaining > 20 and time.time() < deadline:
            sigma_init = 2.0
            bx, bf, used = self._cma_es(f, best_x, sigma_init, remaining, deadline, dim)
            calls[0] += used
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: L-BFGS-B polish
        if calls[0] < budget and time.time() < deadline:
            try:
                res = minimize(f, best_x, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 30, 'maxfun': 30})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: If budget remains, second CMA-ES from a different region
        remaining = budget - calls[0]
        if remaining > 30 and time.time() < deadline:
            # Pick a diverse start far from current best
            candidates = [p for p in all_starts if np.linalg.norm(p - best_x) > 2.0]
            if candidates:
                x_start = candidates[0]
                bx, bf, used = self._cma_es(f, x_start, 1.5, remaining, deadline, dim)
                calls[0] += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
