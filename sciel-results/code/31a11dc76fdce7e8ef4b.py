"""Rewritten optimizer for BBOB-style f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            return x, f

        def cma_es(x0, sigma0, budget_cma):
            nonlocal calls_made, global_best_x, global_best_f
            dim = self.dim
            # CMA-ES parameters
            lambda_ = int(4 + 3 * np.log(dim))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            # Precompute sqrt(C) via Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

            # Evaluate initial point
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            calls_made += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x_clipped.copy()

            best_x = x_clipped.copy()
            best_f = fx

            for it in range(budget_cma):
                if time.time() > deadline - 5:
                    break
                if calls_made >= self.budget:
                    break

                # Generate offspring
                ys = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    try:
                        z = rng.standard_normal(dim)
                        y = x + sigma * (C_sqrt @ z)
                        ys[i] = y
                    except Exception:
                        ys[i] = x + sigma * rng.standard_normal(dim)

                # Evaluate offspring
                fvals = np.zeros(lambda_)
                for i in range(lambda_):
                    if time.time() > deadline - 5:
                        break
                    if calls_made >= self.budget:
                        fvals[i] = np.inf
                        continue
                    xi = np.clip(ys[i], -5, 5)
                    fi = float(func(xi))
                    fvals[i] = fi
                    calls_made += 1
                    if fi < global_best_f:
                        global_best_f = fi
                        global_best_x = xi.copy()

                # Sort by fitness
                order = np.argsort(fvals)
                y_sel = ys[order[:mu]]
                f_sel = fvals[order[:mu]]

                # Update best
                if f_sel[0] < best_f:
                    best_f = f_sel[0]
                    best_x = y_sel[0].copy()

                # Weighted mean of selected
                x_mean = np.sum(weights[:, None] * y_sel, axis=0)

                # Update evolution paths
                try:
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)

                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_mean - x)) / sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) / \
                          (np.sqrt(mu_eff) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))))
                # Approximate h_sigma threshold
                h_sigma_thresh = 1.4 + 2 / (dim + 1)
                if np.linalg.norm(p_sigma) / sigma < h_sigma_thresh * np.sqrt(mu_eff):
                    p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x) / sigma
                    delta_h = 1.0
                else:
                    delta_h = 0.0

                # Update covariance matrix
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - delta_h) * c_c * (2 - c_c) * C)
                for i in range(mu):
                    ydiff = (y_sel[i] - x_mean) / sigma
                    C += c_mu * weights[i] * np.outer(ydiff, ydiff)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 1e3)

                # Update mean
                x = x_mean

                # Recompute Cholesky
                try:
                    # Add small regularization for numerical stability
                    C_reg = C + 1e-10 * np.eye(dim)
                    C_sqrt = np.linalg.cholesky(C_reg)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)

                # Check convergence
                if np.std(f_sel) < 1e-12 or sigma < 1e-10:
                    break

            return best_x, best_f

        # Multi-start strategy: try several initial points with CMA-ES
        n_starts = 4
        budget_per_start = max(100, (self.budget - 20) // n_starts)

        # Initial random samples to find good starting points
        init_candidates = []
        for i in range(min(20, self.budget)):
            if time.time() > deadline - 10:
                break
            if calls_made >= self.budget:
                break
            x_init = rng.uniform(-5, 5, self.dim)
            x_c, f_c = safe_call(x_init)
            init_candidates.append((f_c, x_c))

        # Sort by fitness, take top candidates as starting points
        init_candidates.sort(key=lambda t: t[0])
        start_points = [x for _, x in init_candidates[:min(n_starts, len(init_candidates))]]

        # If we have fewer starts than desired, add some
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-5, 5, self.dim))

        for sp in start_points:
            if time.time() > deadline - 10:
                break
            if calls_made >= self.budget:
                break
            sigma_init = 2.0
            cma_es(sp, sigma_init, budget_per_start)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
