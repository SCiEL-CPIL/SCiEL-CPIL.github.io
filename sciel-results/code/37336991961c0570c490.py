"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart and coordinate polish for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Multi-restart CMA-ES with different initializations
        # Lunacek bi-Rastrigin has two basins; we try to find both
        # Strategy: several short CMA-ES runs with different starts, then polish

        # Phase 1: Multiple short CMA-ES runs to find promising regions
        n_restarts = 6
        budget_per_restart = max(30, budget // n_restarts)

        # Different starting points to explore different basins
        start_points = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            np.concatenate([np.full(dim//2, 3.0), np.full(dim - dim//2, -3.0)]),
        ]

        for restart_idx in range(n_restarts):
            if time.time() > deadline:
                break

            x0 = start_points[restart_idx % len(start_points)].copy()
            # Add small perturbation for diversity
            x0 += rng.normal(0, 0.1, dim)
            x0 = np.clip(x0, -5, 5)

            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
            mu = lambda_ // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights.sum()

            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

            c_1 = (2 + 2 * mueff / (dim + 1)) / ((dim + 1.3) ** 2 + mueff)
            c_mu = (2 + 2 * mueff / (dim + 1)) / ((dim + 1.3) ** 2 + mueff) - c_1
            c_mu = min(c_mu, (1 - c_1) * 0.5)

            # Initial step size - adapt to function landscape
            sigma = 1.5

            # CMA-ES state
            x_mean = x0.copy()
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)

            fx_mean = func(x_mean)
            update_best(x_mean, fx_mean)

            calls_made = 1
            max_iter = budget_per_restart

            for gen in range(max_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Generate population
                for i in range(lambda_):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break

                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    update_best(y, fy)

                    # Store for selection
                    if i == 0:
                        ys, fys = [y], [fy]
                    else:
                        ys.append(y)
                        fys.append(fy)

                if len(ys) < mu:
                    break

                # Selection: sort by fitness
                idx = np.argsort(fys)
                y_selected = np.array(ys)[idx[:mu]]
                f_selected = np.array(fys)[idx[:mu]]

                # Update mean
                x_mean_new = (weights * y_selected.T).T.sum(axis=0)
                x_mean_new = np.clip(x_mean_new, -5, 5)

                # Update evolution paths
                y_diff = (x_mean_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff / dim) * y_diff
                h_sigma = 1.0
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < \
                   (1.4 + 2 / (dim + 1)) * np.trace(C) / dim:
                    h_sigma = 1.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff / dim) * y_diff

                # Update covariance matrix
                delta = (h_sigma * c_c * (2 - c_c)) * np.outer(p_c, p_c)
                rank_mu = 0
                for i in range(mu):
                    zi = (y_selected[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c_1 * (delta / dim - 1) + c_mu * (rank_mu - C)) * C / (1 + c_1 * (delta / dim - 1) + c_mu * (rank_mu - C)) if False else C
                # Simpler update:
                C = (1 - c_1 - c_mu) * C + c_1 * (delta + np.eye(dim) * (c_1 / (1 - c_1) * 0)) + c_mu * rank_mu
                # Ensure positive semi-definite by eigenvalue correction
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                except:
                    C = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x_mean = x_mean_new
                fx_mean = func(x_mean)
                calls_made += 1
                update_best(x_mean, fx_mean)

        # Phase 2: Coordinate descent polish on the best point found
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

            for step in step_sizes:
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                improved = True
                while improved and calls_made < budget:
                    if time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls_made >= budget:
                            break
                        if time.time() > deadline:
                            break

                        for direction in [1, -1]:
                            if calls_made >= budget:
                                break
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            calls_made += 1
                            if f_trial < fx:
                                x = x_trial
                                fx = f_trial
                                update_best(x, fx)
                                improved = True

        return best_x, best_f
# EVOLVE-BLOCK-END
