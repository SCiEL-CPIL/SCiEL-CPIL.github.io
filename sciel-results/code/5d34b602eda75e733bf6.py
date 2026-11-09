"""Optimizer for BBOB f23 (Katsuura) with CMA-ES restarts and multi-start strategy."""
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with Latin Hypercube initialization for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        # Phase 1: Latin Hypercube sampling for initial exploration
        n_lhs = min(30, budget // 12)
        lhs_points = self._latin_hypercube(rng, n_lhs, dim)

        for i in range(n_lhs):
            if call_count >= budget or time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-restart CMA-ES
        # CMA-ES adapts to the problem's anisotropy, which is crucial for rotated Katsuura
        n_restarts = 3
        budget_per_restart = (budget - call_count) // n_restarts
        if budget_per_restart < 50:
            n_restarts = max(1, (budget - call_count) // 50)
            budget_per_restart = (budget - call_count) // n_restarts

        for restart in range(n_restarts):
            if call_count >= budget or time.time() > deadline:
                break

            # Choose starting point: best found or random restart
            if restart == 0:
                x_mean = best_x.copy()
            else:
                # Diversify: mix best with random perturbation
                if rng.random() < 0.3:
                    x_mean = rng.uniform(-4, 4, dim)
                else:
                    x_mean = best_x + 2.0 * rng.standard_normal(dim)
                    x_mean = np.clip(x_mean, -5, 5)

            # CMA-ES parameters
            sigma = 1.0
            pop_size = max(4, int(4 + 2 * np.log(dim)))  # ~9 for dim=5
            c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
            c_mu = min(1 - c_1, 2 * (lambda_ / 2 - 2 + 2 / lambda_) /
                       ((dim + 2) ** 2 + lambda_))
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
            step_size = 1.0
            damping = 0.5 + 2. / (dim + lambda_)

            # Initialize CMA-ES state
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            h_sigma = 1.0

            # Track best within this restart
            restart_best_f = best_f
            restart_best_x = best_x.copy()
            no_improve = 0

            for gen in range(budget_per_restart):
                if call_count >= budget or time.time() > deadline:
                    break

                # Generate population
                ys = np.array([x_mean + sigma * C_sqrt @ rng.standard_normal(dim)
                               for _ in range(pop_size)])
                ys = np.clip(ys, -5, 5)

                # Evaluate population
                fvals = np.array([func(y) for y in ys])
                call_count += pop_size

                # Update best
                for y, f in zip(ys, fvals):
                    if f < restart_best_f:
                        restart_best_f = f
                        restart_best_x = y.copy()
                    if f < best_f:
                        best_f = f
                        best_x = y.copy()

                # Sort by fitness
                order = np.argsort(fvals)
                ys_sorted = ys[order]
                fvals_sorted = fvals[order]

                # Weighted recombination
                weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
                weights = weights / weights.sum()
                mu_eff = 1.0 / np.sum(weights ** 2)

                # Update mean
                y_w = np.sum(weights[:, None] * ys_sorted[:int(np.ceil(mu_eff))], axis=0)
                x_mean = y_w

                # Update step size (CSA)
                y_old = x_mean - x_mean + sigma * C_inv_sqrt @ (y_w - x_mean)
                h_sigma = (np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1) / (pop_size + 1))) *
                           (1 + (1 - damping) * c_1 / (c_mu + c_1)))
                p_sigma = (1 - c_c) * p_sigma + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                          C_inv_sqrt @ (y_w - x_mean) / sigma

                if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1. / dim) * (1 - c_c):
                    sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance
                y_p_c = x_mean - x_mean + sigma * C_sqrt @ (y_w - x_mean)
                p_c = (1 - c_c) * p_c + c_c * y_p_c
                rank_mu = np.outer(ys_sorted[:int(np.ceil(mu_eff))].T,
                                   weights[:int(np.ceil(mu_eff))]) @ \
                          (ys_sorted[:int(np.ceil(mu_eff))] - x_mean).T / sigma ** 2
                rank_one = np.outer(p_c, p_c)
                delta_h = (1 - (1 - c_c) ** (2 * (gen + 1) / (pop_size + 1))) * \
                          (1 + (1 - damping) * c_1 / (c_mu + c_1)) * \
                          np.linalg.norm(p_c) ** 2 / (dim * (1 - (1 - c_c) ** 2))
                C = (1 + c_1 + c_mu - c_1) * C + c_1 * (rank_one - C) + \
                    c_mu * (rank_mu - C) + delta_h * C

                # Periodic Cholesky decomposition for stability
                if gen % 10 == 0:
                    try:
                        C = (C + C.T) / 2
                        C_sqrt = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                        C_inv_sqrt = np.linalg.inv(C_sqrt)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        C_sqrt = np.eye(dim)
                        C_inv_sqrt = np.eye(dim)

                # Check for stagnation
                if restart_best_f < best_f - 1e-6:
                    no_improve = 0
                else:
                    no_improve += 1

                # Early termination if stuck
                if no_improve > 100:
                    break

            # Update global best
            if restart_best_f < best_f:
                best_f = restart_best_f
                best_x = restart_best_x.copy()

        # Phase 3: Final local refinement with (1+1)-ES
        remaining = budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if call_count >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f

    def _latin_hypercube(self, rng, n, dim):
        """Generate Latin Hypercube points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = -5 + 10 * (perm + rng.random(n)) / n
        return points
