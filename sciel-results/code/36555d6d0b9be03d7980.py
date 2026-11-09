"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal step-size ES with Rank-1 + Rank-mu natural gradient adaptation and stagnation restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial diverse sampling to capture multiple basins early for AOCC
        best_x = None
        best_fx = np.inf
        calls = 0

        def eval_point(y):
            nonlocal best_x, best_fx, calls
            fy = func(y)
            calls += 1
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            return fy

        # Initial diverse sampling: 10 random points to explore the space
        n_initial = min(10, self.budget)
        for _ in range(n_initial):
            y = rng.uniform(-5, 5, dim)
            eval_point(y)

        if calls >= self.budget:
            return best_x, best_fx

        # Diagonal CMA-ES parameters
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of parents

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Adaptation rates for diagonal step sizes
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Stagnation detection parameters
        stagnation_window = 3 * lambda_
        restarts = 0
        max_restarts = 5

        while calls < self.budget and time.time() < deadline and restarts <= max_restarts:
            # Initialize CMA-ES state
            if best_x is None:
                x = rng.uniform(-5, 5, dim)
                eval_point(x)
            else:
                x = best_x.copy()

            # Diagonal step sizes (one per dimension)
            sigma = np.ones(dim) * 2.0

            # Evolution paths (diagonal version)
            ps = np.zeros(dim)  # step size evolution path
            pc = np.zeros(dim)  # covariance evolution path (diagonal)

            # Track stagnation
            best_at_window_start = best_fx
            calls_at_window_start = calls
            window_calls = 0

            while calls < self.budget and time.time() < deadline:
                remaining = self.budget - calls
                lambda_current = min(lambda_, remaining)

                if lambda_current < 2:
                    break

                # Generate population
                ys = []
                fs = []
                for i in range(lambda_current):
                    z = rng.standard_normal(dim)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = eval_point(y)
                    ys.append(y)
                    fs.append(fy)

                    if time.time() > deadline:
                        break

                # Sort by fitness
                if len(ys) < mu:
                    break

                idx = np.argsort(fs)[:mu]
                ys_sorted = [ys[i] for i in idx]
                fs_sorted = [fs[i] for i in idx]

                # Update mean
                x_old = x.copy()
                x = sum(w[i] * ys_sorted[i] for i in range(mu))

                # Update evolution path ps for step sizes
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x - x_old) / sigma

                # Heuristic for sigma update
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((calls + 1) // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n

                # Update sigma (overall scale factor applied to all dimensions)
                sigma_scale = np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma * sigma_scale, 1e-10, 10.0)

                # Update evolution path pc for covariance (diagonal)
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma

                # Update diagonal step sizes using Rank-1 + Rank-mu updates
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)

                # Rank-1 update from pc
                sigma_update = c_1 * np.outer(pc, pc).diagonal()

                # Rank-mu update from weighted outer products
                for i in range(mu):
                    diff = (ys_sorted[i] - x_old) / sigma
                    sigma_update += c_mu * w[i] * diff**2

                # Combine updates
                sigma = sigma * np.exp((c_1 * delta_hsig - c_1 - c_mu) + sigma_update / sigma**2)
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Ensure sigma stays positive and reasonable
                sigma = np.maximum(sigma, 1e-10)

                # Check for stagnation
                window_calls += lambda_current
                if window_calls >= stagnation_window:
                    if best_fx < best_at_window_start - 1e-10:
                        # Progress made, reset window
                        best_at_window_start = best_fx
                        calls_at_window_start = calls
                        window_calls = 0
                    else:
                        # Stagnation detected, break out to restart
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
