# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for Katsuura (f23) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        total_calls = 0

        def check_deadline():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Quick multi-start to find a good initial point
        n_starts = 8
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Random points spread across the domain
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        for i in range(n_starts):
            if check_deadline() or total_calls >= budget:
                break
            x0 = np.clip(start_points[i], -5, 5)
            fx0 = func(x0)
            total_calls += 1
            try_update(x0, fx0)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES from the best starting point
        # CMA-ES parameters
        sigma = 1.0
        pop_size = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        # We'll use a smaller pop to conserve budget
        pop_size = min(pop_size, 12)

        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        # Covariance matrix
        C = np.eye(dim)
        # Step sizes for axes
        ds = np.ones(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 - mu_eff) / (dim + 4 - mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while total_calls < budget and not check_deadline():
            # Generate population
            offspring = []
            for i in range(pop_size):
                if check_deadline() or total_calls >= budget:
                    break
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Apply covariance transformation (simplified: use diagonal approximation)
                # For simplicity, use the current ds and C
                y = x + sigma * (ds * (C @ z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            for fy, y in offspring:
                try_update(y, fy)

            # Check convergence
            if best_f - offspring[0][0] < 1e-10:
                # No improvement
                if sigma < 1e-6:
                    break

            # Update CMA-ES parameters
            w = weights[:len(offspring)]
            # Weighted mean of selected individuals
            y_w = np.zeros(dim)
            for i in range(len(offspring)):
                if i >= len(w):
                    break
                y_w += w[i] * (offspring[i][1] - x)
            y_w /= np.sum(w)

            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_w / (sigma * chi_n))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)

            # Update C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w / sigma)
            delta_heav = 1 + 2 * max(0, (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** 20)) / chi_n - 1)
            C = (1 - (c_1 + c_mu) * delta_heav) * C + c_1 * (np.outer(p_c, p_c) + (np.linalg.norm(p_c) ** 2 - 1) * np.eye(dim))
            for i in range(len(offspring)):
                if i >= len(w):
                    break
                diff = (offspring[i][1] - x) / sigma
                C += c_mu * w[i] * np.outer(diff, diff)

            # Update mean
            x = x + y_w
            x = np.clip(x, -5, 5)
            fx = func(x) if total_calls < budget and not check_deadline() else fx
            if total_calls < budget and not check_deadline():
                total_calls += 1
                try_update(x, fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
