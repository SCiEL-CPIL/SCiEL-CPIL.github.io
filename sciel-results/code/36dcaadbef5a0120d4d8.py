"""Model-and-optimize optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize: detect attractive sector geometry, then optimize."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Phase 1: Structured sampling to detect the attractive sector
        # The f6 (Attractive Sector) function has the form:
        # f(x) = f_transformed(x) + 0.01 * ||x||^2 * ||x||^2 (or similar)
        # where f_transformed is a shifted/rotated basic function.
        # Key property: there's a "valley" or sector where the function decreases.
        
        # First, probe the origin and some structured points
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_f = f0

        # Probe along each axis in both directions to detect the attractive direction
        probe_points = [x0]
        for i in range(dim):
            for sign in [1.0, -1.0]:
                for dist in [0.5, 1.0, 2.0, 3.0, 4.0]:
                    p = np.zeros(dim)
                    p[i] = sign * dist
                    probe_points.append(p)
        
        # Also add some random probes
        for _ in range(10):
            p = rng.uniform(-4, 4, dim)
            probe_points.append(p)

        for p in probe_points:
            if calls >= budget or check_deadline():
                break
            fp = func(p)
            calls += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Phase 2: Use a model-based approach
        # Fit a quadratic model to the evaluated points and use it to guide search
        # But with only 500 budget, let's use a more direct approach:
        # CMA-ES with good initialization from the probe phase

        # Use CMA-ES starting from the best point found
        x_mean = best_x.copy()
        sigma = 1.0
        
        # Population size
        lam = max(4, int(4 + 3 * np.log(max(dim, 2))))
        remaining = budget - calls
        max_iter = max(1, remaining // lam)

        # CMA-ES state
        C = np.eye(dim)
        try:
            C_chol = np.linalg.cholesky(C)
        except:
            C_chol = np.eye(dim)

        # Update rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # CMA-ES main loop
        for iteration in range(max_iter):
            if check_deadline() or calls >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if check_deadline() or calls >= budget:
                    break
                z = rng.standard_normal(dim)
                try:
                    y = x_mean + sigma * (C_chol @ z)
                except:
                    y = x_mean + sigma * z
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

            if len(offspring) < mu:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu]

            # Weighted mean of selected
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * selected[i][0]

            # Update paths
            try:
                C_inv = np.linalg.inv(C)
                C_inv_sqrt = np.linalg.cholesky(C_inv)
                C_inv_sqrt_T = C_inv_sqrt.T
            except:
                C_inv_sqrt_T = np.eye(dim)

            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt_T @ diff)

            p_sigma_norm = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            delta_h = (1.8 + 2 / (mu_eff + 1)) * (1 - h_sigma)
            rank_one = np.outer(p_c + delta_h * p_c, p_c + delta_h * p_c)

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = selected[i][0] - x_mean
                rank_mu += weights[i] * np.outer(yi, yi) / (sigma ** 2)

            C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * C
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x_mean = x_new_mean

            try:
                C_chol = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                C_chol = np.eye(dim)
                sigma = 1.0

            # Track best
            best_in_gen = min(selected, key=lambda x: x[1])
            if best_in_gen[1] < best_f:
                best_f = best_in_gen[1]
                best_x = best_in_gen[0].copy()

        # Phase 3: Local refinement with (1+1)-ES
        if not check_deadline() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma_local = max(0.05, sigma * 0.3)
            successes = 0
            total = 0
            for _ in range(min(30, budget - calls)):
                if check_deadline() or calls >= budget:
                    break
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                total += 1
                if total >= 5:
                    sr = successes / total
                    if sr > 0.2:
                        sigma_local *= 1.3
                    elif sr < 0.2:
                        sigma_local *= 0.7
                    successes = 0
                    total = 0
            if fx < best_f:
                best_x = x
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
