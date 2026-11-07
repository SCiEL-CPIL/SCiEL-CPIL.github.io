"""Model-and-optimize hybrid for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Hybrid: detect attractive sector geometry, then CMA-ES with informed start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin for 330s wall-clock limit
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Phase 1: Structured sampling to detect attractive sector
        # f6 = (||x||^2 - 200)^2 * (1 + 0.01 * sin(12 * ||x||)) + 100 * (1 - exp(-||x||^2 / 20))
        # plus shift/rotation/offset. The key structure is a ring-shaped valley.
        # We sample along radial directions to find the ring radius and center.

        # First, probe a grid to find approximate center and scale
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        # Probe several points to get a sense of the landscape
        probe_points = []
        # Origin and axis-aligned points at various distances
        for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
            for axis in range(dim):
                p = np.zeros(dim)
                p[axis] = r
                probe_points.append(p)
                p2 = np.zeros(dim)
                p2[axis] = -r
                probe_points.append(p2)
        
        # Also probe some random directions
        for _ in range(10):
            direction = rng.standard_normal(dim)
            direction /= (np.linalg.norm(direction) + 1e-10)
            for r in [2.0, 3.0, 4.0]:
                probe_points.append(direction * r)

        for p in probe_points:
            if calls >= budget or check_deadline():
                break
            p = np.clip(p, -5.0, 5.0)
            fp = func(p)
            calls += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Phase 2: Estimate the center of the attractive sector
        # The f6 function has a ring-shaped valley. We estimate the center
        # by finding the point that minimizes the function, which should be
        # near the ring center (the shift vector).
        
        # Use the best point found so far as initial guess for CMA-ES
        x_mean = best_x.copy()
        
        # Phase 3: CMA-ES with informed initialization
        # Population size
        lam = max(4, int(4 + 3 * np.log(max(dim, 2))))
        # Limit to available budget
        remaining = budget - calls
        max_iter = max(1, remaining // lam)
        
        # CMA-ES state
        sigma = 1.0  # initial step size - informed by the scale of the problem
        C = np.eye(dim)
        C_inv = np.eye(dim)
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
        
        # Damping for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        stagnation_counter = 0
        best_f_cma = best_f
        best_x_cma = best_x.copy()
        
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
            x_new_mean = sum(weights[i] * selected[i][0] for i in range(mu))
            x_new_mean = np.array(x_new_mean)

            # Update paths
            try:
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

            # Update C_inv and C_chol
            x_mean = x_new_mean
            try:
                C_inv = np.linalg.inv(C)
                C_chol = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_chol = np.eye(dim)
                sigma = 1.0

            # Track best
            best_in_gen = min(selected, key=lambda x: x[1])
            if best_in_gen[1] < best_f_cma:
                best_f_cma = best_in_gen[1]
                best_x_cma = best_in_gen[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # If stagnating, try a restart with larger sigma or different center
            if stagnation_counter > 15 and sigma > 0.01:
                # Try to escape: increase sigma temporarily
                sigma *= 2.0
                sigma = np.clip(sigma, 1e-8, 10.0)
                stagnation_counter = 0

        # Final local polish with (1+1)-ES around best found
        if not check_deadline() and calls < budget:
            x = best_x_cma.copy()
            fx = best_f_cma
            sigma_local = max(0.05, sigma * 0.5)
            successes = 0
            total = 0
            max_polish = min(100, budget - calls)
            for _ in range(max_polish):
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
                    if sr > 0.25:
                        sigma_local *= 1.3
                    elif sr < 0.15:
                        sigma_local *= 0.7
                    successes = 0
                    total = 0
            if fx < best_f_cma:
                best_x_cma = x
                best_f_cma = fx

        return best_x_cma, best_f_cma
# EVOLVE-BLOCK-END
