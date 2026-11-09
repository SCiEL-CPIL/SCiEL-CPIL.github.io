"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global scan followed by CMA-ES population-based refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline - 5.0

        # Phase 1: Global Scan with stratified sampling
        n_scan = min(50, budget // 4)
        n_per_dim = max(2, int(np.ceil(n_scan ** (1.0 / dim))))
        
        scan_points = []
        for i in range(n_scan):
            x = np.zeros(dim)
            for d in range(dim):
                bin_idx = (i + d * n_scan // n_per_dim) % n_per_dim
                frac = (bin_idx + rng.uniform()) / n_per_dim
                x[d] = frac * 10.0 - 5.0
            scan_points.append(x)

        best_fx = float('inf')
        best_x = None
        candidates = []

        for x in scan_points:
            if not time_ok():
                break
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if not candidates:
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            return x0, fx0

        candidates.sort(key=lambda t: t[0])
        remaining_budget = budget - len(candidates)

        # Phase 2: CMA-ES refinement
        lambda_pop = max(10, 4 + int(3 * np.log(dim)))
        
        if len(candidates) >= 2:
            init_sigma = np.linalg.norm(candidates[0][1] - candidates[1][1])
            init_sigma = max(0.1, min(init_sigma, 2.0))
        else:
            init_sigma = 1.0

        mean = candidates[0][1].copy()
        C = np.eye(dim)
        sigma = init_sigma
        
        mu = lambda_pop // 2
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        C_chol = np.linalg.cholesky(C)
        
        no_improvement = 0
        stall_threshold = 5 * lambda_pop
        generation = 0

        while time_ok() and remaining_budget > lambda_pop:
            # Generate population
            offspring_fx = []
            for i in range(lambda_pop):
                if remaining_budget <= 0:
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * C_chol @ z
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                offspring_fx.append((fy, y))
                remaining_budget -= 1

            if len(offspring_fx) < 2:
                break

            offspring_fx.sort(key=lambda t: t[0])
            
            best_offspring_fx = offspring_fx[0][0]
            if best_offspring_fx < best_fx - 1e-10:
                best_fx = best_offspring_fx
                best_x = offspring_fx[0][1].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            # Update mean using weighted recombination
            m_actual = min(mu, len(offspring_fx))
            new_mean = np.zeros(dim)
            for i in range(m_actual):
                new_mean += weights[i] * offspring_fx[i][1]
            
            # Compute C^{-1/2} via Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-15 * np.eye(dim))
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Update path variables
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma

            # Update sigma with proper h_sigma
            expected_norm = np.sqrt(mu_eff) * (1 + 1 / (dim + 1))
            threshold = (1.4 + 2 / (dim + 1)) * expected_norm
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))
            if denom < 1e-10:
                denom = 1.0
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / denom < threshold else 0.0
            
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance matrix (correct signs)
            delta_h = h_sigma * (1 / c_sigma - 1)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(m_actual):
                diff = offspring_fx[i][1] - mean
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 + c_1 * (delta_h - 1) + c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = 0.5 * (C + C.T)

            # Update Cholesky factor
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)

            mean = new_mean
            generation += 1

            # Stagnation detection
            if no_improvement > stall_threshold:
                mean = best_x.copy()
                sigma = max(0.3, sigma * 0.5)
                C = np.eye(dim)
                C_chol = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improvement = 0
                generation = 0

        # Final polish
        if time_ok() and remaining_budget > 5:
            polish_budget = min(remaining_budget, 20)
            x = best_x.copy()
            fx = best_fx
            step = 0.01
            for _ in range(polish_budget):
                if not time_ok() or remaining_budget <= 0:
                    break
                y = x + step * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                remaining_budget -= 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    step *= 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
