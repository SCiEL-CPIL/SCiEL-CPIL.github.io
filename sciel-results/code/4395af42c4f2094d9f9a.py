"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES with random restarts for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Random restarts with CMA-ES
        # For Rastrigin, we need to explore widely first
        # Use multiple restarts with different starting points

        # First, do a coarse random search to find promising regions
        n_random = min(50, budget // 10)
        for i in range(n_random):
            if check_deadline():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: CMA-ES from the best point found
        remaining_budget = budget - calls_used
        if remaining_budget > 10:
            best_x = self._cma_es(func, best_x, best_f, remaining_budget, dim, rng, deadline, calls_used)
            calls_used = budget  # approximate, we track internally

        return best_x, best_f

    def _cma_es(self, func, x0, f0, budget, dim, rng, deadline, calls_used):
        """CMA-ES implementation."""
        x = x0.copy()
        f = f0
        sigma = 0.5
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        
        # CMA-ES parameters
        mu = max(1, dim // 2)
        lambda_ = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        calls = 0
        best_x = x.copy()
        best_f = f
        
        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Generate lambda offspring
            ys = np.zeros((lambda_, dim))
            ws = np.zeros(lambda_)
            xs = np.zeros((lambda_, dim))
            
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Apply C matrix
                y = C_sqrt_inv @ z
                xs[i] = np.clip(x + sigma * y, -5, 5)
                ys[i] = z
                ws[i] = func(xs[i])
                calls += 1
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(ws)
            ys_sorted = ys[order]
            ws_sorted = ws[order]
            xs_sorted = xs[order]
            
            # Update best
            if ws_sorted[0] < best_f:
                best_f = ws_sorted[0]
                best_x = xs_sorted[0].copy()
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys_sorted[i]
            
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y_w)
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) > (1.4 + 1 / dim) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
            
            C = (1 + c_1 * (c_mu - c_1) + c_mu) * C + c_1 * (rank_1 - C) + c_mu * rank_mu
            C_sqrt_inv = np.linalg.inv(np.sqrtm(C + 1e-10 * np.eye(dim)))
            
            x = np.clip(x + sigma * y_w, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
