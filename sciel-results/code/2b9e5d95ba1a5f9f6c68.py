"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global scan followed by CMA-ES refinement with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline - 2.0

        # Phase 1: Global scan to find promising basin
        n_scan = 40
        # Use stratified sampling
        n_per_dim = 3
        grid = np.meshgrid(*[np.linspace(-4, 4, n_per_dim) for _ in range(dim)])
        grid_points = np.column_stack([g.ravel() for g in grid])
        # Add jitter
        jitter = rng.uniform(-1, 1, size=(len(grid_points), dim))
        grid_points = np.clip(grid_points + jitter, -5, 5)
        
        # Shuffle and evaluate
        perm = rng.permutation(len(grid_points))
        scan_points = grid_points[perm[:n_scan]]
        
        evaluated = []
        for x in scan_points:
            if not time_ok():
                break
            fx = func(x)
            evaluated.append((fx, x.copy()))
        
        if not evaluated:
            x0 = rng.uniform(-4, 4, dim)
            fx0 = func(x0)
            return x0, fx0

        evaluated.sort(key=lambda t: t[0])
        best_fx = evaluated[0][0]
        best_x = evaluated[0][1].copy()

        # Phase 2: CMA-ES refinement
        remaining = budget - len(evaluated)
        if remaining <= 0:
            return best_x, best_fx

        # Initialize CMA-ES
        mean = best_x.copy()
        sigma = 0.5
        C = np.eye(dim)
        # Precompute sqrt(C) for sampling
        C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
        
        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of parents
        lambda_pop = 2 * mu  # population size
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        best_fx_cma = best_fx
        best_x_cma = best_x.copy()
        
        n_calls = len(evaluated)
        max_iters = remaining // lambda_pop
        
        for it in range(max_iters):
            if not time_ok():
                break
            if n_calls >= budget:
                break

            # Sample population
            B = np.linalg.cholesky(C)
            pop = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                pop[i] = mean + sigma * (B @ z)
            
            # Evaluate population
            pop_f = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if not time_ok() or n_calls >= budget:
                    pop_f[i] = np.inf
                    continue
                x_i = np.clip(pop[i], -5, 5)
                pop_f[i] = func(x_i)
                n_calls += 1
                if pop_f[i] < best_fx_cma:
                    best_fx_cma = pop_f[i]
                    best_x_cma = x_i.copy()

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_sorted = pop[order]
            pop_f_sorted = pop_f[order]

            # Update mean
            old_mean = mean.copy()
            mean = sum(weights[i] * pop_sorted[i] for i in range(mu))

            # Update C
            y = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * np.outer(pop_sorted[i] - old_mean, pop_sorted[i] - old_mean) / sigma ** 2 for i in range(mu))
            
            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 0.01 * np.eye(dim)
                B = np.linalg.cholesky(C)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10)

        return best_x_cma, best_fx_cma
# EVOLVE-BLOCK-END
