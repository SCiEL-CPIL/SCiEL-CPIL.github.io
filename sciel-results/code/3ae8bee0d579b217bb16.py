"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using CMA-ES with multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with multi-start initialization for Lunacek bi-Rastrigin.
    
    Strategy:
    1. Multi-start with diverse points to find both Lunacek basins
    2. CMA-ES refinement from the best starting point
    3. Additional CMA-ES runs from other promising points if budget remains
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start to find promising basins
        # Lunacek has two basins: one near origin, one at a shifted location
        # Use several diverse starting points
        n_starts = 4
        start_budget = min(50, total_budget // (n_starts + 1))
        
        start_points = []
        # Origin
        start_points.append(np.zeros(n))
        # Points near typical Lunacek offset (shift is hidden but often around 2-4)
        for i in range(n_starts - 1):
            # Spread starts across the domain
            start_points.append(rng.uniform(-4, 4, n))

        # Evaluate starting points
        for sp in start_points:
            if calls_made >= total_budget:
                break
            fx = safe_func(sp)

        # Phase 2: CMA-ES refinement from best point
        if best_x is not None and calls_made < total_budget - 10:
            remaining = total_budget - calls_made
            # Run CMA-ES with the remaining budget
            cma_budget = min(remaining, 400)
            x_best, f_best = self._cma_es(
                safe_func, best_x.copy(), n, cma_budget, deadline, rng
            )
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 3: If budget remains, try CMA-ES from second-best starting point
        if calls_made < total_budget - 10 and time.time() < deadline:
            # Find second-best starting point
            other_starts = [sp for sp in start_points if not np.allclose(sp, best_x)]
            if other_starts:
                # Try the one farthest from best
                sp2 = max(other_starts, key=lambda s: np.linalg.norm(s - best_x))
                remaining = total_budget - calls_made
                if remaining > 20:
                    cma_budget = min(remaining, 100)
                    x2, f2 = self._cma_es(
                        safe_func, sp2.copy(), n, cma_budget, deadline, rng
                    )
                    if f2 < best_f:
                        best_f = f2
                        best_x = x2.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_func(best_x)

        return best_x, best_f

    def _cma_es(self, func, x0, dim, budget, deadline, rng):
        """CMA-ES implementation for continuous optimization."""
        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # lambda = 4 + 3*ln(n)
        pop_size = max(pop_size, 4)
        
        # Strategy parameters
        c_sigma = (1 + 2 * max(0, np.sqrt((pop_size - 1) / dim) - 1)) / (dim + 2)
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))
        mu = pop_size // 2  # number of weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Initial state
        x_mean = x0.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Track best
        best_x = x_mean.copy()
        best_f = func(x_mean)
        
        calls_used = 1
        
        for gen in range(budget // pop_size + 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_used += 1
                offspring.append(x_new)
                offspring_f.append(f_new)
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]
            
            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()
            
            # Update mean
            x_new_mean = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            diff = [offspring[i] - x_mean for i in range(mu)]
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c) ** 2) * C) + \
                c_mu * np.sum([np.outer(diff[i], diff[i]) for i in range(mu)], axis=0)
            
            # Ensure positive definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D ** -1)) @ (x_new_mean - x_mean)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new_mean - x_mean)
            
            # Update mean
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
            
            # Convergence check
            if np.linalg.norm(pc) > (1.4 + 1 / c_c) * chi_n:
                break
            if sigma < 1e-6:
                break
            if np.max(C) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
