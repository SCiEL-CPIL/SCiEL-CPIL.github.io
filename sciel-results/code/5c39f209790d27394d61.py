"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Deterministic global-to-local hierarchical search for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def time_ok():
            return time.time() < deadline

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        n = self.dim

        # Phase 1: Coarse grid scan to identify the global basin
        # Use a stratified grid with ~400 points in [-5, 5]^n
        # For 5D, we can use a 3-point grid per dimension (3^5 = 243) plus some refinement
        # Or use a more efficient sampling approach
        
        # Let's use a combination: coarse grid + random samples
        # Coarse grid: 3 points per dimension in [-4, 4] (where minimum is guaranteed to be)
        # That's 3^5 = 243 points
        # Plus 100 random points for diversity
        
        grid_points = []
        # 3-point grid per dimension in [-4, 4]
        grid_vals = [-4.0, 0.0, 4.0]
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            grid_points.append([grid_vals[i1], grid_vals[i2], grid_vals[i3], grid_vals[i4], grid_vals[i5]])
        
        # Add some random points for better coverage
        rng = np.random.default_rng(42)
        for _ in range(100):
            grid_points.append(list(rng.uniform(-4, 4, n)))
        
        # Evaluate all grid points
        for gp in grid_points:
            if not time_ok() or calls_made >= self.budget:
                break
            eval_point(np.array(gp))
        
        # Phase 2: Localized CMA-ES centered on the best point found
        if time_ok() and calls_made < self.budget and best_x is not None:
            remaining_budget = self.budget - calls_made
            # Use CMA-ES with the remaining budget
            x0 = best_x.copy()
            sigma0 = 1.5  # Start with a reasonable step size
            
            # CMA-ES parameters
            mu = max(1, n // 2)
            lambda_ = 2 * mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, math.sqrt(mueff / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            x = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            
            # Initial evaluation already done for x0
            
            for gen in range(remaining_budget // lambda_ + 1):
                if not time_ok() or calls_made >= self.budget:
                    break
                
                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                
                # Evaluate offspring
                vals = []
                for y in offspring:
                    fy = eval_point(y)
                    vals.append(fy)
                
                if calls_made >= self.budget:
                    break
                
                # Sort by fitness
                idx = np.argsort(vals)
                selected = [offspring[i] for i in idx[:mu]]
                
                # Update mean
                x_new = sum(w * xi for w, xi in zip(weights, selected))
                x_new = np.clip(x_new, -5, 5)
                
                # Update evolution paths
                if sigma > 0:
                    v = (x_new - x) / sigma
                    pc = (1 - c_c) * pc + math.sqrt(c_c * (2 - c_c) * mueff) * v
                    ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B.T @ (D * v))
                    
                    # Adapt covariance matrix
                    if (pc @ pc) / n < (1 + 4 / n) * (1 - c_c):
                        rank1 = np.outer(pc, pc)
                    else:
                        rank1 = np.zeros((n, n))
                    rankmu = sum(w * np.outer(xi - x, xi - x) for w, xi in zip(weights, selected)) / sigma ** 2
                    
                    C = (1 + c_1 * (n + 1 - 2) + c_mu - c_1) * C + c_1 * (rank1 - C) + c_mu * rankmu
                    C = (C + C.T) / 2
                
                # Eigen decomposition
                try:
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-12))
                    B = B2
                    D = D2
                except np.linalg.LinAlgError:
                    break
                
                # Step size adaptation
                if sigma > 0:
                    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-12, 10.0)
                
                x = x_new
        
        # Phase 3: Local refinement with coordinate descent
        if time_ok() and calls_made < self.budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.3
            for _ in range(30):
                if not time_ok() or calls_made >= self.budget:
                    break
                improved = False
                for d in range(n):
                    if calls_made >= self.budget:
                        break
                    for sign in [1, -1]:
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_point(y)
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
