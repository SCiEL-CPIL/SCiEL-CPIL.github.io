"""CMA-ES with multi-restart and deadline guard for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        # Use a fixed seed for reproducibility within a single instance
        rng = np.random.default_rng(42)

        # Strategy: split budget among several restarts.
        # For dim=5, CMA-ES is fast. We'll do up to 5 restarts if budget allows.
        # Each restart gets a chunk of the remaining budget.
        
        # First, do a quick random search to find a decent starting point
        # and get an initial f_opt estimate.
        n_init = min(20, budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Now do CMA-ES restarts from diverse starting points
        # Starting points: best found, plus random points, plus a few structured points
        start_points = [best_x.copy()]
        # Add some structured points
        for _ in range(4):
            start_points.append(rng.uniform(-5, 5, dim))

        # Sort start points by their function values (evaluate the new ones)
        start_fvals = []
        for sp in start_points:
            if np.array_equal(sp, best_x):
                start_fvals.append(best_f)
            else:
                sf = func(sp)
                start_fvals.append(sf)
                if sf < best_f:
                    best_f = sf
                    best_x = sp.copy()

        # Sort by fitness
        order = np.argsort(start_fvals)
        start_points = [start_points[i] for i in order]
        start_fvals = [start_fvals[i] for i in order]

        # Budget allocation: give more budget to better starts
        # But also ensure each restart gets at least some budget
        n_restarts = min(len(start_points), 5)
        # Allocate budget proportionally but with minimum
        remaining_budget = budget - n_init
        # Give 60% to best restart, 25% to second, 15% to third, etc.
        weights = [0.5, 0.25, 0.15, 0.075, 0.025]
        weights = weights[:n_restarts]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        budgets = [max(30, int(remaining_budget * w)) for w in weights]
        # Adjust so total doesn't exceed remaining
        total_alloc = sum(budgets)
        if total_alloc > remaining_budget:
            scale = remaining_budget / total_alloc
            budgets = [int(b * scale) for b in budgets]

        calls_made = n_init

        for restart_idx in range(n_restarts):
            if time.time() > deadline - 5:
                break
            if calls_made >= budget:
                break

            x0 = start_points[restart_idx].copy()
            f0 = start_fvals[restart_idx]
            alloc_budget = min(budgets[restart_idx], budget - calls_made)

            if alloc_budget < 10:
                break

            # CMA-ES parameters
            sigma = 1.0
            mu = max(1, alloc_budget // 10)  # number of selected individuals
            # Population size
            lambda_pop = max(4, 2 * mu + 1)
            
            # CMA-ES state
            mean = x0.copy()
            C = np.eye(dim)
            Cinv = np.eye(dim)
            step_size = sigma
            step_size_history = []
            
            # Cumulation paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Update rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            # mu_eff = effective selection mass
            mu_eff = sum(1 / (2 ** (i / mu)) for i in range(mu))
            
            # Weights
            weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
            weights /= weights.sum()
            
            # Cholesky of C
            try:
                B = np.linalg.cholesky(C)
            except:
                B = np.eye(dim)
            
            # Reset cumulation after first few steps
            n_steps_done = 0
            
            for step in range(alloc_budget):
                if time.time() > deadline - 2:
                    break
                if calls_made >= budget:
                    break
                
                # Generate population
                Y = rng.standard_normal((lambda_pop, dim))
                X = mean + step_size * (Y @ B.T)
                # Clip to box
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(x) for x in X])
                calls_made += lambda_pop
                
                # Sort by fitness
                idx = np.argsort(F)
                X_sel = X[idx[:mu]]
                F_sel = F[idx[:mu]]
                
                # Update best
                if F_sel[0] < best_f:
                    best_f = F_sel[0]
                    best_x = X_sel[0].copy()
                
                # Update mean
                old_mean = mean.copy()
                mean = weights @ X_sel
                
                # Update step size (CSA)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ ((mean - old_mean) / step_size))
                sigma_ratio = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (step + 1)))
                sigma_ratio = min(sigma_ratio, 1e5)
                step_size *= np.exp(c_sigma * (sigma_ratio / np.sqrt(1 - (1 - c_sigma) ** (2 * (step + 1))) - 1))
                step_size = min(max(step_size, 1e-10), 1e5)
                
                # Update covariance
                if (step + 1) % 2 == 0:  # Update every 2 steps for stability
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ ((mean - old_mean) / step_size))
                    v = 1 + (1 - c_c) * (1 - np.dot(p_c, p_c) / dim)
                    
                    # Rank-1 update
                    delta_h = (1.1 + 0.5 / c_c) * np.sqrt(1 - (1 - c_c) ** (2 * (step + 1))) - 1
                    h_sigma = int(np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (step + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_c) ** (2 * (step + 1))))
                    
                    C = (1 + c_1 * delta_h * h_sigma - c_1 - c_mu) * C + \
                        c_1 * np.outer(p_c, p_c) + \
                        c_mu * sum(w_i * np.outer(x - old_mean, x - old_mean) / step_size**2 
                                   for w_i, x in zip(weights, X_sel))
                    
                    # Ensure positive definiteness
                    C = (C + C.T) / 2
                    try:
                        B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                    except:
                        B = np.eye(dim)
                
                n_steps_done += 1
                
                # Convergence check
                if step_size < 1e-12:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
