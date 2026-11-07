"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid optimizer exploiting Rastrigin structure with CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls_made = 0
        rng = np.random.default_rng(42)

        # Phase 1: Evaluate integer grid points to find promising candidates
        # Rastrigin has global minima at integer coordinates; with rotation/shift,
        # the true minimum is near some rotated/shifted integer point
        grid_vals = np.arange(-4, 5, 1)  # -4 to 4 inclusive
        grid_points = []
        for v0 in grid_vals:
            for v1 in grid_vals:
                for v2 in grid_vals:
                    for v3 in grid_vals:
                        for v4 in grid_vals:
                            grid_points.append([v0, v1, v2, v3, v4])
        
        # Shuffle grid points for diversity in exploration order
        rng.shuffle(grid_points)
        
        best_x = None
        best_f = np.inf
        
        # Evaluate up to 50 grid points to establish a good baseline
        grid_budget = min(50, budget // 4)
        for pt in grid_points:
            if calls_made >= grid_budget:
                break
            x_pt = np.clip(np.array(pt, dtype=float), -5, 5)
            f_pt = func(x_pt)
            calls_made += 1
            if f_pt < best_f:
                best_f = f_pt
                best_x = x_pt.copy()
        
        # Phase 2: Run CMA-ES with multiple restarts from best candidates
        # Use remaining budget for CMA-ES refinement
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f
        
        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = 5  # number of parents
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Run CMA-ES with restarts
        num_restarts = 3
        budget_per_restart = remaining // num_restarts
        best_overall_x = best_x.copy()
        best_overall_f = best_f
        
        for restart in range(num_restarts):
            if calls_made >= budget:
                break
            
            # Initialize CMA-ES from best point found so far
            x = best_overall_x.copy()
            f_best = best_overall_f
            
            # Start with a moderate sigma
            sigma = 1.0 if restart == 0 else 0.5
            
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # CMA-ES loop for this restart
            while calls_made < budget and calls_made < grid_budget + (restart + 1) * budget_per_restart:
                # Generate population
                n_offspring = min(lambda_, budget - calls_made)
                offspring = []
                for i in range(n_offspring):
                    z = rng.standard_normal(dim)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    offspring.append((fy, y))
                    calls_made += 1
                
                if len(offspring) == 0:
                    break
                
                # Sort offspring
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < f_best:
                    f_best = offspring[0][0]
                    x_best = offspring[0][1].copy()
                else:
                    x_best = x
                
                # Update overall best
                if f_best < best_overall_f:
                    best_overall_f = f_best
                    best_overall_x = x_best.copy()
                
                # Update mean
                y_w = np.array([offspring[i][1] for i in range(min(mu, len(offspring)))])
                y_w -= x
                x_new = x + sigma * np.sum(weights[:len(y_w)][:, np.newaxis] * y_w, axis=0)
                
                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * np.sum(weights[:len(y_w)][:, np.newaxis] * y_w, axis=0)
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) < (1 + 2 / dim) * np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made + 1))) else 0.0
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * np.sum(weights[:len(y_w)][:, np.newaxis] * y_w, axis=0)
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 / (1 - (1 - c_c) ** 2) * np.outer(p_c, p_c) + c_mu / (1 - (1 - c_c) ** 2) * np.sum(weights[:len(y_w)][:, np.newaxis, np.newaxis] * np.outer(y_w, y_w), axis=0)
                
                # Renormalize covariance
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 0.01, 3.0)
                
                x = x_new
                
                # Early stopping if sigma is very small
                if sigma < 0.01:
                    break
        
        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
