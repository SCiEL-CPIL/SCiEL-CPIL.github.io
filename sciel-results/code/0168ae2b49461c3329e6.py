"""Rewritten program: CMA-ES-like optimizer with adaptive step size and restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with population-based search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))
        
        # Initial state
        x_best = rng.uniform(-5, 5, d)
        f_best = func(x_best)
        budget_used = 1

        # CMA-ES parameters
        sigma = 2.0
        # Covariance matrix initialization (identity)
        C = np.eye(d)
        # Evolution paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Learning rates
        c_sigma = (lam / 2 + 1) / (d + 4 + lam / 2)
        c_c = (4 + lam / d) / (d + 4 + 2 * lam / d)
        c_1 = 2 / ((d + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 - 1) / (d + 2) ** 2 - 1))
        
        # Weights for recombination
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Chi approximation
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Track best solution found
        global_best_x = x_best.copy()
        global_best_f = f_best

        # Main loop with restarts
        calls_since_improvement = 0
        restart_threshold = 50 # Restart if no improvement for 50 calls
        
        while budget_used < budget:
            # Generate population
            # Sample from multivariate normal using Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d) * 1e-8
                L = np.sqrt(1e-8) * np.eye(d)
            
            population = []
            fitnesses = []
            
            for i in range(lam):
                if budget_used + 1 > budget:
                    break
                z = rng.standard_normal(d)
                y = x_best + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                population.append(y)
                fitnesses.append(fy)
                budget_used += 1
            
            if len(population) == 0:
                break
            
            # Sort by fitness
            indices = np.argsort(fitnesses)
            sorted_pop = [population[i] for i in indices]
            sorted_fit = [fitnesses[i] for i in indices]
            
            # Update best
            if sorted_fit[0] < global_best_f:
                global_best_f = sorted_fit[0]
                global_best_x = sorted_pop[0].copy()
                calls_since_improvement = 0
            else:
                calls_since_improvement += lam
            
            # Check for restart
            if calls_since_improvement > restart_threshold and budget_used < budget - lam:
                # Reset CMA-ES state, keep global best as new center
                x_best = global_best_x.copy()
                C = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                # Try a slightly larger sigma to explore, or smaller?
                # If stuck, maybe we are in a flat area. Let's try to jump out?
                # For linear functions, we just need to move in the correct direction.
                # Resetting C to Identity allows re-learning the direction.
                sigma = 1.0 
                calls_since_improvement = 0
                continue
            
            # Update center of mass
            x_mean = np.sum(np.array(sorted_pop[:mu]) * weights[:, np.newaxis], axis=0)
            
            # Update evolution paths
            # Compute step direction
            diff = x_mean - x_best
            norm_diff = np.linalg.norm(diff)
            
            if norm_diff > 1e-16:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / (sigma * norm_diff)
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget_used - 1)))) / chi_d < 1.4 + 2 / (d + 1)
                
                if h_sigma:
                    p_c = (1 - c_c) * p_c + c_c * diff / sigma
                else:
                    p_c = (1 - c_c) * p_c
                
                # Update covariance matrix
                delta_h = (1.4 + 2 / (d + 1)) - 1
                rank_one_update = (1 + (2 * delta_h - 1) * c_c) * np.outer(p_c, p_c)
                rank_mu_update = np.sum(weights[:, np.newaxis] * np.outer(np.array(sorted_pop[:mu]) - x_best, np.array(sorted_pop[:mu]) - x_best), axis=0)
                
                C = (1 + c_1 * (mu_eff - 1) - c_1 - c_mu) * C + c_1 * rank_one_update + c_mu * rank_mu_update
                C = (C + C.T) / 2  # Symmetrize
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
                
                # Update mean
                x_best = x_mean
            
            # Safety: ensure sigma is reasonable
            sigma = np.clip(sigma, 1e-4, 10.0)
            
            # Check for convergence
            if norm_diff < 1e-10 and sigma < 1e-6:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
