"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with Nelder-Mead polish for bent cigar optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Initial point near center
        x_best = np.zeros(n)
        fx_best = func(x_best)
        budget_used = 1
        
        # CMA-ES parameters
        # Start with a larger sigma to explore the box quickly
        sigma = 2.0
        pop_size = 4  # lambda
        # Initial covariance is identity
        C = np.eye(n)
        C_inv = np.eye(n)
        
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Adaptation parameters
        mu = pop_size // 2  # weights for top mu
        weights = np.ones(mu)
        weights /= np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff)
        c_c = (4 + mueff / n) / (n + 4)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff)))
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Cholesky factor of C
        # We will maintain C and update it, but for simplicity in this constrained 
        # black-box setting, we'll use a simplified update or standard CMA-ES.
        # Given the small dimension (5) and budget (500), full CMA-ES is feasible.
        
        # Helper to compute Cholesky
        def cholesky(mat):
            try:
                return np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
                # If not positive definite, add small diagonal
                return np.linalg.cholesky(mat + 1e-10 * np.eye(n))
        
        B = cholesky(C)
        D = np.sqrt(np.diag(B))
        B_inv = np.linalg.inv(B)
        
        # Track budget
        for iteration in range(budget - 1):
            if time.time() > deadline:
                break
            if budget_used >= budget:
                break
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                # Standard normal
                z = rng.standard_normal(n)
                # Transform to CMA-ES space
                step = sigma * B @ z
                x_new = x_best + step
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                
                if budget_used >= budget:
                    break
                fx_new = func(x_new)
                budget_used += 1
                offspring.append((fx_new, x_new))
            
            if not offspring:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            fx_best_off, x_best_off = offspring[0]
            if fx_best_off < fx_best:
                x_best, fx_best = x_best_off, fx_best_off
            
            # Update evolution paths and covariance
            # Weighted mean of selected offspring
            x_mean = np.zeros(n)
            for i in range(mu):
                x_mean += weights[i] * offspring[i][1]
            
            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv @ (x_mean - x_best)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (iteration + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            
            sigma *= np.exp((1 / c_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update pc
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_best) / sigma
            
            # Update C
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                v = (x_mean - x_best) / sigma # Actually should be (offspring[i][1] - x_best) / sigma
                # Correction: rank-mu update uses the selected offspring
                v = (offspring[i][1] - x_best) / sigma
                rank_mu += weights[i] * np.outer(v, v)
            
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C + c_1 * (rank_one - (1 - hsig * (1 - c_c)) * C) + c_mu * rank_mu
            
            # Ensure positive definite
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(n)
            
            # Update B and D
            B = cholesky(C)
            D = np.sqrt(np.diag(B))
            B_inv = np.linalg.inv(B)
            C_inv = B_inv.T @ B_inv
            
            # Update center
            x_best = x_mean # Move center to weighted mean? Standard CMA-ES moves to weighted mean.
            # But we keep x_best as the best found for clipping and reporting?
            # Actually, CMA-ES updates the mean m. Let's use x_mean as the new center.
            # However, the best point found so far is tracked separately for the return.
            # For the next iteration, we generate around x_mean.
            
            # Update the best point tracking
            # The variable x_best in the loop above was updated to the best offspring.
            # But for CMA-ES, the distribution center is x_mean.
            # Let's use x_mean as the center for the next generation.
            # But we need to be careful: if x_mean is worse than previous best, 
            # the distribution might drift. Standard CMA-ES handles this via the paths.
            
            # Re-center
            # In standard CMA-ES, x_best is replaced by the weighted mean of the selected offspring.
            # Let's do that.
            # But we also track the global best for the return value.
            # The variable x_best is currently the best offspring.
            # Let's rename: x_center is the distribution center.
            # Initially x_center = 0.
            
            # Wait, the code above updated x_best to the best offspring.
            # For CMA-ES, the next generation is centered at the weighted mean of the selected ones.
            # Let's use x_mean as the new center.
            
            # However, the clipping might cause issues if the mean goes out of bounds.
            # The offspring are clipped, so the mean of clipped points might be biased.
            # This is a known issue with CMA-ES on bounded domains.
            # A common heuristic is to reflect or just clip the center.
            # Let's clip the center.
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update the center for the next iteration
            # We will use x_mean as the new x_best for the generation step?
            # No, x_best in the loop `x_best + step` is the center.
            # So we set x_best = x_mean.
            x_best = x_mean
            
            # Optional: Local polish if we are close and have budget
            # If we have spent more than 400 calls and sigma is small, switch to Nelder-Mead
            if budget_used > 400 and sigma < 0.1:
                # Use remaining budget for Nelder-Mead
                # We need to count calls inside minimize
                # SciPy's minimize doesn't expose call count easily without wrapping.
                # We can wrap func.
                calls_made = [0]
                def wrapped_func(x):
                    calls_made[0] += 1
                    if calls_made[0] > budget - budget_used:
                        raise ValueError("Budget exceeded")
                    return func(x)
                
                try:
                    res = minimize(wrapped_func, x_best, method='Nelder-Mead', 
                                   options={'maxiter': budget - budget_used, 'xatol': 1e-6, 'fatol': 1e-6})
                    if res.success and res.fun < fx_best:
                        x_best = res.x
                        fx_best = res.fun
                except:
                    pass
                break
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
