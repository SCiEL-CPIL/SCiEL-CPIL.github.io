"""
Multi-Start CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
Strategy:
  - CMA-ES is superior for ill-conditioned functions because it adapts the full
    covariance matrix, aligning the search distribution with the function's
    elongated minima.
  - We use a small population (lambda=15) to keep the number of evaluations low
    per generation, allowing more generations within the 500-call budget.
  - Multi-start: if the current run stagnates (no improvement in fitness for a
    while), we restart the CMA-ES from the current global best with a reduced
    sigma (step size) to refine the solution, or with a slightly larger sigma
    if we think we are stuck in a flat region.
  - We strictly track the number of function calls to stay under the 500 limit.
  - Time guard ensures we finish well before the 330s limit.
"""

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin: 30s buffer
        
        dim = self.dim
        budget = self.budget
        n_evals = 0
        
        # CMA-ES parameters
        # For dim=5, lambda=15 is a reasonable choice.
        # mu = lambda // 2 = 7 (number of parents)
        # weights are standard CMA-ES weights
        lambda_ = 15
        mu = lambda_ // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # CMA-ES adaptation parameters
        # Standard formulas for CMA-ES
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial step size (sigma)
        # The box is [-5, 5]^5, so width is 10.
        # Starting sigma should be large enough to explore but not too large.
        # A common choice is 0.3 * range or similar.
        # Since we don't know the scale of f_opt (can be up to 100), 
        # but the domain is fixed [-5, 5], we start with a moderate sigma.
        sigma = 2.0
        
        # Initial mean
        # Start at the center? Or random?
        # Let's start with a few random initializations to pick a good start.
        # But CMA-ES is sensitive to the start.
        # Let's do a quick random search to find a decent start point.
        
        # Phase 1: Quick random search to find a good initial point
        # Use 20 calls for this.
        init_budget = min(20, budget)
        best_x = np.random.uniform(-5, 5, dim)
        best_f = func(best_x)
        n_evals += 1
        
        for _ in range(init_budget - 1):
            if n_evals >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            fx = func(x)
            n_evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES iterations
        # We will restart if we stagnate.
        # Define stagnation: no improvement in best_f for 'stagnation_limit' generations.
        
        while n_evals < budget and time.time() < deadline:
            # Initialize CMA-ES state from current best
            mean = best_x.copy()
            sigma = 1.5  # Reset sigma for restart, slightly smaller than initial
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Stagnation detection within this CMA-ES run
            local_best_f = best_f
            local_best_x = best_x.copy()
            no_improve_count = 0
            max_no_improve = 10  # Generations without improvement
            
            # Run CMA-ES until stagnation or budget/time limit
            while n_evals < budget and time.time() < deadline:
                # Generate population
                # Cholesky decomposition of C
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    B = np.eye(dim)
                    
                pop = np.empty((lambda_, dim))
                fit = np.empty(lambda_)
                
                for i in range(lambda_):
                    if n_evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Sample from N(mean, sigma^2 * C)
                    z = np.random.randn(dim)
                    x = mean + sigma * (B @ z)
                    x = np.clip(x, -5, 5)
                    
                    fx = func(x)
                    n_evals += 1
                    
                    pop[i] = x
                    fit[i] = fx
                    
                if n_evals < lambda_:
                    # Fill remaining if budget ran out mid-generation
                    # This shouldn't happen if we break correctly, but just in case
                    # We just stop the inner loop
                    break
                
                # Sort by fitness
                order = np.argsort(fit)
                pop_sorted = pop[order]
                fit_sorted = fit[order]
                
                # Update best
                if fit_sorted[0] < local_best_f:
                    local_best_f = fit_sorted[0]
                    local_best_x = pop_sorted[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
                
                # Update mean
                mean_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)
                
                # Update covariance matrix
                # Diff = (pop_sorted[:mu] - mean) / sigma
                diff = (pop_sorted[:mu] - mean) / sigma
                C_new = (1 - c_1 - c_mu) * C + c_1 * ((1 + (1 - c_c) * c_c) * np.outer(pc, pc) - C) + c_mu * np.sum(weights[:, None, None] * np.transpose(diff) * diff, axis=0)
                
                # Update path accumulators
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (mean_new - mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean_new - mean) / sigma
                
                # Update sigma
                sigma *= np.exp(c_sigma / chiN * (np.linalg.norm(ps) / chiN - 1))
                
                # Check for sigma explosion or collapse
                if sigma > 100:
                    sigma = 100
                if sigma < 1e-10:
                    sigma = 1e-10
                    
                # Update state
                mean = mean_new
                C = C_new
                pc = pc
                ps = ps
                
                # Check for stagnation
                if no_improve_count >= max_no_improve:
                    break
                
                # Check for convergence (sigma too small or C too singular)
                if sigma < 1e-6:
                    break
                    
            # If we broke out due to stagnation, we restart the outer loop
            # The outer loop will re-initialize CMA-ES from the new best_x
            # We might want to adjust sigma based on stagnation
            # For now, the outer loop just restarts.
            
        return best_x, best_f
