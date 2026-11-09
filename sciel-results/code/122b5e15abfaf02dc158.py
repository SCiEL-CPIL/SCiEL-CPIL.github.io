"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with periodicity-aware restart mechanism for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        
        # CMA-ES Parameters
        dim = self.dim
        # Initial sigma: 2.0 is a good start for [-5, 5] box
        sigma = 2.0 
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Covariance matrix initialization: Identity
        C = np.eye(dim)
        # Eigenvalue decomposition for sqrt(C)
        evals, evecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues (numerical stability)
        evals = np.maximum(evals, 1e-12)
        sqrtC = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        
        # Step size control parameters (standard CMA-ES)
        mu = dim // 2 + 1  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / dim) / ((dim + 2)**2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Cumulation path
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (c_sigma + 1) / (dim + c_sigma + 3)
        hs = (1 - np.exp(-1 / (dim * cs))) ** 0.5
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Best solution tracking
        best_x = mean.copy()
        best_f = func(best_x)
        
        # Stagnation tracking
        last_improvement_call = 0
        stagnation_threshold = 50 # Number of function evaluations without improvement
        
        # Budget management
        calls_made = 1
        lambda_ = 2 * mu # Population size
        
        # Initial population generation
        # We need to generate lambda individuals
        # To save time, we will generate them in batches or one by one
        # Given budget 500 and dim 5, lambda=6 or 8 is efficient.
        # Let's use lambda = 2 * mu = 2 * (5//2 + 1) = 6 for dim=5.
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Standard CMA-ES generates lambda = 2*mu individuals
            # For dim=5, mu=3, lambda=6
            # This is very fast.
            
            # Check for stagnation and restart
            if calls_made - last_improvement_call > stagnation_threshold and calls_made > 100:
                # Restart mechanism:
                # Re-center mean at best_x
                mean = best_x.copy()
                # Reset sigma to a larger value to escape local minima
                # Rastrigin has periodicity, so a larger step might jump to another basin
                sigma = 1.5 
                # Reset covariance to identity (forget rotation, assume new landscape orientation or isotropic exploration)
                C = np.eye(dim)
                evals, evecs = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-12)
                sqrtC = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                last_improvement_call = calls_made
                # Optionally, we could also try a random jump, but re-centering with larger sigma is safer
                # Let's also add a small random perturbation to the mean to break symmetry if stuck in a periodic trap
                mean += 0.5 * np.random.default_rng(calls_made).standard_normal(dim)
                mean = np.clip(mean, -5, 5)
            
            # Generate lambda offspring
            # We will generate them one by one to respect the call budget strictly and allow early exit if time runs out
            # But CMA-ES usually requires a full generation. 
            # With budget 500 and lambda 6, we can do ~80 generations.
            # This is sufficient for convergence in 5D.
            
            # To optimize for AOCC (early progress), we should ensure we are making steady progress.
            # If we are stuck, the restart helps.
            
            # Generate lambda points
            z = np.random.default_rng(calls_made).standard_normal((lambda_, dim))
            # Apply transformation: x = mean + sigma * sqrtC @ z
            # Note: sqrtC @ z is equivalent to z @ sqrtC.T, but we want matrix multiplication
            # z is (lambda, dim). sqrtC is (dim, dim).
            # We need z @ sqrtC.T
            offspring = mean[None, :] + sigma * (z @ sqrtC.T)
            offspring = np.clip(offspring, -5, 5)
            
            # Evaluate offspring
            # We must be careful not to exceed budget
            current_lambda = min(lambda_, self.budget - calls_made)
            
            if current_lambda <= 0:
                break
                
            # Evaluate only the first current_lambda offspring
            f_offspring = np.empty(current_lambda)
            for i in range(current_lambda):
                f_offspring[i] = func(offspring[i])
                calls_made += 1
                
                # Update best
                if f_offspring[i] < best_f:
                    best_f = f_offspring[i]
                    best_x = offspring[i].copy()
                    last_improvement_call = calls_made
                    
                if time.time() > deadline:
                    break
            
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # CMA-ES Update Step
            # Sort offspring by fitness
            # We only evaluated current_lambda points. If current_lambda < lambda_, we can't do a full standard CMA-ES update properly with the predefined weights.
            # However, for the first few iterations, if budget allows, current_lambda == lambda_.
            # If budget runs out mid-generation, we just stop.
            # For simplicity and robustness, if we don't have a full generation, we skip the update and just continue sampling from the current distribution? 
            # No, better to just break if we can't complete a generation, as partial updates are complex.
            # Given budget 500 and lambda 6, we will have full generations for most of the run.
            
            if current_lambda < lambda_:
                # We have partial data. Let's just use the best found so far and maybe do a simple (1+1) step for the rest of the budget?
                # Or just stop. Stopping is safer.
                break
            
            # Sort indices
            sorted_idx = np.argsort(f_offspring)
            # Select top mu
            selected_idx = sorted_idx[:mu]
            selected_offspring = offspring[selected_idx]
            selected_fitness = f_offspring[selected_idx]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * selected_offspring, axis=0)
            
            # Update covariance matrix
            # Difference matrix
            diff = (selected_offspring - mean[None, :]) / sigma
            diff = diff @ sqrtC.T # Transform back to standard normal space? 
            # Actually, the update rule uses the differences in the search space transformed by C^-1/2.
            # x_i = m + sigma * C^1/2 * z_i
            # z_i = C^-1/2 * (x_i - m) / sigma
            # So diff is the z-vector.
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - (1 - c_sigma)^2) * C) + c_mu * sum(weights_i * z_i * z_i.T)
            # Wait, the standard formula is:
            # C <- (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1 - c_sigma)^2) C) + c_mu sum(w_i y_i y_i^T)
            # where y_i = C^{-1/2} (x_i - m) / sigma
            
            # Let's compute y_i
            # y_i = sqrtC_inv @ (x_i - m) / sigma
            # Since sqrtC = evecs @ diag(sqrt(evals)) @ evecs.T
            # sqrtC_inv = evecs @ diag(1/sqrt(evals)) @ evecs.T
            
            inv_sqrtC = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            y = (selected_offspring - mean[None, :]) / sigma
            y = y @ inv_sqrtC.T
            
            # Update pc
            # pc = (1 - c_sigma) * pc + hs * c_sigma * sqrt(mu_eff) * sum(weights_i * y_i)
            pc = (1 - c_sigma) * pc + hs * c_sigma * np.sqrt(mu_eff) * np.sum(weights[:, None] * y, axis=0)
            
            # Update ps
            # ps = (1 - c_sigma) * ps + hs * sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (new_mean - mean) / sigma
            # Wait, ps is in the original space? No, ps is usually in the standard normal space?
            # Let's check Hansen's CMA-ES paper.
            # ps is the evolution path in the standard normal space.
            # ps <- (1-c_sigma) ps + hs * sqrt(c_sigma(2-c_sigma) mu_eff) * (new_mean - mean) / sigma * C^{-1/2}
            # Actually, the update for ps is:
            # ps <- (1-c_sigma) ps + hs * sqrt(c_sigma(2-c_sigma) mu_eff) * C^{-1/2} (new_mean - mean) / sigma
            
            ps = (1 - c_sigma) * ps + hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrtC @ (new_mean - mean) / sigma)
            
            # Update C
            # Term 1: (1 - c_1 - c_mu) * C
            # Term 2: c_1 * (pc @ pc.T + (1 - (1 - c_sigma)**2) * C)
            # Term 3: c_mu * sum(weights_i * y_i @ y_i.T)
            
            term1 = (1 - c_1 - c_mu) * C
            term2 = c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C)
            
            # Compute sum of weighted outer products
            # y is (mu, dim)
            # weights is (mu,)
            # We want sum(weights_i * y_i * y_i.T)
            # This is y.T @ diag(weights) @ y
            term3 = c_mu * (y.T @ (weights[:, None] * y))
            
            C = term1 + term2 + term3
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Update sigma
            # sigma = sigma * exp(c_sigma/d_sigma * (||ps||/chi_n - 1))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Prevent sigma from becoming too large or small
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update sqrtC and inv_sqrtC
            # We need to recompute eigen decomposition
            # This is the expensive part.
            # For dim=5, it's very fast.
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-12)
            sqrtC = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            inv_sqrtC = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            # Update best
            # Already done in the loop above
            
        return best_x, best_f
# EVOLVE-BLOCK-END
