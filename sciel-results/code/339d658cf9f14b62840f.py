"""Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with random initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: Evaluator kills at 330s. We aim to finish well before.
        start_time = time.time()
        deadline = start_time + 325.0 # 5s safety margin

        # CMA-ES Parameters
        # Population size lambda
        # Standard recommendation: lambda = 4 + floor(3 * ln(n))
        # For n=5: lambda = 4 + floor(3*1.609) = 4 + 4 = 8
        # Smaller lambda allows more generations. Let's use 10 for better sampling.
        lam = 10
        
        # Initial step size (sigma)
        # Domain is [-5, 5], width 10.
        # Start with a reasonably large step to cover space, but not too large.
        # 0.5 is about 10% of the domain width.
        sigma = 0.5
        
        # Initial mean (center)
        # The problem states minimizer is in [-4, 4]^5.
        # Start at 0.
        x_mean = np.zeros(dim)
        
        # Covariance matrix C (initially identity)
        C = np.eye(dim)
        
        # Learning rates
        # Standard values for CMA-ES
        mu = lam // 2  # Number of parents
        # Weights (logarithmic ranking)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper to check time
        def time_ok():
            return time.time() < deadline

        # 1. Initial Random Sampling
        # Use 10% of budget for random sampling to find a good starting point
        # and potentially identify multiple basins for multimodal functions.
        n_random = max(10, budget // 10)
        n_random = min(n_random, budget - lam) # Ensure we have budget for CMA-ES
        
        best_x = None
        best_fx = float('inf')
        calls_made = 0
        
        # Use a fixed seed for reproducibility in random init? 
        # The task says "hidden seeded shift", implying the function is fixed per instance.
        # Using a fixed seed for our random init is fine, or use system time.
        # Let's use a fixed seed for the random init to be deterministic in the "initial phase"
        # but CMA-ES will use its own RNG. Actually, let's just use default RNG.
        rng = np.random.default_rng()
        
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_made += 1
            
        # Initialize CMA-ES from the best random point
        x_mean = best_x.copy()
        # Reduce sigma if we found a very good point? 
        # Keep sigma = 0.5 for now, CMA-ES will adapt.
        # If the function is very flat, we might want larger sigma.
        # If it's sharp, smaller. 0.5 is a decent start for [-5,5].

        # 2. Main CMA-ES Loop
        # Each generation costs lam evaluations.
        while calls_made + lam <= budget and time_ok():
            
            # Sample population
            pop = np.zeros((lam, dim))
            # Cholesky decomposition of C for sampling
            # To avoid numerical issues, we recompute Cholesky every generation
            # or use a stable method. For small dim, recomputing is fine.
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                B = np.eye(dim)
                
            for i in range(lam):
                z = rng.standard_normal(dim)
                # x_i = x_mean + sigma * B @ z
                pop[i] = x_mean + sigma * (B @ z)
                # Clip to bounds? CMA-ES usually handles bounds via resampling or penalties.
                # The task says func(x) clips input. So we can just pass x.
                # However, sampling outside [-5,5] is wasted if it maps to boundary.
                # Better to clip the sampled points to [-5,5] to stay in domain.
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            fit = np.zeros(lam)
            for i in range(lam):
                if not time_ok():
                    # If time is up, break and return best
                    # We need to fill remaining fits with inf or break loop properly
                    # Since we break, we might have partial fit array.
                    # Let's just break out of the whole while loop if time is up.
                    # But we are inside the eval loop.
                    # Let's set a flag.
                    pass
                
                if calls_made >= budget:
                    break
                    
                fx = func(pop[i])
                fit[i] = fx
                calls_made += 1
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = pop[i].copy()
            
            # If we ran out of budget or time during evaluation, break
            if calls_made >= budget or not time_ok():
                break
            
            # Check if all points were evaluated
            # If we broke early in the eval loop, fit[i] for i >= current are 0.
            # We should only proceed if we evaluated all lam points.
            # The condition `calls_made + lam <= budget` at the start of the while loop
            # ensures we have enough budget for a full generation.
            # So if we didn't break due to time, all are evaluated.
            
            # Sort indices by fitness (ascending)
            idx_sorted = np.argsort(fit)
            
            # Update mean
            # x_mean_new = sum(weights[i] * pop[idx_sorted[i]])
            x_mean_new = np.sum(weights[:, np.newaxis] * pop[idx_sorted], axis=0)
            
            # Update evolution path p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B^T * (x_mean_new - x_mean) / sigma)
            # Note: B is from Cholesky of C. The transformation is B^T.
            # Actually, the standard update uses the inverse of the square root of C.
            # x_mean_new - x_mean = sigma * B @ z_bar
            # So (x_mean_new - x_mean) / sigma = B @ z_bar
            # B^T @ (B @ z_bar) is not quite right for the path in the transformed space.
            # The path p_sigma is in the "z" space (standard normal space).
            # z_bar = mean of z's corresponding to selected parents.
            # Let's compute z_bar directly.
            
            # We need the z vectors for the selected parents.
            # We didn't store them. Let's recompute or store them.
            # Storing is easier. Let's modify the sampling loop to store z.
            
            # ... Wait, I didn't store z. Let's refactor slightly to store z.
            # For now, let's approximate or recompute.
            # Actually, the standard CMA-ES update for p_sigma is:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (1/sigma) * B^{-1} * (x_mean_new - x_mean)
            # Since x = mean + sigma * B @ z, then z = B^{-1} * (x - mean) / sigma.
            # B^{-1} is the inverse of the Cholesky factor.
            
            try:
                B_inv = np.linalg.inv(B)
                z_bar = np.sum(weights[:, np.newaxis] * (B_inv @ (pop[idx_sorted] - x_mean).T).T / sigma, axis=0)
            except np.linalg.LinAlgError:
                z_bar = np.zeros(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_bar
            
            # Hessian correction factor
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lam + 1)))) / np.sqrt(mu_eff) - 1
            # If denominator is 0 or small, h_sigma might be unstable.
            # Standard check: if norm(p_sigma) < (1.4 + 1/dim) * sqrt((mu_eff+2)/(mu_eff-1)) * sqrt(1 - (1-c_sigma)^(2*gen))
            # Let's use a simpler threshold to avoid division by zero issues early on.
            gen = calls_made // lam
            if gen > 0:
                threshold = (1.4 + 1/dim) * np.sqrt((mu_eff + 2) / (mu_eff - 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * gen))
                if np.linalg.norm(p_sigma) < threshold:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
            else:
                h_sigma = 0.0
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update evolution path p_c
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (1/sigma) * B^{-1} * (x_mean_new - x_mean)
            # Note: (1/sigma) * B^{-1} * (x_mean_new - x_mean) is exactly z_bar.
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_bar
            
            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C)
            #    + c_mu * sum(weights[i] * (y_i @ y_i))
            # where y_i = (1/sigma) * B^{-1} * (x_i - x_mean_old)
            
            # Calculate y_i for selected parents
            Y = np.zeros((mu, dim))
            for i in range(mu):
                x_i = pop[idx_sorted[i]]
                try:
                    Y[i] = (B_inv @ (x_i - x_mean).T).T / sigma
                except:
                    Y[i] = np.zeros(dim)
                    
            # Rank-1 update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            # sum(weights[i] * outer(Y[i], Y[i]))
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(Y[i], Y[i])
                
            C += c_mu * rank_mu
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2.0
            
            # Update mean
            x_mean = x_mean_new
            
            # Check for convergence
            # If sigma is too small or C is singular, break
            if sigma < 1e-12:
                break
            if np.linalg.cond(C) > 1e12:
                # Reset C if it becomes ill-conditioned
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)
                sigma = 0.5 # Reset to reasonable value? Or keep current?
                # Resetting sigma might be too aggressive. Let's just reset C.
                # If C is singular, sampling fails.
                pass

        # Final Polish?
        # CMA-ES is generally good at polishing. 
        # If we have budget left, we can do a few more iterations with smaller sigma?
        # Or just return best.
        # Given the budget constraints and the robustness of CMA-ES, returning the best found is usually sufficient.
        # The random init phase helps with multimodality.
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
