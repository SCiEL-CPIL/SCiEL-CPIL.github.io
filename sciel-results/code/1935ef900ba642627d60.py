"""Rewritten program: CMA-ES optimizer for BBOB f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES continuous optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        # Standard formula: lambda = 4 + floor(3 * log(n + log(n)))
        # For n=5: 4 + floor(3 * log(5 + 1.6)) = 4 + floor(3 * 1.89) = 4 + 5 = 9? 
        # Let's use a slightly larger population for better exploration in multimodal landscape.
        # Lambda = 10 is a good balance for n=5 and budget=500.
        lambda_pop = 10
        
        # Initial step size
        # The box is [-5, 5]. The minimum is in [-4, 4].
        # Starting sigma should be large enough to explore the box but not too large to waste calls.
        # 1.0 or 2.0 is reasonable. Let's start with 2.0.
        sigma = 2.0
        
        # Initial mean
        # Uniform random start within the box, or center?
        # Center (0,0,0) is often good if the shift is small, but the problem says shift is in [-4,4].
        # A random start is safer.
        rng = np.random.default_rng(42)
        mean = rng.uniform(-4.0, 4.0, size=n)
        
        # Covariance matrix
        C = np.eye(n)
        # Inverse square root of C (for sampling)
        C_sqrt_inv = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
        mu = lambda_pop // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_sigma = c_sigma * (1 + 2 * max(0, sqrt(mu_eff / n - 1) - 1) + c_c)
        c_sigma = min(c_sigma, 0.5) # Cap to avoid instability
        
        c_cov_pos = (2 + 2 * mu_eff / n) / (n + 2 + mu_eff / n)
        c_cov_neg = (2 + 2 * mu_eff / n) / (n + 2 + 2 * mu_eff / n)
        
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / n - 1)) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # Call counter
        calls_made = 0
        
        # Main Loop
        while calls_made < budget:
            # Generate population
            # We need to sample lambda_pop points
            # But we must respect the budget.
            # If remaining budget < lambda_pop, we can only evaluate the rest.
            # However, CMA-ES works best with full generations.
            # If remaining < lambda_pop, we can do a final partial generation or just stop.
            # Given 500 budget and lambda=10, we will have 50 full generations.
            
            remaining = budget - calls_made
            if remaining <= 0:
                break
            
            # Determine how many to evaluate in this generation
            # If we have enough for a full generation, do it.
            # Otherwise, evaluate as many as possible.
            num_evals = min(lambda_pop, remaining)
            
            # Sample num_evals points
            # We need to update C_sqrt_inv if C changes, but we update C after the generation.
            # So we sample using the current C_sqrt_inv.
            
            # Note: Standard CMA-ES samples z ~ N(0, I), x = mean + sigma * C_sqrt_inv @ z
            # We need to be careful with the inverse square root.
            
            # For simplicity and stability in a short budget, we can use a simplified CMA-ES
            # or ensure the matrix operations are correct.
            
            # Let's stick to the standard update rules.
            
            # Sample
            z = rng.standard_normal((num_evals, n))
            # x = mean + sigma * (C_sqrt_inv @ z.T).T
            # This requires C_sqrt_inv to be the inverse square root of C.
            # Initially C=I, so C_sqrt_inv=I.
            
            pop = mean + sigma * (z @ C_sqrt_inv.T)
            
            # Clip to bounds? 
            # The problem says func(x) clips. It's better to let func clip to respect the black-box nature,
            # but for the optimizer's internal state, we should probably keep the mean within bounds
            # or allow it to go outside and let the function handle it.
            # Since func clips, evaluating outside is fine, it just returns the value at the boundary.
            # However, if we sample far outside, we waste calls.
            # A common trick is to reflect or clip the samples before evaluation to keep the search focused.
            # But CMA-ES adapts sigma. If we clip, the distribution gets distorted.
            # Given the box is [-5, 5] and we start in [-4, 4] with sigma=2,
            # most samples will be inside. We'll let func clip.
            
            # Evaluate
            fitness = np.array([func(pop[i]) for i in range(num_evals)])
            calls_made += num_evals
            
            # Sort indices by fitness
            # If num_evals < lambda_pop, we only have partial info.
            # In a full generation, we sort all lambda.
            # If partial, we sort the available ones.
            # For CMA-ES update, we typically need the top mu.
            # If num_evals < mu, we can't do a proper update.
            # But with budget 500 and lambda 10, we always have full generations.
            
            if num_evals < mu:
                 # Not enough data for a standard CMA-ES update.
                 # Just update mean to the best point found so far?
                 # Or just stop?
                 # Let's just update mean to the best of this partial set and break if we can't do a full gen.
                 # Actually, if we can't do a full gen, we should probably just evaluate the best remaining points
                 # or stop. Given the constraint, we'll assume full generations fit.
                 pass

            # Sort
            idx = np.argsort(fitness)
            # Take top mu
            idx_mu = idx[:mu]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * pop[idx_mu], axis=0)
            
            # Update covariance matrix
            # y = (new_mean - mean) / sigma
            y = (new_mean - mean) / sigma
            
            # Update p_sigma
            h_sigma = (np.linalg.norm(p_sigma / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (n + 1)) * chi_n)
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            
            # Update p_c
            h_c = (np.linalg.norm(p_c / np.sqrt(1 - (1 - c_c)**2)) < (1.4 + 2 / (n + 1)) * chi_n)
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Update C
            # C = (1 - c_cov_pos - c_cov_neg) * C + c_cov_pos * (p_c * p_c.T + (1 - h_c) * c_c * (2 - c_c) * C) 
            #     + c_cov_neg * sum(weights[i] * (x_i - mean) * (x_i - mean).T)
            
            # Let's use the standard rank-mu update
            # delta_h = 1 - h_c
            # C = (1 - c_cov_pos - c_cov_neg) * C + c_cov_pos * (p_c * p_c.T + delta_h * c_c * (2 - c_c) * C)
            #    + c_cov_neg * sum(weights[i] * y_i * y_i.T) where y_i = (x_i - mean)/sigma
            
            # Wait, the standard formula uses the standardized vectors.
            # Let's compute the rank-mu update term.
            diff = (pop[idx_mu] - mean) / sigma
            # diff is (mu, n)
            rank_mu_term = np.sum(weights[:, np.newaxis, np.newaxis] * (diff[:, np.newaxis, :] * diff[:, np.newaxis, :].T), axis=0)
            
            delta_h = 1 - h_c
            C = (1 - c_cov_pos - c_cov_neg) * C + c_cov_pos * (np.outer(p_c, p_c) + delta_h * c_c * (2 - c_c) * C) + c_cov_neg * rank_mu_term
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Update C_sqrt_inv
            # We need the inverse square root of C.
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid negative/zero issues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Check for convergence or stagnation
            # If sigma is too small, we might stop.
            if sigma < 1e-8:
                break
                
        return mean, fitness[idx[0]] if num_evals > 0 else float('inf')
# EVOLVE-BLOCK-END
