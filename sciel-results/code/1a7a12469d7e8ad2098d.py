"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES optimizer for BBOB f14.
    Strategy:
    1. Use a small population (size 4) to maintain diversity while being efficient.
    2. Adapt the covariance matrix and step size to the landscape geometry.
    3. Start from the origin, leveraging the known locality of f14's minimum 
       (often near 0 or slightly shifted) while allowing the CMA-ES to 
       correct for hidden rotations and shifts.
    4. Strict budget management to ensure exactly 500 calls are used if possible, 
       maximizing the AOCC score which rewards early progress.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Small population size for efficiency in 5D with 500 budget
        # lambda = 4 is small, but helps avoid stagnation compared to (1+1)
        # and is much faster than standard lambda ~ 4+3*ln(n) which would be ~15
        # We choose lambda=4 to allow for more generations (500/4 = 125 gens)
        lambda_ = 4
        
        # Initial step size
        # f14 is typically well-scaled, but hidden shifts might require larger steps initially.
        # 1.0 is a reasonable start for [-5,5] box.
        sigma = 1.0
        
        # Initial mean
        # The problem states the minimizer is in [-4, 4]^5. 
        # Starting at 0 is a strong prior for BBOB functions.
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        Cinv_sqrt = np.eye(dim)
        
        # Weights for ranking
        # Standard weights for CMA-ES
        weights = np.zeros(lambda_)
        # Use a simple weight scheme: w_i = ln((lambda+1)/2) - ln(i) for i=1..lambda/2
        # Then symmetric.
        # For lambda=4:
        # i=1: ln(2.5) - ln(1) = 0.916
        # i=2: ln(2.5) - ln(2) = 0.223
        # w = [0.916, 0.223, 0.223, 0.916] normalized
        # Actually, let's use a simpler robust set for small lambda.
        # w = [1, 0.5, 0.5, 1] -> normalized? No, standard is better.
        # Let's compute standard weights.
        for i in range(1, lambda_ + 1):
            if i <= lambda_ / 2:
                weights[i-1] = np.log((lambda_ + 1) / 2) - np.log(i)
            else:
                weights[i-1] = weights[lambda_ - i] # Symmetry
        
        # Normalize weights
        weights /= weights.sum()
        
        # mu = number of individuals used for mean update
        mu = lambda_ // 2
        
        # nu = effective population size
        mu_eff = 1 / np.sum(weights[:mu]**2)
        
        # Learning rates
        # c_sigma: cumulative step-size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # d_sigma: damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # c_c: evolution path for covariance
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        # c_1: rank-1 update
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        # c_mu: rank-mu update
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Chi_lambda: expected norm of standard normal
        chi_lambda = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        calls_used = 1
        
        # Main Loop
        # We have budget calls. 
        # Each generation uses lambda_ calls.
        # Total generations = budget // lambda_
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            # 1. Sample lambda_ individuals
            # Generate standard normal matrix
            Z = rng.standard_normal((lambda_, dim))
            
            # Transform to x = mean + sigma * Z @ Cinv_sqrt
            # Note: Cinv_sqrt is the matrix square root of the inverse covariance.
            # In standard CMA-ES, we sample z ~ N(0, I), then x = mean + sigma * Cinv_sqrt @ z
            # Wait, the standard update is:
            # x_i = mean + sigma * (Cinv_sqrt @ z_i)
            # Where Cinv_sqrt is updated such that C = (Cinv_sqrt)^T @ Cinv_sqrt
            # Actually, usually we store C and use Cholesky or eigen decomposition.
            # For simplicity and speed in 5D, we can store C and compute eigendecomposition each step?
            # Or use the BFGS-like update?
            # Given the small dimension (5) and small population (4), eigendecomposition is cheap.
            
            # Let's switch to a slightly simpler but robust implementation:
            # We will maintain C and update it.
            # To sample, we need the square root of C.
            # Let's use eigen decomposition for sampling.
            
            # However, doing eigendecomposition every generation might be slower than necessary.
            # Alternative: Use the "CMA-ES with fixed covariance" or simpler "Nelder-Mead"?
            # No, CMA-ES is required by the prompt's inspiration.
            
            # Let's optimize the CMA-ES implementation for speed.
            # Instead of full CMA-ES with matrix updates, we can use a simplified version
            # or ensure the matrix operations are vectorized.
            
            # Actually, for dim=5, even numpy.linalg.eigh is fast.
            # Let's proceed with standard CMA-ES logic.
            
            # Sample
            # We need Cinv_sqrt.
            # Let's maintain C and compute eigendecomposition each step.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-12)
            Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # x_i = mean + sigma * Cinv_sqrt @ z_i
            X = mean[None, :] + sigma * (Z @ Cinv_sqrt.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_used += lambda_
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
            
            # 2. Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # 3. Compute weighted mean
            y = X[:mu]
            new_mean = np.sum(weights[:mu, None] * y, axis=0)
            
            # 4. Update step size
            # p_sigma = (1-c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * Cinv_sqrt @ (new_mean - mean)/sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (Cinv_sqrt @ (new_mean - mean) / sigma)
            
            h_sigma = 1.0
            # h_sigma condition: ||p_sigma|| / (1-(1-c_sigma)^(2*gen)) < (1+2/sqrt(dim+2)) * chi_lambda
            # We don't track gen count for the power term easily here without a counter, 
            # but for a short run, we can approximate or skip the h_sigma check if it's complex.
            # Let's include a simple check.
            # Actually, let's just update sigma if p_sigma is not too large.
            # Standard condition:
            # if ||p_sigma|| / chi_lambda < (1 + 2/sqrt(dim+2)) * (1 - (1-c_sigma)**(2*gen))
            # Since we don't have gen, let's use a simpler heuristic or just always update?
            # No, that can be unstable.
            # Let's track generation count.
            
            # Let's restart the loop logic to be cleaner.
            break # Break out to rewrite the loop properly with gen counter

        # The above loop structure was getting messy. Let's write a clean CMA-ES loop.
        
        # Reset state for clean implementation
        mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = mean.copy()
        best_f = func(best_x)
        calls_used = 1
        gen = 0
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
                
            gen += 1
            
            # Sample
            # Compute Cinv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            Z = rng.standard_normal((lambda_, dim))
            X = mean[None, :] + sigma * (Z @ Cinv_sqrt.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_used += lambda_
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
                
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # New Mean
            new_mean = np.sum(weights[:mu, None] * X[:mu], axis=0)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (Cinv_sqrt @ (new_mean - mean) / sigma)
            
            # h_sigma check
            # Approximation: if ||p_sigma|| is small relative to chi_lambda, update sigma
            # The factor (1 - (1-c_sigma)**(2*gen)) approaches 1 quickly.
            # Let's use the standard condition.
            factor = 1 - (1 - c_sigma) ** (2 * gen)
            if np.linalg.norm(p_sigma) / chi_lambda < (1 + 2 / np.sqrt(dim + 2)) * factor:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * Cinv_sqrt @ (new_mean - mean) / sigma
            
            # Update C
            # Rank-1 and Rank-mu updates
            # v = (new_mean - mean) / sigma
            v = (new_mean - mean) / sigma
            
            # Rank-1
            C = (1 - c_c * (2 - c_c) * (1 - h_sigma) - c_1 - c_mu) * C \
                + c_c * (2 - c_c) * (1 - h_sigma) * np.outer(p_c, p_c) \
                + c_1 * (np.outer(v, v) + (1 - h_sigma) * C) \
                + c_mu * np.sum(weights[:mu, None, None] * np.einsum('ij,ik->ijk', v, v), axis=0)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                # Reset C to identity if it becomes singular
                C = np.eye(dim)
            else:
                # Clip small eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            mean = new_mean
            
        return best_x, best_f
# EVOLVE-BLOCK-END
