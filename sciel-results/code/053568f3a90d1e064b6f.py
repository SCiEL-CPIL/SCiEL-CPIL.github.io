"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation.
    
    This is a state-of-the-art black-box optimizer that adapts the covariance matrix
    of the sampling distribution to the shape of the fitness landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        # For dim=5, a population size of 10 is reasonable.
        # This allows ~50 generations with budget 500.
        lambda_ = 10
        mu = lambda_ // 2  # Number of best solutions used for update
        
        # Initial mean: center of the box
        x_mean = np.zeros(self.dim)
        
        # Initial step size: cover a significant portion of the domain
        # Domain is [-5, 5], so width is 10. Sigma=1.0 covers 68% of a 1D dimension.
        # For multi-dim, this is a reasonable start.
        sigma = 1.0
        
        # Initial covariance matrix: identity
        C = np.eye(self.dim)
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Learning rates
        # Standard CMA-ES rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        
        # mu_eff is the effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_cov_pos = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_cov_neg = min(1 - c_cov_pos, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        
        # Inverse square root of C (Cholesky based)
        # We need to sample from N(0, C)
        # y = L @ z, where L @ L.T = C
        # We maintain L such that C = L @ L.T
        
        # Initial L is identity
        L = np.eye(self.dim)
        
        # Paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # We need to generate lambda_ points: x_i = x_mean + sigma * L @ z_i
            # z_i ~ N(0, I)
            
            # To save budget, we might not be able to run a full generation if we are close to limit
            # But lambda_ is small (10), so we can usually do a full generation.
            # If calls_made + lambda_ > budget, we stop.
            if calls_made + lambda_ > self.budget:
                break
                
            z_samples = rng.standard_normal((lambda_, self.dim))
            # Sample points
            # y = L @ z.T -> (dim, lambda_)
            y_samples = z_samples @ L.T  # (lambda_, dim)
            x_samples = x_mean + sigma * y_samples
            
            # Clip to box [-5, 5]
            x_samples = np.clip(x_samples, -5, 5)
            
            # 2. Evaluate fitness
            f_values = np.array([func(x) for x in x_samples])
            calls_made += lambda_
            
            # Update best
            if np.min(f_values) < best_f:
                idx_min = np.argmin(f_values)
                best_f = f_values[idx_min]
                best_x = x_samples[idx_min].copy()
            
            # 3. Sort and select
            # We need the indices that would sort f_values
            sorted_indices = np.argsort(f_values)
            # Take the first mu indices
            selected_indices = sorted_indices[:mu]
            
            # 4. Update mean
            # x_mean_new = sum(w_i * x_selected_i)
            x_mean_new = np.sum(weights[:, np.newaxis] * x_samples[selected_indices], axis=0)
            
            # 5. Update sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * C^-1/2 * (x_mean_new - x_mean) / sigma
            # C^-1/2 is needed. We can approximate or compute.
            # For simplicity and speed, we can use the inverse of L if C = L L^T? No, C^-1/2 is not L^-1.
            # C^-1/2 = (L L^T)^-1/2.
            # Let's compute C^-1/2 using eigen decomposition or Cholesky of C.
            # Since dim is small, we can just invert C and take sqrtm, or use Cholesky.
            # C = L @ L.T.
            # We need C^-1/2.
            # Let's compute eigenvalues and vectors of C.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Handle numerical issues
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Normalize the difference
            diff = x_mean_new - x_mean
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Update sigma
            sigma *= np.exp((1 / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(2 * self.dim) - 1))
            
            # 6. Update covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * C^-1/2 * (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Rank-one update
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2g}) < (1.4 + 2/(dim+1)) * sqrt((mu_eff+2)/mu_eff - 1)
            # We need to track generation counter for this condition, or approximate.
            # For a short run, we can skip the h_sigma check or set it to 1.
            # Let's use a simple heuristic: if p_sigma norm is not huge, h_sigma=1.
            h_sigma = 1.0  # Simplified
            
            # Rank-one update term
            rank_one = np.outer(p_c, p_c) + (h_sigma - 1 + c_c * (2 - c_c)) * np.outer(C_inv_sqrt @ diff, C_inv_sqrt @ diff) / (sigma**2)
            
            # Rank-mu update term
            # Sum of w_i * outer(y_i, y_i)
            y_selected = y_samples[selected_indices]  # (mu, dim)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * y_selected[:, np.newaxis, :] * y_selected[:, :, np.newaxis], axis=0)
            
            # C update
            C = (1 - c_cov_pos - c_cov_neg) * C + c_cov_pos * rank_one + c_cov_neg * rank_mu
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization to prevent singularity
            C += 1e-12 * np.eye(self.dim)
            
            # Update L (Cholesky factor)
            # We need C = L_new @ L_new.T
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity and L to identity
                C = np.eye(self.dim)
                L = np.eye(self.dim)
                p_c = np.zeros(self.dim)
                p_sigma = np.zeros(self.dim)
                
            # Update mean
            x_mean = x_mean_new
            
        return best_x, best_f

# EVOLVE-BLOCK-END
