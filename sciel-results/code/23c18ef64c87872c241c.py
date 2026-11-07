"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard CMA-ES lambda is ceil(4 + log(10*dim))
        # For dim=5, lambda = ceil(4 + log(50)) ≈ 7
        # We use a slightly larger population for better exploration initially
        lambda_pop = 7
        
        # Initial mean and sigma
        # Start near center, but f12 is shifted/rotated. 
        # A uniform start is safer, but CMA-ES adapts quickly.
        # However, since we have a fixed budget of 500, we want to be efficient.
        # Let's start at 0 (center of box) as a reasonable prior for BBOB instances.
        mean = np.zeros(dim)
        
        # Initial sigma: 2.0 is standard for box [-5,5]
        sigma = 2.0
        
        # Covariance matrix C = I
        C = np.eye(dim)
        
        # Eigendecomposition cache
        # C = B * D * B.T
        B, D = np.linalg.eigh(C)
        inv_D = 1.0 / np.sqrt(D)
        D_inv_sqrt = 1.0 / np.sqrt(D)
        
        # Step size control (CSA)
        # c_sigma = (mu_eff + 2) / (dim + 4 + mu_eff)
        # c_c = (4 + mu_eff / dim) / (dim + 4 + 3 * mu_eff)
        # mu_eff = sum of weights
        # Weights: standard CMA-ES weights
        # w_i ∝ log((lambda+1)/2) - log(i)
        # mu = largest index such that sum_{i=1}^mu w_i > 0.5
        
        # Calculate weights
        mu = lambda_pop // 2
        # Standard weights formula
        w = np.zeros(lambda_pop)
        for i in range(mu):
            w[i] = np.log((lambda_pop + 1) / 2.0) - np.log(i + 1)
        w_sum = w.sum()
        w /= w_sum
        mu_eff = 1.0 / np.sum(w**2)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + 4 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 3 * mu_eff)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        x = np.clip(mean, -5, 5)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Loop for remaining budget
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            # y = mean + sigma * B * (D_inv_sqrt * z)
            # z is standard normal
            
            # Check for stagnation or time, but mainly just run CMA-ES
            # If sigma becomes too small, we might stop early, but let's try to use budget
            
            # Sample
            z = np.random.randn(lambda_pop, dim)
            # Transform
            # y_i = mean + sigma * B * (D_inv_sqrt * z_i)
            # Vectorized:
            # z_scaled = z * D_inv_sqrt[None, :]
            # y = mean[None, :] + sigma * (z_scaled @ B.T)
            
            z_scaled = z * D_inv_sqrt[None, :]
            y = mean[None, :] + sigma * (z_scaled @ B.T)
            
            # Clip to box
            y = np.clip(y, -5, 5)
            
            # Evaluate
            f_y = np.array([func(yi) for yi in y])
            calls += lambda_pop
            
            # Sort by fitness
            order = np.argsort(f_y)
            y_sorted = y[order]
            f_sorted = f_y[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
                
            # Update mean
            # mu-weighted mean of top mu solutions
            new_mean = np.sum(w[:mu, None] * y_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} * (y_new - y_old) / sigma
            # C^{-1/2} * v = B * (D_inv_sqrt * (B.T * v))
            # v = (new_mean - mean) / sigma
            v = (new_mean - mean) / sigma
            C_inv_sqrt_v = B @ (D_inv_sqrt * (B.T @ v))
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_v
            
            # h_sigma: check if p_sigma is consistent with noise
            # chi_n = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2/(dim+1)) * chi_n else 0.0
            
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * C^{-1/2} * (new_mean - mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt_v
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / lambda_sigma * (||p_sigma|| / chi_n - 1))
            # lambda_sigma = (mu_eff + 2) / (dim + 4 + mu_eff) is actually c_sigma? No.
            # In CMA-ES, lambda_sigma is usually c_sigma.
            # Wait, standard formula: sigma = sigma * exp( c_sigma/lambda_sigma * ( ||p_sigma||/chi_n - 1 ) )
            # lambda_sigma is often set to c_sigma.
            # Let's use c_sigma for both to be safe, or standard values.
            # Actually, standard CMA-ES uses c_sigma for the rate.
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i y_i^T)
            
            # Term 1: (1 - c_1 - c_mu) * C
            C_new = (1 - c_1 - c_mu) * C
            
            # Term 2: c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C)
            C_new += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Term 3: c_mu * sum(w_i * (y_i - mean) (y_i - mean)^T)
            # y_i - mean = sigma * B * (D_inv_sqrt * z_i)
            # Let d_i = D_inv_sqrt * z_i
            # y_i - mean = sigma * B * d_i
            # (y_i - mean)(y_i - mean)^T = sigma^2 * B * d_i d_i^T * B^T
            # Sum w_i * ... = sigma^2 * B * (sum w_i d_i d_i^T) * B^T
            
            # We need sum w_i d_i d_i^T for the top mu
            # d_i = D_inv_sqrt * z_i_sorted
            # z_sorted = z[order]
            # d_sorted = z_sorted * D_inv_sqrt[None, :]
            
            # Only top mu
            d_top = z_sorted[:mu] * D_inv_sqrt[None, :]
            
            # Accumulate H = sum w_i d_i d_i^T
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(d_top[i], d_top[i])
                
            C_new += c_mu * (sigma**2) * (B @ H @ B.T)
            
            # Update mean
            mean = new_mean
            
            # Update C
            C = C_new
            
            # Update eigendecomposition
            # C should be symmetric positive definite
            C = (C + C.T) / 2.0
            B, D = np.linalg.eigh(C)
            # Ensure D is positive
            D = np.maximum(D, 1e-16)
            D_inv_sqrt = 1.0 / np.sqrt(D)
            
            # If sigma is too small, we might break, but let's continue
            if sigma < 1e-8:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
