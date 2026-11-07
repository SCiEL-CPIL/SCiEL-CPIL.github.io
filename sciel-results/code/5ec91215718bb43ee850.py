"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with Multi-Start for BBOB-style f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        # Parameters
        dim = self.dim
        total_budget = self.budget
        
        # We will split the budget among multiple starts.
        # CMA-ES typically needs more evaluations than (1+1)-ES to converge well,
        # but it is more efficient per evaluation. 
        # Let's try 5 starts.
        n_starts = 5
        budget_per_start = max(50, total_budget // n_starts)
        
        best_x = None
        best_f = np.inf
        rng_master = np.random.default_rng(12345)

        for i in range(n_starts):
            if time.time() > deadline:
                break
            
            # Initialize CMA-ES
            # Start points: Randomly chosen within the domain
            # A common strategy is to start near the center or random points.
            # Given the rotation/shift, random points are safer.
            x0 = rng_master.uniform(-5, 5, dim)
            
            # Initial sigma: 
            # For Rastrigin, the scale of local minima is 1 unit.
            # A starting sigma of 1.0 or 2.0 is reasonable.
            sigma = 1.5
            
            # CMA-ES State
            x_mean = x0.copy()
            C = np.eye(dim)
            # Pre-compute eigenvalues for stability? No, let's use standard update.
            # We need to maintain the state for the duration of this start.
            
            # Standard CMA-ES parameters
            mu = max(1, dim // 2)  # Number of parents
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Step size control
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            
            # Covariance matrix adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Chi-normal
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma_history = chi_n
            
            calls_made = 0
            
            while calls_made < budget_per_start:
                if time.time() > deadline:
                    break
                
                # Generate candidates
                # For CMA-ES, we generate mu+lambda points. 
                # To keep it simple and fast in 5D, let's use a smaller lambda.
                # Standard lambda is roughly 4+floor(3*dim). For dim=5, lambda=19.
                # But with limited budget per start, maybe smaller lambda is better.
                # Let's use lambda = 2 * mu.
                lambda_ = 2 * mu
                
                Y = []
                F = []
                
                for j in range(lambda_):
                    # B is sqrt of C. We update B incrementally or recompute?
                    # Recomp is expensive for large dim, but dim=5 is small.
                    # However, doing it every step in a tight loop might be slow in Python.
                    # Alternative: Use a simpler (1, 1)-ES or (mu+1)-ES?
                    # No, CMA-ES is the goal. 
                    # Optimization: We can use the Cholesky decomposition of C.
                    # L = cholesky(C)
                    # z ~ N(0, I)
                    # y = x_mean + sigma * L @ z
                    
                    # To avoid full Cholesky every step, we can use the matrix square root
                    # via eigen decomposition, but that's also costly.
                    # Given dim=5, np.linalg.cholesky is very fast.
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # If C is not positive definite, reset or regularize
                        C = np.eye(dim)
                        L = np.eye(dim)
                        
                    z = rng_master.standard_normal(dim)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    
                    fy = func(y)
                    calls_made += 1
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    
                    Y.append(y)
                    F.append(fy)
                
                if calls_made >= budget_per_start:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                Y = np.array(Y)
                F = np.array(F)
                Y_sorted = Y[idx]
                F_sorted = F[idx]
                
                # Update mean
                x_mean_new = np.sum(w[:, np.newaxis] * Y_sorted[:mu], axis=0)
                
                # Update covariance matrix
                # p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L @ (x_mean_new - x_mean) / sigma)
                
                # h_sigma
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                
                # p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
                
                # C update
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(w[:, np.newaxis, np.newaxis] * np.outer(Y_sorted[:mu] - x_mean, Y_sorted[:mu] - x_mean), axis=0)
                
                C = (1 + c_1 * h_sigma * (1 - d_sigma) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Ensure positive definiteness
                # Add small regularization if needed
                C = (C + C.T) / 2.0
                # If eigenvalues are too small, add identity
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-6:
                     C += 1e-6 * np.eye(dim)
                     
                x_mean = x_mean_new
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Clamp sigma
                if sigma < 1e-10:
                    break
                if sigma > 100:
                    sigma = 100

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
