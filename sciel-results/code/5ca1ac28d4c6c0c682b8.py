"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES implementation for BBOB f14.
    Uses population-based sampling and covariance matrix adaptation 
    to handle the anisotropic landscape of 'Different Powers'.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda is population size
        # For dim=5, lambda ~ 4 + floor(3 * log(5)) ~ 4 + 4 = 8 is reasonable
        # Let's use lambda = 10 for slightly better exploration
        lambda_pop = max(4, 4 + int(3 * np.log(dim)))
        mu = lambda_pop // 2  # number of weights
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / (dim + 2)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        # Start at origin, which is often a good starting point for BBOB
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        calls_used = 0
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls_used += 1
        
        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)
        
        while not check_time():
            # Generate population
            pop_x = []
            pop_f = []
            
            for i in range(lambda_pop):
                # Sample from N(x_mean, sigma^2 * C)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset to identity
                    C = np.eye(dim)
                    L = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                
                z = np.random.randn(dim)
                x_new = x_mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                
                pop_x.append(x_new)
                pop_f.append(func(x_new))
                calls_used += 1
                
                if check_time():
                    break
            
            if check_time():
                break
                
            # Sort by fitness
            indices = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()
                
            # Recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop_x[i]
            x_mean = x_new
            
            # Update covariance matrix adaptation
            # Evolution paths
            inv_sqrt_C = np.linalg.inv(np.linalg.cholesky(C))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (x_new - x_mean))
            norm_ps = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used // lambda_pop)))
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (inv_sqrt_C @ (x_new - x_mean))
            
            # Rank-mu update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (c_c * (2 - c_c) if norm_ps**2 > (1.4 + 0.1 / c_c) * (dim + 1) else 0))
            for i in range(mu):
                y_i = inv_sqrt_C @ (pop_x[i] - x_mean)
                C += c_mu * weights[i] * np.outer(y_i, y_i)
                
            # Step size control
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(mu_eff) - 1))
            
            # Clamp sigma
            sigma = max(1e-10, min(sigma, 10.0))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(dim)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
