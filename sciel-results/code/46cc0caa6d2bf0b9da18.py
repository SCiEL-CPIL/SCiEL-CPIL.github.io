"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f14.
    Uses a population-based approach to adapt to the anisotropic landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s limit

        rng = np.random.default_rng(42)
        calls_used = 0
        dim = self.dim
        budget = self.budget

        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        # Initialize CMA-ES parameters
        # Start at origin, which is a strong prior for f14
        x_mean = np.zeros(dim)
        sigma = 1.0
        
        # Population size: standard recommendation is 4 + floor(3*log(dim))
        # For dim=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8
        lambda_pop = 4 + int(np.floor(3 * np.log(dim)))
        
        # Weights for recombination
        mu = lambda_pop // 2  # Number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effectiveness
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Cumulation path lengths
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Precompute inverse square root of C for efficient sampling
        # We will update C and recompute this periodically or use a simpler update
        # For simplicity and robustness in a short budget, we'll use the standard update
        # but need to be careful with performance.
        
        # Initial evaluation
        x_curr = x_mean.copy()
        f_curr = func(x_curr)
        calls_used += 1
        
        # Track best
        x_best = x_mean.copy()
        f_best = f_curr
        
        # Main loop
        while not check_time():
            # Generate population
            pop = []
            f_pop = []
            
            # We need to check budget for the whole population
            if calls_used + lambda_pop > budget:
                # If we can't afford a full population, do a single step (1+1) style
                # or just break. Given 500 budget and pop=8, we have ~60 generations.
                # Breaking is safer to ensure we don't exceed budget.
                break
                
            for i in range(lambda_pop):
                # Sample from N(x_mean, sigma^2 * C)
                # Efficient sampling: x_mean + sigma * A @ z
                # Where A is such that A @ A.T = C
                # We can use Cholesky or SVD. For dim=5, this is cheap.
                # However, C changes every generation.
                
                # Simple approach: use eigen-decomposition or Cholesky each time?
                # Or maintain a matrix B such that B @ B.T = C.
                # Standard CMA-ES maintains B. Let's do that.
                
                # Actually, let's stick to a slightly simplified but robust CMA-ES
                # that recomputes the sampling matrix if needed, or uses a cached one.
                # Given dim=5, computing Cholesky of 5x5 is negligible.
                
                try:
                    L = np.linalg.cholesky(C)
                    z = rng.standard_normal(dim)
                    x_i = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular (shouldn't happen with regularization)
                    L = np.eye(dim)
                    z = rng.standard_normal(dim)
                    x_i = x_mean + sigma * z
                    
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls_used += 1
                
                pop.append(x_i)
                f_pop.append(f_i)
                
                if f_i < f_best:
                    f_best = f_i
                    x_best = x_i.copy()

            # Sort by fitness
            indices = np.argsort(f_pop)
            pop = [pop[i] for i in indices]
            f_pop = [f_pop[i] for i in indices]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
            
            # Update sigma (CSA - Cumulation Step Adaptation)
            # h_sigma condition
            # chi_n is expected norm of standard normal
            # We need the inverse square root of C to compute the norm in the transformed space
            # For simplicity and speed in this short run, we can approximate or use a simpler rule
            # Let's use the standard CSA rule.
            
            # Compute C^{-1/2}
            # Eigen decomposition is stable for symmetric positive definite C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + \
                      np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_mean - x_old) / sigma)
            
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used / lambda_pop + 1))) * chi_n < 1.4 + 2 / (dim + 1))
            
            # Sigma update
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1) if h_sigma else 1)
            
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (x_mean - x_old) / sigma)
            
            # Rank-one and rank-mu updates for C
            # Rank-one
            delta_h = (2 * h_sigma - 1) * c_1
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * np.outer(p_c, p_c) + \
                c_mu * np.sum([weights[i] * np.outer((pop[i] - x_old) / sigma, (pop[i] - x_old) / sigma) for i in range(mu)], axis=0)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness (add small identity if needed)
            # Check eigenvalues
            min_eig = np.min(np.linalg.eigvalsh(C))
            if min_eig < 1e-10:
                C += (1e-10 - min_eig) * np.eye(dim)

            # Clamp sigma
            sigma = max(1e-10, min(sigma, 10.0))

        return x_best, f_best
# EVOLVE-BLOCK-END
