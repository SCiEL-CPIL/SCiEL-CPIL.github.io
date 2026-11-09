"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f14.
    Uses a population-based approach with covariance adaptation to handle
    rotations and scaling inherent in the 'Different powers' function.
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

        # CMA-ES Parameters
        # Population size lambda. For dim=5, lambda=4+floor(3*log(5)) = 4+4=8 is standard, 
        # but we want to conserve budget. Let's use a smaller population to allow more generations.
        # With budget 500, if lambda=4, we get ~125 generations. If lambda=8, ~62 generations.
        # CMA-ES converges faster per generation. Let's try lambda=4 for max generations.
        lambda_pop = 4
        mu = lambda_pop // 2  # Number of parents for recombination
        
        # Initial step size
        sigma = 1.0
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Initial covariance matrix
        C = np.eye(dim)
        
        # Weights (standard weights for CMA-ES)
        weights = np.ones(mu)
        weights /= np.sum(weights)
        # Standard weights formula: w_i = log(mu/2 + 1) - log(i)
        # But for small mu, simple uniform or slightly biased is okay.
        # Let's use the standard formula for robustness.
        for i in range(mu):
            weights[i] = np.log(mu / 2 + 1) - np.log(i + 1)
        weights /= np.sum(weights)
        
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Cholesky factor of C
        D, B = np.linalg.cholesky(C)
        B_inv = np.linalg.inv(B)
        D_inv = np.diag(1.0 / np.sqrt(np.diag(C)))
        
        # Initial evaluation
        fx_best = func(mean)
        calls_used += 1
        
        # Track best solution found so far
        x_best = mean.copy()
        f_best = fx_best
        
        while not check_time():
            # Generate population
            ys = np.zeros((lambda_pop, dim))
            fs = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls_used += 1
                
                if fs[i] < f_best:
                    f_best = fs[i]
                    x_best = y.copy()
                
                # Check time after each call to be safe
                if check_time():
                    break
            
            if check_time():
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            mean_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            # Update step size control variable ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B_inv @ D_inv @ (mean_new - mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B_inv.T @ (D_inv @ (mean_new - mean)))
            
            # Update covariance control variable pc
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lambda_pop + 1)))) < (1 + 2 / (dim + 1)) * np.sqrt(c_sigma)
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - mean) / sigma
            
            # Update covariance matrix C
            # Delta H = c_1 * (pc @ pc.T - C) + c_mu * sum(weights_i * (y_i - mean) @ (y_i - mean).T)
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (y_i - mean) @ (y_i - mean).T)
            
            # Correction term for pc update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Rank-1 update part
            rank1 = pc @ pc.T
            
            # Rank-mu update part
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = ys_sorted[i] - mean
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + delta_h * C) + c_mu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update Cholesky factorization
            try:
                D, B = np.linalg.cholesky(C)
                B_inv = np.linalg.inv(B)
                D_inv = np.diag(1.0 / np.sqrt(np.diag(C)))
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity or diagonal
                C = np.eye(dim)
                D, B = np.linalg.cholesky(C)
                B_inv = np.linalg.inv(B)
                D_inv = np.diag(1.0 / np.sqrt(np.diag(C)))
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                
            # Update step size
            # chi_n is the expected norm of a standard normal vector
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma
            sigma = max(1e-10, min(sigma, 10.0))
            
            # Update mean
            mean = mean_new

        return x_best, f_best
# EVOLVE-BLOCK-END
