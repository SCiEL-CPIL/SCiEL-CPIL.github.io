"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-stage optimizer for Büche-Rastrigin f4 on [-5,5]^5.
    
    Stage 1: Dense stratified search using Sobol-like sequence to identify promising basins
    Stage 2: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for local refinement
    
    The Büche-Rastrigin function has periodic local minima. After hidden rotation/shift,
    we use a two-phase approach: broad exploration followed by intensive local optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = float('inf')
        budget_used = 0

        # Stage 1: Stratified grid search using Halton/Sobol-like sequence
        # Use first ~100 evaluations for broad exploration
        n_explore = min(100, budget)
        
        # Generate well-spread points using a quasi-random sequence
        # Halton sequence for low-discrepancy sampling
        def halton(index, base):
            f = 1.0
            result = 0.0
            while index > 0:
                f /= base
                result += f * (index % base)
                index //= base
            return result

        # Generate Halton sequence points in [0,1]^dim, then map to [-5,5]^dim
        # Skip first few points (they cluster at corners)
        skip = 5
        for i in range(skip + n_explore):
            if budget_used >= budget or not time_ok():
                break
            if i < skip:
                continue
            # Generate Halton coordinates
            coords = []
            base = 2
            for d in range(dim):
                h = halton(i, base)
                coords.append(h)
                base += 1  # Use next prime-like base (2,3,4,5,6...)
            
            # Map from [0,1]^dim to [-5,5]^dim
            pt = np.array(coords, dtype=float) * 10.0 - 5.0
            pt = np.clip(pt, -5, 5)
            
            f_val = func(pt)
            budget_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Stage 2: CMA-ES for local refinement
        # Initialize CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 0.5  # Initial step size
        
        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Cholesky factor for efficient sampling
        try:
            C_sqrt = np.linalg.cholesky(C)
        except:
            C_sqrt = np.eye(dim)
        
        # Population size for CMA-ES
        mu = max(4, dim // 2 + 1)  # Number of parents
        lam = max(2 * mu, dim + 2)  # Number of offspring
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 1.5 + 2 / dim)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / (dim ** 1.7 + 13 + mu)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1 + 2 / (mu - 2)) / (dim ** 1.7 + 15))
        
        # State vectors
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best
        while budget_used < budget and time_ok():
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if budget_used >= budget or not time_ok():
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                child = x_mean + sigma * (C_sqrt @ z)
                child = np.clip(child, -5, 5)
                
                f_val = func(child)
                budget_used += 1
                offspring.append(child)
                offspring_f.append(f_val)
            
            if len(offspring) == 0:
                break
            
            # Sort offspring by fitness
            order = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in order]
            sorted_offspring_f = [offspring_f[i] for i in order]
            
            # Update best
            for x_i, f_i in zip(offspring, offspring_f):
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            # Recombination: weighted average of best mu offspring
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (sorted_offspring[i] - x_mean) / sigma
            
            # Update step size
            norm_y_w = np.linalg.norm(y_w)
            if norm_y_w > 1e-10:
                h_sigma = 1.0 - (norm_y_w / (np.sqrt(dim) * (1 + 1 / (2 * dim)))) ** 2
                h_sigma = max(0, min(1, h_sigma))
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu) * y_w
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                sigma = max(1e-6, min(sigma, 10.0))
            
            # Update mean
            x_mean = x_mean + sigma * y_w
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update covariance
            y_c = np.zeros(dim)
            for i in range(mu):
                y_c += weights[i] * (sorted_offspring[i] - x_mean) / sigma
            
            p_c = (1 - c_c) * p_c + c_c * y_c
            
            delta_h = (1 + (1 - c_c) * (1 - c_c)) * (1 - c_c) / c_c
            C = (1 + c_1 * delta_h + c_mu) * C - c_1 * np.outer(p_c, p_c) - c_mu * (np.outer(y_c, y_c) / mu)
            
            # Ensure C is positive definite
            try:
                C_sqrt = np.linalg.cholesky(C)
            except:
                # Reset C if it becomes non-positive definite
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_c = np.zeros(dim)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
