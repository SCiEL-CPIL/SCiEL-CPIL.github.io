"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Single-run CMA-ES with proper covariance adaptation for BBOB f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, 10s buffer
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial parameters
        # Start center of the box, with a step size that covers a large portion
        # of the search space initially.
        x = np.zeros(dim)
        sigma = 2.0
        
        # CMA-ES parameters
        # Standard recommendations for dim=5
        # lambda = 4 + floor(2*ln(dim))
        lam = 4 + int(2 * np.log(dim))
        if lam > 10:
            lam = 10
        # Log-weighted selection
        weights = np.log(np.arange(lam, 0, -1))
        weights /= weights.sum()
        mu = int(lam / 2)
        
        # Adaptation rates
        c_sigma = (mu + 2) / (dim + mu + 5)
        c_c = (4 + dim / dim) / (dim + 4 + 2 * dim / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        mu_sigma = 1 / np.sum(weights[:mu]**2)
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        
        # Chi_n for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        best_x = x.copy()
        best_f = np.inf
        calls_used = 0
        
        # Pre-calculate inverse square root of C for whitening (optional, 
        # but for dim=5 we can just use Cholesky or eigendecomposition each step 
        # or simply generate from standard normal and transform by C^0.5)
        # For simplicity and robustness in dim=5, we use the standard CMA-ES 
        # update steps. Generating Y = B @ C is not strictly correct for sampling 
        # from N(0, C). We need to sample from N(0, C).
        # If C is the covariance matrix, we can sample z ~ N(0, I) and y = L @ z 
        # where C = L @ L.T. Or we can use the fact that if we maintain the 
        # transformation, we can just use standard normals and scale.
        # Standard CMA-ES implementation:
        # 1. Generate B ~ N(0, I)
        # 2. Y = B @ C  <-- This is WRONG if C is the covariance.
        #    Correct: We need to sample from N(0, C).
        #    If C is updated via rank-one/mu updates, it remains symmetric positive definite.
        #    We can compute the square root of C. For dim=5, eigendecomposition is fast.
        
        # Let's stick to a robust implementation.
        
        for gen in range(budget):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            # Generate population
            # Sample from N(0, I)
            B = rng.standard_normal((lam, dim))
            
            # Transform to N(0, C)
            # Compute square root of C
            # For dim=5, eigendecomposition is very fast and robust
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Y = B @ C_sqrt
            D = np.linalg.norm(Y, axis=1)
            D = np.maximum(D, 1e-10)
            Y = Y / D[:, None]
            
            # Generate candidates
            pops = x + sigma * Y
            # Clip to box [-5, 5]
            pops = np.clip(pops, -5, 5)
            
            # Evaluate
            pop_f = np.empty(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                pop_f[i] = func(pops[i])
                calls_used += 1
            
            if calls_used < budget:
                break
                
            # Sort by fitness
            sorted_indices = np.argsort(pop_f)
            
            # Update best
            for idx in sorted_indices:
                if pop_f[idx] < best_f:
                    best_f = pop_f[idx]
                    best_x = pops[idx].copy()
                    
            # Update mean
            y_sigma = np.zeros(dim)
            y_c = np.zeros(dim)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma
            p_c = (1 - c_c) * p_c
            
            for i in range(mu):
                idx = sorted_indices[i]
                w = weights[i]
                y_i = (pops[idx] - x) / sigma
                y_sigma += w * y_i
                y_c += w * y_i
                p_sigma += w * y_i
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 50)
            
            # Update C
            # Rank-mu update
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(dim+1)) * chi_n
            # For simplicity, we can skip the h_sigma heuristic or use a simple version.
            # Let's include a basic check.
            h_sigma = 1.0
            # This is an approximation, usually involves tracking the norm over time.
            # For dim=5 and short runs, it's often safe to just use the update.
            
            rank_mu = 2 * (1 + c_sigma) - c_c
            # h_delta = 1 if ||p_c|| / sqrt(1 - (1-c_c)^{2(gen+1)}) < (1.4 + c_c) * chi_n
            h_delta = 1.0
            
            C = (1 + h_sigma * c_sigma * (d_sigma - 1 + c_sigma) + h_delta * c_c * (2 - c_c) - c_c) * C
            
            # Wait, the standard CMA-ES update for C is:
            # C = (1 + c_sigma * (d_sigma - 1 + c_sigma) + c_c * (2 - c_c) - c_c) * C 
            #     + c_sigma * (1 - c_sigma) * p_sigma * p_sigma.T
            #     + c_c * (1 - c_c) * p_c * p_c.T
            #     + mu_sigma * sum_{i=1}^{mu} weights[i] * y_i * y_i.T
            
            # Let's re-implement the C update correctly based on Hansen's paper.
            
            # Recalculate C update properly
            C = (1 + c_sigma * (d_sigma - 1 + c_sigma) + c_c * (2 - c_c) - c_c) * C
            C += c_sigma * (1 - c_sigma) * np.outer(p_sigma, p_sigma)
            C += c_c * (1 - c_c) * np.outer(p_c, p_c)
            
            # Rank-mu update
            for i in range(mu):
                idx = sorted_indices[i]
                w = weights[i]
                y_i = (pops[idx] - x) / sigma
                C += mu_sigma * w * np.outer(y_i, y_i)
                
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x = x + sigma * y_sigma
            x = np.clip(x, -5, 5)
            
            # Optional: If sigma becomes very small, we might be stuck.
            # But for dim=5 and budget 500, it should converge well.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
