import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda=10 is a good balance between exploration and exploitation
        lambda_ = 10
        mu = lambda_ // 2  # Number of weights
        
        # Weights (log-linear)
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial state
        # Start with a broad distribution to cover the box [-5, 5]^dim
        # The problem has a hidden shift/rotation, so starting at center is safe.
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers ~80% of [-5,5] range
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Pre-compute random number generator
        rng = np.random.default_rng(42)
        
        # Helper to evaluate and update best
        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
            
        # Main Loop
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Generate lambda offspring
            # Use Cholesky decomposition for sampling from N(mean, sigma^2 * C)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular (should be rare with CMA-ES updates)
                C = np.eye(dim)
                L = np.eye(dim)
                
            ys = []
            fs = []
            
            for i in range(lambda_):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                
                fy = eval_point(y)
                if fy is None:
                    break
                ys.append(y)
                fs.append(fy)
                
            if len(ys) == 0:
                break
                
            # Sort offspring by fitness
            indices = np.argsort(fs)
            ys_sorted = [ys[i] for i in indices[:mu]]
            fs_sorted = [fs[i] for i in indices[:mu]]
            
            # Update mean
            new_mean = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update covariance matrix
            # pc = (1 - c_c) * pc + c_c * (1/sigma) * (new_mean - mean)
            pc = (1 - c_c) * pc + c_c * (1.0 / sigma) * (new_mean - mean)
            
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (new_mean - mean)
            ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (new_mean - mean)
            
            # Rank-1 and Rank-mu update
            # hsig = (||ps|| / sqrt(1 - (1 - c_sigma)^{2/gen})) / chi_n < 1.4 + 2/(dim+1)
            # We don't track generation exactly for the exponent, but we can approximate or just use a simpler check
            # For robustness, we'll use a simplified hsig check based on ps norm
            ps_norm = np.linalg.norm(ps)
            # Approximate expected norm of ps
            # In standard CMA-ES, this involves generation count. 
            # A common simplification is to check if ps is unexpectedly large.
            # Let's use a fixed threshold relative to chi_n
            hsig = 1.0
            if ps_norm > chi_n * 1.4:
                hsig = 0.0
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * np.eye(dim))
            
            # Rank-mu update
            y_diffs = np.array([y - mean for y in ys_sorted])
            # Weighted sum of outer products
            rank_mu_term = sum(weights[i] * np.outer(y_diffs[i], y_diffs[i]) for i in range(mu))
            C += c_mu * rank_mu_term
            
            # Update sigma (1/5 rule)
            sigma = sigma * math.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Ensure C is symmetric (numerical stability)
            C = (C + C.T) / 2
            
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)
            
        return best_x, best_f
