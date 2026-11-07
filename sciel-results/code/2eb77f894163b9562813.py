import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        total_calls = 0
        
        # Phase 1: Global exploration with multiple random starts
        # Use first 25% of budget for exploration
        explore_budget = int(budget * 0.25)
        num_starts = 8
        budget_per_start = max(1, explore_budget // num_starts)
        
        for s in range(num_starts):
            if time.time() > deadline or total_calls >= budget:
                break
                
            x0 = rng.uniform(-5, 5, dim)
            fx = func(x0)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
            
            # Brief local search from this start
            x = x0.copy()
            sigma = 1.5
            for _ in range(budget_per_start - 1):
                if time.time() > deadline or total_calls >= budget:
                    break
                    
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                else:
                    # Shrink sigma on failure
                    sigma *= 0.85
                
                sigma = max(0.01, min(sigma, 5.0))
        
        # Phase 2: Intensive local search around best solution using CMA-ES-like approach
        if best_x is not None and total_calls < budget:
            x = best_x.copy()
            fx = best_f
            
            # Use a simple covariance matrix adaptation approach
            # Initialize with identity matrix scaled by sigma
            sigma = 0.5
            C = np.eye(dim) * sigma**2
            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(dim) * sigma
            
            successes = 0
            attempts = 0
            window_size = 15
            pc = np.zeros(dim)
            cs = 1.0 + 2 * ((dim + 1) / (dim + 4) - 1)
            c1 = (2 / ((dim + 1.3)**2 + 10)) * 0.5
            cmu = min(0.3 - c1, 2 / ((dim + 2)**2 - 4))
            cc = (4 + dim / dim) / (4 + dim + 2)
            hsig = (1 + 2 * max(0, (1 - cc) / (1 + cc))) * (1 - cc)
            
            while total_calls < budget:
                if time.time() > deadline:
                    break
                    
                # Generate offspring
                z = rng.standard_normal(dim)
                # Transform z using Cholesky factor
                y = x + L @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    # Update mean
                    x_new = y.copy()
                    # Update path
                    pc = (1 - cs) * pc + cs * (z / sigma)
                    # Update covariance
                    C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cs * (2 - cs) - 1) * np.eye(dim)) + cmu * np.outer(z, z)
                    # Renormalize
                    C = (C + C.T) / 2
                    # Ensure positive semi-definite
                    try:
                        L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                    except:
                        L = np.eye(dim) * sigma
                        C = np.eye(dim) * sigma**2
                    
                    x = x_new
                    fx = fy
                    successes += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = x.copy()
                else:
                    # No update to mean, but still adapt
                    pc = (1 - cs) * pc
                    C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                    C = (C + C.T) / 2
                    try:
                        L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                    except:
                        L = np.eye(dim) * sigma
                        C = np.eye(dim) * sigma**2
                
                attempts += 1
                
                if attempts >= window_size:
                    success_rate = successes / attempts
                    if success_rate > 0.4:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    else:
                        sigma *= 1.0
                    sigma = np.clip(sigma, 0.001, 3.0)
                    # Rescale C
                    C = C * (sigma**2 / (np.trace(C) / dim)) if np.trace(C) > 0 else np.eye(dim) * sigma**2
                    try:
                        L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                    except:
                        L = np.eye(dim) * sigma
                        C = np.eye(dim) * sigma**2
                    successes = 0
                    attempts = 0
        
        return best_x, best_f
