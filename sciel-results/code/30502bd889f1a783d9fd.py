import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        max_time = 290.0 # Safety margin for 300s limit
        
        # Initial point: Center of the box
        x = np.zeros(dim)
        fx = func(x)
        
        # CMA-ES (1+1) parameters
        sigma = 1.0
        C = np.eye(dim)
        
        # Learning rates for CMA-ES
        c_1 = 2 / ((dim + 1.3)**2 + 10)
        c_sigma = (dim + 2) / (dim + 4)
        
        # Expected norm of a standard normal vector
        # E[||N(0,1)||] = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2)
        # For d=5, this is approx 2.09
        # We can approximate or calculate.
        import math
        if dim == 1:
            exp_norm = 0.7979
        else:
            exp_norm = math.sqrt(2) * math.gamma((dim + 1) / 2) / math.gamma(dim / 2)
            
        # To avoid division by zero or issues with very small y
        min_sigma = 1e-8
        max_sigma = 10.0
        
        calls = 1
        while calls < budget:
            # Check time
            if time.time() - start_time > max_time:
                break
                
            # Sample
            # Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular or invalid, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
                
            z = np.random.randn(dim)
            y = L @ z
            
            # Normalize y to have unit norm in the C-metric? 
            # Standard CMA samples y ~ N(0, C). 
            # The step is sigma * y.
            
            x_trial = x + sigma * y
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls += 1
            
            if f_trial < fx:
                # Success
                x = x_trial
                fx = f_trial
                
                # Update Covariance (Rank-1)
                # C = (1 - c_1) * C + c_1 * y * y^T
                # Note: y is the step in the transformed space.
                C = (1 - c_1) * C + c_1 * np.outer(y, y)
                
                # Update Sigma
                # h = 1 if ||y|| is close to expected norm
                norm_y = np.linalg.norm(y)
                if norm_y > 1e-10:
                    # Standard CMA-ES sigma update
                    # c_sigma is usually larger, but for (1+1) it's simpler
                    # sigma *= exp(c_sigma * (norm_y / exp_norm - 1))
                    # Let's use a simpler heuristic for (1+1) to be safe
                    # If step was good, increase sigma slightly, else decrease
                    # But CMA-ES logic is better.
                    ratio = norm_y / exp_norm
                    sigma *= np.exp(c_sigma * 0.5 * (ratio - 1)) # Dampened to avoid instability
                else:
                    sigma *= 0.9
                    
                # Clamp sigma
                if sigma < min_sigma:
                    sigma = min_sigma
                elif sigma > max_sigma:
                    sigma = max_sigma
                    
            else:
                # Failure
                # Shrink sigma
                sigma *= 0.9
                if sigma < min_sigma:
                    sigma = min_sigma

        return x, fx
