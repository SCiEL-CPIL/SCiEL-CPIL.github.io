import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Parameters
        cs = (n + 2) / (n**2 + 4 * n + 4)
        sigma_norm = np.sqrt(n)
        
        x = np.zeros(n)
        fx = func(x)
        sigma = 1.0
        p_sigma = np.zeros(n)
        
        calls = 1
        start_time = time.time()
        deadline = start_time + 290
        
        best_x = x
        best_f = fx
        
        no_improve_count = 0
        max_no_improve = 50
        
        while calls < budget and (time.time() - start_time) < deadline:
            z = np.random.standard_normal(n)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_x, best_f = x, fx
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                # Update p_sigma
                p_sigma = (1 - cs) * p_sigma + cs * (z / np.linalg.norm(z))
            else:
                no_improve_count += 1
                p_sigma = (1 - cs) * p_sigma
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(cs * (norm_p_sigma / sigma_norm - 1))
            
            # Keep sigma within reasonable bounds
            if sigma < 1e-6: sigma = 1e-6
            if sigma > 10: sigma = 10
            
            # Restart if stuck
            if no_improve_count > max_no_improve:
                # Restart from a random point
                x = np.random.uniform(-5, 5, n)
                fx = func(x)
                calls += 1
                sigma = 2.0
                p_sigma = np.zeros(n)
                no_improve_count = 0
                if fx < best_f:
                    best_x, best_f = x, fx

        return best_x, best_f
