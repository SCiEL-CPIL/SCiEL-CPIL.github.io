"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to count calls and track best
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # CMA-ES Parameters
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - c_c + 2 * mu_eff / (n - 1)) / ((n + 2) * (c_c + 2 - mu_eff / (n - 1)))
        c_mu = (2 * (mu_eff - 2 + 1.0 / mu_eff)) / ((n + 2) * (c_c + 2 - mu_eff / (n - 1)))
        
        # Initial State
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Generation counter for h_sigma approximation
        gen = 0
        
        while True:
            # Check if we have enough budget for a full generation
            if calls_made + lambda_ > budget:
                break
            
            if time.time() > deadline:
                break
                
            # 1. Sample lambda_ points
            points = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                
            for i in range(lambda_):
                z = np.random.standard_normal(n)
                y = L @ z
                x = x_mean + sigma * y
                points.append(x)
            
            # 2. Evaluate points
            f_vals = []
            for x in points:
                f_val = safe_call(x)
                if f_val is None:
                    return best_x, best_f
                f_vals.append(f_val)
                
            # 3. Sort by fitness
            indices = np.argsort(f_vals)
            points = [points[i] for i in indices]
            f_vals = [f_vals[i] for i in indices]
            
            # 4. Update Mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum([weights[i] * points[i] for i in range(mu)], axis=0)
            
            # 5. Update Step Size (Sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean - x_mean_old)
            
            # h_sigma calculation
            norm_ps = np.linalg.norm(ps)
            # Approximate the decay term
            decay_term = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / mu_eff - 1) * decay_term
            h_sigma = 1.0 if norm_ps < threshold else 0.0
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(2 * n) - 1))
            
            # 6. Update Covariance Matrix C
            # Update pc
            pc = (1 - c_c) * pc + h_sigma * c_c * np.sqrt(mu_eff) * (x_mean - x_mean_old) / sigma
            
            # Rank-one update
            C = (1 + c_1 * (h_sigma * (2 - c_c) - c_c)) * C + c_1 * np.outer(pc, pc)
            
            # Rank-mu update
            y_selected = np.array([(points[i] - x_mean_old) / sigma for i in range(mu)])
            C += c_mu * (y_selected.T @ (weights[:, None] * y_selected))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            gen += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
