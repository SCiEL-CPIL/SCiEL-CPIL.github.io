import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            dim = self.dim
            budget = self.budget
            
            start_time = time.time()
            deadline = start_time + 290
            
            rng = np.random.default_rng(42)
            
            # CMA-ES Parameters
            lambda_ = 4 + int(3 * np.log(dim)) # 4 + 3*1.6 = 8.8 -> 8? Let's use 10 for dim=5
            if lambda_ < 4:
                lambda_ = 4
            # For dim=5, lambda=10 is a good choice.
            lambda_ = 10 
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Strategy parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((dim + 2)**2 + mu_eff))
            
            # Initialize
            x_mean = rng.uniform(-5, 5, dim)
            sigma = 2.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # Pre-compute Cholesky decomposition of C
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)
                
            evals_used = 0
            best_x = x_mean.copy()
            best_f = float('inf')
            
            # We need to be careful with the budget.
            # Each iteration uses lambda_ evaluations.
            
            while evals_used + lambda_ <= budget and time.time() < deadline:
                # Generate population
                ys = rng.standard_normal((lambda_, dim))
                xs = np.array([x_mean + sigma * (C_chol @ y) for y in ys])
                
                # Clip to box? The problem says func clips. 
                # But for CMA-ES, clipping might distort the distribution.
                # However, the domain is [-5, 5]. If we generate outside, func clips.
                # It is better to keep the search inside or rely on clipping.
                # Let's rely on func clipping as per contract: "func(x) clips ... to the box".
                
                # Evaluate
                fits = np.array([func(x) for x in xs])
                evals_used += lambda_
                
                # Update best
                idx_min = np.argmin(fits)
                if fits[idx_min] < best_f:
                    best_f = fits[idx_min]
                    best_x = xs[idx_min].copy()
                    
                # Sort
                order = np.argsort(fits)
                xs_sorted = xs[order]
                fits_sorted = fits[order]
                
                # Update mean
                x_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # Update evolution paths
                y_mean = (x_mean - x_mean_prev) / sigma # Wait, x_mean_prev is needed
                # I need to store x_mean_prev.
                
                # ... (Implementation complexity is high for CMA-ES in a short script)
