import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, global_best_x, global_best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls += 1
            
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x_clipped.copy()
            
            return f_val

        # 1. Global Exploration Phase
        # Sample a diverse set of points to find the basin of the global minimum.
        # Rastrigin is multimodal. We need to be careful not to waste too many
        # calls here, but 50-60 is reasonable for 500 total.
        n_global = 50
        rng = np.random.default_rng(42)
        
        # Stratified sampling or just uniform? Uniform is simpler.
        # Let's use a mix of uniform and some structured points (like center).
        points = []
        # Add center
        points.append(np.zeros(dim))
        # Add random points
        for _ in range(n_global - 1):
            points.append(rng.uniform(-5, 5, dim))
            
        values = []
        for p in points:
            v = evaluate(p)
            values.append(v)
            
        if calls >= budget:
            return global_best_x, global_best_f

        # Identify the best starting point from global exploration
        best_idx = np.argmin(values)
        start_x = points[best_idx].copy()
        
        # 2. CMA-ES Phase
        # We will run CMA-ES starting from the best point found.
        # Parameters for CMA-ES
        # lambda = 4 + floor(3 * log(dim))
        # For dim=5, log(5) ~ 1.6, 3*1.6 = 4.8, floor=4, 4+4=8.
        # Let's use lambda=10 for slightly better exploration in the basin.
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        x_mean = start_x.copy()
        # Initial sigma: if we found a good point, the basin might be narrow.
        # Rastrigin basins are roughly 2 units wide. 
        # If start_x is close to min, sigma can be smaller. 
        # But we don't know how close. 
        # A safe bet is sigma=1.0 or 0.5. Let's start with 1.0.
        sigma = 1.0
        C = np.eye(dim)
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + 4)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # CMA-ES Loop
        # We run until budget is exhausted or convergence/stagnation.
        # Since we have a hard budget, we just loop while calls < budget.
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Check for convergence
            # If sigma is very small, or variance is tiny
            if sigma < 1e-8:
                break
            
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or fix
                C = np.eye(dim)
                L = np.eye(dim)
                
            samples = []
            f_vals = []
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * L @ z
                samples.append(y)
            
            # Evaluate all samples
            for y in samples:
                f_val = evaluate(y)
                f_vals.append(f_val)
                
            if calls >= budget:
                break
                
            # Sort by fitness
            indices = np.argsort(f_vals)
            sorted_samples = [samples[i] for i in indices]
            sorted_f_vals = [f_vals[i] for i in indices]
            
            # Update mean
            new_x_mean = np.sum(weights[:, np.newaxis] * np.array(sorted_samples[:mu]), axis=0)
            
            # Update step size sigma
            y_sigma = (new_x_mean - x_mean) / (sigma * np.linalg.norm(L))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lam + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            y_c = (new_x_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Rank-one update
            C = (1 + c_1 * (2 - h_sigma) - c_1) * C + c_1 * np.outer(y_c, y_c)
            
            # Rank-mu update
            # y_i = (sorted_samples[i] - x_mean) / sigma
            # delta = h_sigma * c_c * (2 - c_c)
            delta = h_sigma * c_c * (2 - c_c)
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (sorted_samples[i] - x_mean) / sigma
                rank_mu_term += weights[i] * np.outer(y_i, y_i)
            
            C = C + c_mu * (delta * np.outer(p_c, p_c) + (1 - delta) * np.eye(dim) - rank_mu_term)
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Update x_mean
            x_mean = new_x_mean
            
            # Check if C is still valid
            # If eigenvalues become negative, reset C
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C = np.eye(dim) * np.max(eigvals) if np.max(eigvals) > 0 else np.eye(dim)

        return global_best_x, global_best_f
