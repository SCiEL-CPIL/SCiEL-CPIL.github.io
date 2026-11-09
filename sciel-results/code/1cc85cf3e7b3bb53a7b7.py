import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
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

        rng = np.random.default_rng(42)

        # Strategy:
        # 1. Global Exploration: 40 calls to find a good basin.
        # 2. CMA-ES: Run for remaining budget.
        #    - If stagnation occurs (no improvement in 20 calls), restart CMA-ES
        #      with a larger sigma and a new random center, but keep the global best.
        #    - This handles the multimodality of Rastrigin better than a single run.

        # Phase 1: Global Search
        n_global = 40
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
        
        # Phase 2: CMA-ES with Restarts
        # Parameters for CMA-ES
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + 4)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        stagnation_window = 20
        max_restarts = 5
        restarts = 0
        
        # Initial state
        x_mean = start_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        last_improvement_call = calls
        best_f_so_far = global_best_f
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Check for convergence
            if sigma < 1e-8:
                break
            
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
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
            
            # Check for improvement
            if sorted_f_vals[0] < best_f_so_far:
                best_f_so_far = sorted_f_vals[0]
                last_improvement_call = calls
            
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
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C = np.eye(dim) * np.max(eigvals) if np.max(eigvals) > 0 else np.eye(dim)
            
            # Stagnation Check
            if calls - last_improvement_call > stagnation_window:
                # Restart
                restarts += 1
                if restarts > max_restarts:
                    break
                
                # New random center
                x_mean = rng.uniform(-5, 5, dim)
                # Larger sigma to explore more broadly
                sigma = 2.0 
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                
                # Evaluate the new center to potentially update global best
                f_new = evaluate(x_mean)
                if f_new < best_f_so_far:
                    best_f_so_far = f_new
                    last_improvement_call = calls
                else:
                    last_improvement_call = calls # Reset timer for this new lineage

        return global_best_x, global_best_f
