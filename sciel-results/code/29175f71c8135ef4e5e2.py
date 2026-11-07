import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration
        # Sample points to find a good starting location
        n_explore = 15
        if budget < 50:
            n_explore = max(5, budget // 5)
        
        points = rng.uniform(-5, 5, (n_explore, n))
        values = np.zeros(n_explore)
        
        for i in range(n_explore):
            if time.time() > deadline:
                break
            values[i] = func(points[i])
        
        # Fill any unevaluated points with a large value
        for i in range(n_explore):
            if values[i] == 0 and i >= n_explore:
                values[i] = 1e10
        
        # Actually, let's be more careful with the exploration
        # Re-do exploration more carefully
        values = np.full(n_explore, np.inf)
        calls_made = 0
        for i in range(n_explore):
            if time.time() > deadline:
                break
            values[i] = func(points[i])
            calls_made += 1
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]
        
        # Select top 3 points for exploitation
        top_k = min(3, n_explore)
        top_indices = np.argsort(values)[:top_k]
        
        best_x_global = points[top_indices[0]].copy()
        best_f_global = values[top_indices[0]]
        
        # Phase 2: Local Exploitation with CMA-ES-like approach
        # Use (1+1)-ES with adaptive step size, but with better initialization
        # For sharp ridge functions, we want to gradually align with the ridge
        
        # Run CMA-ES on the best point
        x = points[top_indices[0]].copy()
        fx = values[top_indices[0]]
        
        # Initialize step size based on problem scale
        sigma = 1.0
        
        # Track success rate for step size adaptation
        successes = 0
        total = 0
        target_success_rate = 1/(n + 1)  # ~0.17 for n=5
        
        # CMA-ES parameters
        c_sigma = 1 / (5 * n)
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
        c_c = (4 + n / 4) / (4 + n + 4)
        c_1 = 2 / ((n + 1.3) ** 2 + 10)
        c_mu = min(1 - c_1, 2 * ((n + 1) / (n + 3)) ** 2 - 1)
        
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()
        
        # Covariance matrix adaptation
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        c_c = (4 + n / 4) / (4 + n + 4)
        c_1 = 2 / ((n + 1.3) ** 2 + 10)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 1) / (n + 1)) ** 2 - 1)
        
        # Cholesky factor of C
        try:
            C_cholesky = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_cholesky = np.eye(n)
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate lambda_ candidates
            ys = []
            fs = []
            
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (C_cholesky @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls_made += 1
            
            if len(ys) == 0:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update best
            if fs[0] < best_f_global:
                best_f_global = fs[0]
                best_x_global = ys[0].copy()
            
            # Take top mu
            mu_actual = min(mu, len(ys))
            if mu_actual == 0:
                break
            
            # Weighted mean of top mu
            y_mean = np.zeros(n)
            for i in range(mu_actual):
                y_mean += weights[i] * ys[i]
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_cholesky.T @ (y_mean / sigma))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - n_explore) / (n * ds))) < (1.4 + 1/n) * np.sqrt(n):
                sigma *= np.exp(c_sigma * d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (y_mean - x)
            delta = (1 - c_c) if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (calls_made - n_explore) / (n * c_c))) < (1.4 + 2/n) * np.sqrt(n) else 1
            
            rank_mu = np.zeros((n, n))
            for i in range(mu_actual):
                vi = (ys[i] - x) / sigma
                rank_mu += np.outer(vi, vi) * weights[i]
            
            C = (1 + c_1 * (delta - 1) - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
            
            # Ensure positive semi-definite
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-6 * np.eye(n)
                try:
                    C_cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_cholesky = np.eye(n)
            
            # Update mean
            x = y_mean
            
            # Check convergence
            if sigma < 1e-8:
                break
        
        return best_x_global, best_f_global
