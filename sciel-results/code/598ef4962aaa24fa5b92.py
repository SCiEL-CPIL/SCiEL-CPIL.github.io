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
        deadline = start_time + 290.0

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                f_val = float(func(x))
            except:
                f_val = np.inf
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Grid Search (3^5 = 243 evals)
        # Points: -5, 0, 5 in each dimension
        grid_points = np.array(np.meshgrid(
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3),
            np.linspace(-5, 5, 3)
        )).T.reshape(-1, 5)
        
        for x in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)
            
        if best_f == np.inf:
            best_x = np.zeros(dim)
            evaluate(best_x)

        # Phase 2: CMA-ES Local Refinement
        # Use remaining budget
        remaining_budget = budget - calls
        
        if remaining_budget > 10:
            # CMA-ES Parameters
            # Population size lambda
            lam = 4 + int(3 * np.log(dim))
            # Capping lambda for small budgets
            if lam > remaining_budget // 2:
                lam = max(4, remaining_budget // 2)
            
            # Initialize CMA-ES
            x_mean = best_x.copy()
            sigma = 1.0 # Initial step size
            C = np.eye(dim) # Covariance matrix
            B = np.eye(dim) # Eigenvectors
            D = np.ones(dim) # Eigenvalues
            # State variables for CSA
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(20*dim*dim))
            
            # Weights
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Main Loop
            while calls < budget and time.time() < deadline:
                # Generate population
                population = []
                for i in range(lam):
                    z = np.random.multivariate_normal(np.zeros(dim), C, 1).flatten()
                    # Or use B * (D * z) for efficiency
                    z = B @ (D * np.random.randn(dim))
                    y = x_mean + sigma * z
                    population.append(y)
                
                # Evaluate population
                f_values = np.array([evaluate(y) for y in population])
                
                # Sort
                order = np.argsort(f_values)
                selected = [population[i] for i in order[:mu]]
                selected_f = f_values[order[:mu]]
                
                # Update mean
                x_mean_new = np.sum(np.array(selected) * weights[:, None], axis=0)
                
                # Update step size (CSA)
                # Calculate ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (B.T @ (x_mean_new - x_mean) / sigma)))
                
                # Check if ps is too large
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((budget - calls) / (dim + c_sigma))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    # Clamp sigma
                    sigma = np.clip(sigma, 1e-10, 10.0)
                    
                # Update covariance matrix
                # Rank-one update
                v = (x_mean_new - x_mean) / sigma
                C = (1 + c_1 * c_mu) * C - c_1 * C + c_1 * np.outer(v, v)
                
                # Rank-mu update
                y_sigma = x_mean_new - x_mean
                # Note: In standard CMA-ES, the rank-mu update uses the selected points relative to the old mean
                # But here we use the new mean? No, standard is:
                # C += c_mu * sum(wi * (yi - x_mean_old) * (yi - x_mean_old).T)
                # However, we already updated x_mean. Let's use the standard formulation carefully.
                # Actually, it's safer to use the `cma` library logic if possible, but we must implement it.
                # Let's simplify: Just do rank-one update for stability in short runs?
                # Rank-mu is crucial for adapting to shape.
                
                # Re-calculate rank-mu update properly
                # We need the vectors from the OLD mean
                # But we overwrote x_mean? No, x_mean is still old.
                # v was calculated as (x_mean_new - x_mean) / sigma
                # The rank-mu update is:
                # C += c_mu * sum( wi * (yi - x_mean) * (yi - x_mean).T ) / sigma^2
                # Let's add this
                rank_mu_update = np.zeros((dim, dim))
                for i in range(mu):
                    yi = selected[i]
                    vi = (yi - x_mean) / sigma
                    rank_mu_update += weights[i] * np.outer(vi, vi)
                C = C + c_mu * rank_mu_update
                
                # Ensure positive definite (regularization)
                C = (C + C.T) / 2
                # Add small diagonal for stability
                C += 1e-12 * np.eye(dim)
                
                x_mean = x_mean_new
                
                # Update B and D (Eigen decomposition)
                # Do this every 10 generations or when necessary
                # For simplicity and speed in short runs, do it every generation
                # But eigen decomposition is expensive O(n^3). n=5, so it's fast.
                D, B = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                D = np.maximum(D, 1e-10)
                
        return best_x, best_f
