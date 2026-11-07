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
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Global Search using Latin Hypercube Sampling (LHS)
        # BBOB-21 has many local optima. We need a dense global search to locate
        # the basin of the global optimum. We allocate 60% of the budget here.
        n_global = int(budget * 0.6)
        n_global = max(n_global, 50)
        
        # Generate LHS points in [-5, 5]^dim
        lhs = np.zeros((n_global, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_global + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs[:, d] = points

        # Evaluate global points
        f_values = np.zeros(n_global)
        calls = 0
        
        for i in range(n_global):
            if time.time() > deadline:
                break
            f_values[i] = func(lhs[i])
            calls += 1
            
        if calls == 0:
            return np.zeros(dim), 0.0

        # Sort by fitness
        sorted_indices = np.argsort(f_values)
        best_x = lhs[sorted_indices[0]].copy()
        best_f = f_values[sorted_indices[0]]

        # Phase 2: Local Refinement using CMA-ES
        # We take the top K candidates from the global search and run CMA-ES on them.
        # CMA-ES adapts the covariance matrix to better navigate correlated structures.
        
        n_local_candidates = 4
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget per local search
        # With 60% global, we have 40% local. 
        # 200 calls / 4 candidates = 50 calls each.
        budget_per_local = max(10, remaining_budget // n_local_candidates)
        
        for k in range(n_local_candidates):
            if calls >= budget or time.time() > deadline:
                break
            
            # Start from the k-th best global sample
            x = lhs[sorted_indices[k]].copy()
            f = f_values[sorted_indices[k]]
            
            # CMA-ES parameters
            # Initial step size
            sigma = 0.5
            # Population size
            lam = 10
            # Number of weights
            mu = lam // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            # Variance effective population size
            mu_eff = 1.0 / np.sum(w**2)
            # Learning rates
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # State variables
            mean = x.copy()
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            normC = np.sqrt(np.trace(C) / dim)
            hs = (1 + (1 - cc) / 2) * np.sqrt((dim + 1.3) / (dim + 1))
            
            n_evals_local = 0
            max_local_evals = budget_per_local
            no_improvement_count = 0
            max_no_improvement = 20
            
            while n_evals_local < max_local_evals and calls < budget and time.time() < deadline:
                # Generate population
                B = rng.multivariate_normal(np.zeros(dim), C, size=lam)
                # Normalize to unit variance
                B = B / np.linalg.norm(B, axis=1, keepdims=True)
                
                # Sort by norm
                norms = np.linalg.norm(B, axis=1)
                sorted_B = B[np.argsort(norms)]
                
                # Weighted sum of top mu
                y = np.zeros(dim)
                for i in range(mu):
                    y += w[i] * sorted_B[i]
                
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C @ (y / sigma)
                if np.linalg.norm(ps) / hs < np.sqrt(1 - (1 - cs)**2):
                    pc = (1 - cc) * pc + cc * (y / sigma)
                
                # Update covariance matrix
                delta_hs = (np.linalg.norm(ps) / hs - 1) * (1 - cc)
                C = (1 - c1 - cmu + (1 - c1 - cmu + cc * (2 - cc)) * delta_hs) * C \
                    + c1 * (np.outer(pc, pc) - C) \
                    + cmu * np.sum([w[i] * np.outer(sorted_B[i], sorted_B[i]) for i in range(mu)], axis=0)
                
                # Update mean
                mean = mean + sigma * y
                
                # Generate offspring
                offspring = mean + sigma * B
                
                # Clip to box
                offspring = np.clip(offspring, -5, 5)
                
                # Evaluate offspring
                f_offspring = np.array([func(x) for x in offspring])
                calls += lam
                n_evals_local += lam
                
                # Select best
                sorted_idx = np.argsort(f_offspring)
                best_idx = sorted_idx[0]
                
                if f_offspring[best_idx] < f:
                    # Accept
                    x = offspring[best_idx].copy()
                    f = f_offspring[best_idx]
                    no_improvement_count = 0
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    no_improvement_count += 1
                
                # Check for restart
                if no_improvement_count >= max_no_improvement:
                    break

        return best_x, best_f
