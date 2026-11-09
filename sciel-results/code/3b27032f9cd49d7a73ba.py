import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        # 1. Setup
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 2. Fix variables 3..11 to midpoint
        mid = (lo + hi) / 2.0
        fixed_vars = mid[3:]
        
        # 3. Two-stage adaptive strategy
        # Stage 1: Low-resolution grid to map objective space
        # 4 * 4 * 4 = 64 points
        n0, n1, n2 = 4, 4, 4
        xs0 = np.linspace(lo[0], hi[0], n0)
        xs1 = np.linspace(lo[1], hi[1], n1)
        xs2 = np.linspace(lo[2], hi[2], n2)
        
        G0, G1, G2 = np.meshgrid(xs0, xs1, xs2, indexing='ij')
        X0 = G0.flatten()
        X1 = G1.flatten()
        X2 = G2.flatten()
        
        N_grid = len(X0)
        X_grid = np.zeros((N_grid, self.dim))
        X_grid[:, 0] = X0
        X_grid[:, 1] = X1
        X_grid[:, 2] = X2
        X_grid[:, 3:] = fixed_vars
        
        # Evaluate grid points
        results = []
        for i in range(N_grid):
            if time.time() > deadline:
                break
            f = problem(X_grid[i])
            results.append(f)
        
        # Identify non-dominated points from grid
        if len(results) < 2:
            return X_grid
        
        F_grid = np.array(results)
        
        # Find non-dominated indices
        def is_dominated(f, F):
            # f is dominated if there exists g in F such that g <= f in all objectives and g < f in at least one
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
        
        non_dom_indices = []
        for i in range(len(F_grid)):
            if not is_dominated(F_grid[i], F_grid):
                non_dom_indices.append(i)
        
        # Stage 2: Local search from non-dominated points
        # Use remaining budget for refinement
        remaining_budget = self.budget - len(results)
        
        if remaining_budget <= 0 or len(non_dom_indices) == 0:
            return X_grid
        
        # Perform greedy local search from each non-dominated point
        # Try axis-aligned steps to push toward boundaries
        step_sizes = [(hi - lo) * 0.1]
        
        search_candidates = []
        for idx in non_dom_indices:
            x_start = X_grid[idx].copy()
            f_start = F_grid[idx]
            
            # Try moving each of the first 3 variables in both directions
            for dim_i in range(3):
                for direction in [-1, 1]:
                    for step in step_sizes:
                        x_new = x_start.copy()
                        x_new[dim_i] += direction * step
                        x_new = np.clip(x_new, lo, hi)
                        search_candidates.append(x_new)
            
            # Also try smaller steps
            for dim_i in range(3):
                for direction in [-1, 1]:
                    x_new = x_start.copy()
                    x_new[dim_i] += direction * step_sizes[0] * 0.3
                    x_new = np.clip(x_new, lo, hi)
                    search_candidates.append(x_new)
        
        # Deduplicate and limit to remaining budget
        if len(search_candidates) > 0:
            # Remove duplicates
            unique_candidates = []
            seen = set()
            for xc in search_candidates:
                key = tuple(np.round(xc, 8))
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(xc)
            
            # Take up to remaining_budget candidates
            unique_candidates = unique_candidates[:remaining_budget]
            
            # Evaluate candidates
            for xc in unique_candidates:
                if time.time() > deadline:
                    break
                f = problem(xc)
                results.append(f)
        
        return X_grid
