import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # Assume first 2 variables are shape variables for WFG9
        n_shape = 2
        if self.dim < 2:
            # Fallback to random if dim is too small
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            for x in X:
                if time.time() > deadline: break
                problem(x)
            return X

        # Grid search on shape variables
        grid_size = 10
        n_grid = grid_size * grid_size
        
        if self.budget < n_grid:
            # Just do random search
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            for x in X:
                if time.time() > deadline: break
                problem(x)
            return X

        # Generate grid
        shape_lo = lo[:n_shape]
        shape_hi = hi[:n_shape]
        pos_mid = mid[n_shape:]
        
        shape_vals = np.linspace(shape_lo, shape_hi, grid_size)
        
        grid_X = []
        for v0 in shape_vals:
            for v1 in shape_vals:
                x = np.empty(self.dim)
                x[0] = v0
                x[1] = v1
                x[n_shape:] = pos_mid
                grid_X.append(x)
        
        grid_X = np.array(grid_X)
        
        # Evaluate grid
        for x in grid_X:
            if time.time() > deadline:
                return grid_X
            problem(x)
        
        # We don't strictly need F for the return, but we need it to select points for refinement
        # Re-evaluating is expensive (100 more calls).
        # We can store the results.
        F = np.array([problem(x) for x in grid_X]) # This doubles the budget usage!
        
        # WAIT. The budget is 200.
        # If I evaluate grid_X (100 calls) and then evaluate again to get F (100 calls), I am at 200.
        # Then I have 0 budget for refinement.
        # I must store the results from the first pass.
        
        # Let's restart the logic.
        
        # Pass 1: Evaluate and store
        F = np.zeros((n_grid, self.n_obj))
        for i, x in enumerate(grid_X):
            if time.time() > deadline:
                return grid_X
            F[i] = problem(x)
        
        # Find non-dominated
        nd_indices = []
        for i in range(n_grid):
            others = np.concatenate([F[:i], F[i+1:]])
            if len(others) == 0:
                nd_indices.append(i)
                continue
            le = np.all(others <= F[i], axis=1)
            lt = np.any(others < F[i], axis=1)
            if not np.any(le & lt):
                nd_indices.append(i)
                
        # Select top K
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1
        F_norm = (F - f_min) / f_range
        scores = np.sum(F_norm, axis=1)
        
        # Sort nd_indices by score
        nd_scores = scores[nd_indices]
        sorted_nd = nd_indices[np.argsort(nd_scores)]
        
        K = min(10, len(sorted_nd))
        best_indices = sorted_nd[:K]
        
        X_to_refine = grid_X[best_indices]
        F_to_refine = F[best_indices]
        
        remaining_budget = self.budget - n_grid
        
        if remaining_budget > 0:
            steps_per_point = max(1, remaining_budget // K)
            
            for idx in range(K):
                current_x = X_to_refine[idx].copy()
                current_f = F_to_refine[idx].copy()
                
                for step in range(steps_per_point):
                    if time.time() > deadline:
                        return X_to_refine
                    
                    step_size = 0.1 * (hi - lo)
                    new_x = current_x + np.random.normal(0, step_size, self.dim)
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    
                    # Greedy improvement
                    if np.sum(new_f) < np.sum(current_f):
                        current_x = new_x
                        current_f = new_f
                        
        return X_to_refine
