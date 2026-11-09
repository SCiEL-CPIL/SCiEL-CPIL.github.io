import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_F = []
        eval_count = 0
        
        # 1. Determine fixed values for non-critical variables
        # Critical: x[0], x[1]
        # Semi-critical: x[2] (hint says fix to 0.75 range)
        # Bulk: x[3]...x[11] (fix to midpoint)
        
        x_fixed = np.zeros(self.dim)
        # x[0] and x[1] will be varied
        # x[2]
        x_fixed[2] = lo[2] + 0.75 * (hi[2] - lo[2])
        # x[3] to x[11]
        for i in range(3, self.dim):
            x_fixed[i] = (lo[i] + hi[i]) / 2.0
            
        # 2. Grid search on x[0] and x[1]
        n_grid = 12
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        grid_points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = x_fixed.copy()
                x[0] = x0
                x[1] = x1
                
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                f = problem(x)
                all_F.append(f)
                grid_points.append(x.copy())
                eval_count += 1
            if eval_count >= self.budget or time.time() > deadline:
                break
        
        if eval_count >= self.budget or time.time() > deadline:
            return np.array(all_F)

        # 3. Local Refinement
        # Find non-dominated points from the grid
        F_grid = np.array(all_F)
        n_points = len(F_grid)
        
        # Simple non-dominated check
        non_dom_mask = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            if not non_dom_mask[i]:
                continue
            for j in range(n_points):
                if i == j or not non_dom_mask[j]:
                    continue
                # If j dominates i, mark i as dominated
                if np.all(F_grid[j] <= F_grid[i]) and np.any(F_grid[j] < F_grid[i]):
                    non_dom_mask[i] = False
                    break
        
        non_dom_indices = np.where(non_dom_mask)[0]
        
        # If we have few non-dominated points, we can refine them more
        # Use remaining budget
        remaining_budget = self.budget - eval_count
        
        # Shuffle non-dominated points to pick a subset if too many
        if len(non_dom_indices) > 0:
            np.random.shuffle(non_dom_indices)
            
            # We have 56 evals left. 
            # Let's try to refine each non-dominated point with a few steps.
            # Or just do random local searches.
            
            # Strategy: For each non-dominated point, try a few random perturbations.
            # Number of points to refine: min(len(non_dom_indices), remaining_budget // 5)
            n_refine = min(len(non_dom_indices), max(1, remaining_budget // 5))
            
            for idx in non_dom_indices[:n_refine]:
                if time.time() > deadline:
                    break
                if eval_count >= self.budget:
                    break
                    
                x_best = grid_points[idx]
                f_best = F_grid[idx]
                
                # Local search
                step_size = 0.05 * (hi - lo)
                
                for _ in range(5): # 5 steps per point
                    if time.time() > deadline:
                        break
                    if eval_count >= self.budget:
                        break
                        
                    delta = np.random.randn(self.dim) * step_size
                    x_new = x_best + delta
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    all_F.append(f_new)
                    eval_count += 1
                    
                    # Check if f_new dominates f_best or is non-dominated against the set?
                    # For simplicity, just accept if it's better in sum or just add it.
                    # Adding it is fine, hypervolume calculation will handle it.
                    # But to guide the search, let's keep the best local point.
                    # A simple check: if sum(f_new) < sum(f_best), it's a good candidate.
                    # However, multi-objective is tricky. 
                    # Let's just record it. The hypervolume calculator will pick the best.
                    
                    # Update x_best if f_new seems promising (e.g., lower sum)
                    if np.sum(f_new) < np.sum(f_best):
                        x_best = x_new
                        f_best = f_new
                    else:
                        # Sometimes a point with higher sum is on the front.
                        # We can't easily tell without full dominance check against all_F.
                        # Given the budget, simple sum-based local search is a heuristic.
                        pass

        return np.array(all_F)
