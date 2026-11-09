import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 structure: first 3 variables shape the Pareto front,
        # remaining 9 contribute to a distance penalty B(x) that we want to minimize.
        # Strategy: fix x4..x12 to midpoints to minimize B(x),
        # then do an adaptive grid search over x1, x2, x3.
        
        # Fix non-shaping variables to midpoints
        x_fixed = (lo + hi) / 2.0
        
        # Extract bounds for the 3 shaping variables
        lo3 = lo[:3]
        hi3 = hi[:3]
        
        # Phase 1: Coarse grid over x1, x2, x3
        # Use 4 points per dimension for coarse grid = 64 evaluations
        n_coarse = 4
        grid1 = np.linspace(lo3[0], hi3[0], n_coarse)
        grid2 = np.linspace(lo3[1], hi3[1], n_coarse)
        grid3 = np.linspace(lo3[2], hi3[2], n_coarse)
        
        used = 0
        all_points = []  # List of (x, f) pairs
        all_F = []  # List of objective vectors
        
        # Generate coarse grid points
        coarse_points = []
        for g1 in grid1:
            for g2 in grid2:
                for g3 in grid3:
                    x = x_fixed.copy()
                    x[0] = g1
                    x[1] = g2
                    x[2] = g3
                    coarse_points.append(x)
        
        # Evaluate coarse grid
        for x in coarse_points:
            if used >= self.budget or time.time() > deadline:
                break
            f = problem(x)
            used += 1
            all_points.append(x)
            all_F.append(f)
        
        if used >= self.budget or time.time() > deadline:
            return np.array(all_F)
        
        # Phase 2: Adaptive refinement
        # Find the region with the lowest objective sum (likely densest Pareto front region)
        F_arr = np.array(all_F)
        sums = F_arr.sum(axis=1)
        min_sum_idx = np.argmin(sums)
        best_x = all_points[min_sum_idx].copy()
        
        # Refine around the best region with a finer grid
        # Use 3 points per dimension in a local region
        n_fine = 3
        # Define a local region around the best point
        local_lo = np.maximum(lo3, best_x - 0.5 * (hi3 - lo3))
        local_hi = np.minimum(hi3, best_x + 0.5 * (hi3 - lo3))
        
        grid1_fine = np.linspace(local_lo[0], local_hi[0], n_fine)
        grid2_fine = np.linspace(local_lo[1], local_hi[1], n_fine)
        grid3_fine = np.linspace(local_lo[2], local_hi[2], n_fine)
        
        fine_points = []
        for g1 in grid1_fine:
            for g2 in grid2_fine:
                for g3 in grid3_fine:
                    x = x_fixed.copy()
                    x[0] = g1
                    x[1] = g2
                    x[2] = g3
                    fine_points.append(x)
        
        # Evaluate fine grid
        for x in fine_points:
            if used >= self.budget or time.time() > deadline:
                break
            f = problem(x)
            used += 1
            all_points.append(x)
            all_F.append(f)
        
        if used >= self.budget or time.time() > deadline:
            return np.array(all_F)
        
        # Phase 3: Use remaining budget for local search around the best points
        F_arr = np.array(all_F)
        
        # Sort by objective sum to find the best points
        sums = F_arr.sum(axis=1)
        sorted_idx = np.argsort(sums)
        
        # Take the top few points and do local search
        top_k = min(5, len(sorted_idx))
        remaining_budget = self.budget - used
        
        if remaining_budget > 0:
            steps_per_point = max(1, remaining_budget // top_k)
            
            for k in range(top_k):
                if used >= self.budget or time.time() > deadline:
                    break
                
                idx = sorted_idx[k]
                x_center = all_points[idx].copy()
                f_center = F_arr[idx]
                
                for _ in range(steps_per_point):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Small perturbation in the 3 shaping variables
                    x_new = x_center.copy()
                    for dim in range(3):
                        range_dim = hi[dim] - lo[dim]
                        step = 0.05 * range_dim
                        x_new[dim] += np.random.randn() * step
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    all_points.append(x_new)
                    all_F.append(f_new)
        
        return np.array(all_F)
