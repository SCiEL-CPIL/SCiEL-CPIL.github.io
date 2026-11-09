import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive of all non-dominated points found
        archive_X = []
        archive_F = []
        used = 0
        
        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive_F_list."""
            for f_arch in archive_F_list:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    return True
            return False
        
        def update_archive(x, f):
            """Update archive with new point if non-dominated."""
            nonlocal archive_X, archive_F
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return
            # Remove points dominated by new point
            new_X = []
            new_F = []
            for xi, fi in zip(archive_X, archive_F):
                if not (np.all(f <= fi) and np.any(f < fi)):
                    new_X.append(xi)
                    new_F.append(fi)
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F
        
        # Phase 1: Deterministic grid-based initialization exploiting WFG9 structure
        # WFG9 has 12 variables. The first 6 are position variables, last 6 are distance variables.
        # We use a structured approach: vary position variables systematically while keeping
        # distance variables at values that produce good hypervolume coverage.
        
        # For WFG9, the Pareto front is determined by the position variables (first 6)
        # while distance variables affect the objectives but can be optimized separately.
        
        # Strategy: Use a grid over the 6 position variables (2^6 = 64 points) 
        # with distance variables set to their optimal values for each position point.
        
        N_grid = 64  # 2^6 for 6 position dimensions
        if N_grid > budget:
            N_grid = budget
        
        # For each grid point, we need to find good distance variable values.
        # We'll use a two-phase approach:
        # Phase 1a: Evaluate grid points with default distance values
        # Phase 1b: Refine distance variables for the best grid points
        
        # Default distance values: midpoint of their ranges
        dist_mid = (lo[6:] + hi[6:]) / 2.0
        
        # Generate grid points for position variables (first 6)
        # Use 2 levels per dimension (low and high) for 64 total points
        grid_points = []
        for i in range(N_grid):
            x = np.zeros(dim)
            for d in range(6):  # Position variables
                bit = (i >> d) & 1
                x[d] = lo[d] if bit == 0 else hi[d]
            # Distance variables at midpoint initially
            x[6:] = dist_mid
            grid_points.append(x)
        
        # Evaluate grid points
        for i in range(N_grid):
            if time.time() > deadline or used >= budget:
                break
            x = grid_points[i]
            f = np.array(problem(x))
            update_archive(x.copy(), f)
            used += 1
        
        # Phase 2: Refine distance variables for promising grid points
        # Sort archive by normalized sum of objectives
        if archive_X and used < budget:
            F_arr = np.array(archive_F)
            f_min = F_arr.min(axis=0)
            f_max = F_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            F_norm = (F_arr - f_min) / f_range
            scores = F_norm.sum(axis=1)
            sorted_indices = np.argsort(scores)
            
            # Refine top points
            n_refine = min(len(sorted_indices), max(3, (budget - used) // 8))
            
            for idx in sorted_indices[:n_refine]:
                if time.time() > deadline or used >= budget:
                    break
                
                x_base = archive_X[idx].copy()
                
                # Try different distance variable configurations
                # Use a small grid over distance variables (3 levels each for 6 dims = 729, too many)
                # Instead, use random sampling in distance space
                n_dist_samples = min(10, (budget - used) // max(1, n_refine))
                
                for _ in range(n_dist_samples):
                    if time.time() > deadline or used >= budget:
                        break
                    
                    x_trial = x_base.copy()
                    # Randomize distance variables
                    x_trial[6:] = np.random.uniform(lo[6:], hi[6:])
                    
                    f_trial = np.array(problem(x_trial))
                    used += 1
                    
                    if not is_dominated(f_trial, archive_F):
                        update_archive(x_trial.copy(), f_trial)
                        x_base = x_trial.copy()
        
        # Phase 3: Use remaining budget for local refinement around best archive points
        remaining = budget - used
        if remaining > 0 and archive_X:
            F_arr = np.array(archive_F)
            f_min = F_arr.min(axis=0)
            f_max = F_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            F_norm = (F_arr - f_min) / f_range
            scores = F_norm.sum(axis=1)
            sorted_indices = np.argsort(scores)
            
            n_local = min(len(sorted_indices), max(2, remaining // 5))
            steps_per_point = max(2, remaining // max(1, n_local))
            
            for idx in sorted_indices[:n_local]:
                if time.time() > deadline or used >= budget:
                    break
                
                x_base = archive_X[idx].copy()
                step_scale = 0.05
                
                for _ in range(steps_per_point):
                    if time.time() > deadline or used >= budget:
                        break
                    
                    direction = np.random.randn(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    
                    x_trial = x_base + direction * step_scale * (hi - lo)
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = np.array(problem(x_trial))
                    used += 1
                    
                    if not is_dominated(f_trial, archive_F):
                        update_archive(x_trial.copy(), f_trial)
                        x_base = x_trial.copy()
                        step_scale *= 0.7
                    else:
                        step_scale = min(step_scale * 1.2, 0.2)
        
        # Phase 4: Final random sampling with any remaining budget
        remaining = budget - used
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline or used >= budget:
                    break
                
                x = np.random.uniform(lo, hi)
                f = np.array(problem(x))
                used += 1
                
                if not is_dominated(f, archive_F):
                    update_archive(x.copy(), f)
        
        # Return the archive points (the evaluator records all returned objective vectors)
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([])
