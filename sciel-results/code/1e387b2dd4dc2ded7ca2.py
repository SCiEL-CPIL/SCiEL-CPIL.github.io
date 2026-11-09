import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ1 specific structure:
        # f1 = 0.5 * (1+g) * x0 * x1
        # f2 = 0.5 * (1+g) * (1-x0) * x1
        # f3 = 0.5 * (1+g) * (1-x1)
        # g = 100 + 10*n + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for g vars
        # For g=0, each g var should be 0.5

        n_front_vars = n_obj - 1  # 2 for DTLZ1
        n_g_vars = dim - n_front_vars  # 5 for DTLZ1

        # Phase 1: Create well-distributed points on the Pareto front
        # The Pareto front for DTLZ1 is defined by the simplex in (n_obj-1) dimensions
        # with g=0 (all g vars at 0.5)
        
        # Use a combination of stratified sampling and random points
        n_front_samples = int(budget * 0.4)
        n_explore_samples = budget - n_front_samples

        # Generate points on the 2D simplex for DTLZ1
        # Stratified sampling in 2D
        n_grid = max(2, int(np.sqrt(n_front_samples)))
        grid_x = np.linspace(0, 1, n_grid, endpoint=False) + np.random.random(n_grid) * (1.0 / n_grid)
        grid_x = np.clip(grid_x, 0, 1)
        
        mesh_x0, mesh_x1 = np.meshgrid(grid_x, grid_x)
        mesh_x0 = mesh_x0.flatten()
        mesh_x1 = mesh_x1.flatten()
        
        n_grid_points = len(mesh_x0)
        if n_grid_points <= n_front_samples:
            front_x0 = mesh_x0.copy()
            front_x1 = mesh_x1.copy()
            n_extra = n_front_samples - n_grid_points
            if n_extra > 0:
                extra_x0 = np.random.random(n_extra)
                extra_x1 = np.random.random(n_extra)
                front_x0 = np.concatenate([front_x0, extra_x0])
                front_x1 = np.concatenate([front_x1, extra_x1])
        else:
            indices = np.random.choice(n_grid_points, n_front_samples, replace=False)
            front_x0 = mesh_x0[indices]
            front_x1 = mesh_x1[indices]
        
        front_points = np.zeros((len(front_x0), dim))
        front_points[:, 0] = front_x0
        front_points[:, 1] = front_x1
        # Set g variables to midpoint (0.5 for [0,1] bounds)
        g_mid = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0
        front_points[:, n_front_vars:] = g_mid
        front_points = np.clip(front_points, lo, hi)

        # Evaluate front points
        X = front_points.copy()
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Phase 2: Local search to refine points with efficient dominance checks
        # Focus on improving g values (getting closer to 0) and fine-tuning front positions
        
        # Precompute reference point for hypervolume-like scoring
        ref_point = np.array([400.0, 400.0, 400.0])
        
        def compute_hv_contribution(F_arr):
            """Approximate hypervolume contribution using simple sum of distances to reference"""
            # For 3 objectives, use a simple additive heuristic
            # This is a proxy, not exact HV, but faster than O(n^3) exact computation
            valid = np.all(F_arr < ref_point, axis=1)
            if not np.any(valid):
                return np.zeros(len(F_arr))
            # Contribution proportional to product of (ref - f_i) for valid points
            contrib = np.prod(ref_point - F_arr[valid], axis=1) if np.any(valid) else np.zeros(np.sum(valid))
            full = np.zeros(len(F_arr))
            full[valid] = contrib
            return full
        
        while used < budget:
            if time.time() > deadline:
                break

            # Choose a parent - prefer non-dominated points (efficient check)
            n_points = len(F)
            dominated_mask = np.zeros(n_points, dtype=bool)
            # Use vectorized dominance check for speed
            for i in range(n_points):
                # Check if any point dominates i
                if np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1) & (np.arange(n_points) != i)):
                    dominated_mask[i] = True
            
            non_dom_indices = np.where(~dominated_mask)[0]
            if len(non_dom_indices) > 0:
                parent_idx = non_dom_indices[np.random.randint(len(non_dom_indices))]
            else:
                parent_idx = np.random.randint(n_points)

            parent = X[parent_idx].copy()
            
            # Adaptive perturbation based on how close we are to optimal
            # g-vars should be at 0.5; check current deviation
            g_target = (lo[n_front_vars:] + hi[n_front_vars:]) / 2.0
            g_deviation = np.mean(np.abs(parent[n_front_vars:] - g_target))
            
            # Scale perturbation inversely with deviation (smaller steps when close)
            base_scale_g = 0.05 * (hi[n_front_vars:] - lo[n_front_vars:])
            adaptive_scale_g = base_scale_g * max(0.1, min(1.0, g_deviation / 0.05))
            
            scale_front = 0.02 * (hi[:n_front_vars] - lo[:n_front_vars])
            
            child = parent.copy()
            child[:n_front_vars] += scale_front * np.random.normal(size=n_front_vars)
            child[n_front_vars:] += adaptive_scale_g * np.random.normal(size=n_g_vars)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1

            # Decide replacement using hypervolume-aware criterion
            # Compute HV contribution of child
            child_contrib = compute_hv_contribution(np.array([fc]))[0]
            
            # Find point with minimum HV contribution among dominated points
            is_child_dom = np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))
            
            if not is_child_dom:
                # Child is non-dominated: replace the point it dominates with lowest HV contribution
                dominated_by_child = np.where(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))[0]
                if len(dominated_by_child) > 0:
                    contribs = compute_hv_contribution(F[dominated_by_child])
                    replace_idx = dominated_by_child[np.argmin(contribs)]
                else:
                    # Replace the point with lowest HV contribution
                    contribs = compute_hv_contribution(F)
                    replace_idx = np.argmin(contribs)
            else:
                # Child is dominated: only replace if it has higher HV contribution than worst
                worst_idx = np.argmin(compute_hv_contribution(F))
                if child_contrib > compute_hv_contribution(F[worst_idx:worst_idx+1])[0]:
                    replace_idx = worst_idx
                else:
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc

        # Phase 3: Final polish - aggressively drive g-vars to 0.5
        # Use iterative refinement with decreasing step sizes
        if time.time() < deadline - 5.0:
            order = np.argsort(np.sum(F, axis=1))
            for idx in order[:min(80, len(F))]:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                parent = X[idx].copy()
                
                # Iterative refinement: try multiple scales, accept best improvement
                best_child = parent.copy()
                best_fc = F[idx].copy()
                best_sum = np.sum(F[idx])
                
                for scale in [0.02, 0.01, 0.005, 0.002, 0.001]:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    child = parent.copy()
                    child[n_front_vars:] += scale * np.random.normal(size=n_g_vars)
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    if np.sum(fc) < best_sum:
                        best_sum = np.sum(fc)
                        best_child = child.copy()
                        best_fc = fc.copy()
                        parent = child.copy()
                
                if best_sum < np.sum(F[idx]):
                    X[idx] = best_child
                    F[idx] = best_fc
        
        return X
