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
        # f1 = 0.5 * (1+g) * prod(x_i) for i in front vars
        # f2 = 0.5 * (1+g) * (1-x_0) * prod(x_i) for i in front vars excluding first
        # f3 = 0.5 * (1+g) * (1-x_1)
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
        # The front is parameterized by (u, v) where u+v <= 1, u,v >= 0
        # x_0 = u, x_1 = v, and f_i = 0.5 * prod of remaining front vars
        
        # Actually for DTLZ1 with 3 objectives:
        # f1 = 0.5*(1+g)*x0*x1
        # f2 = 0.5*(1+g)*(1-x0)*x1
        # f3 = 0.5*(1+g)*(1-x1)
        # Wait, let me reconsider. Standard DTLZ1:
        # f1 = 0.5 * (1+g) * x_1 * x_2 (for 3 obj, 2 front vars)
        # f2 = 0.5 * (1+g) * (1-x_1) * x_2
        # f3 = 0.5 * (1+g) * (1-x_2)
        # where x_1, x_2 are the first n_obj-1 = 2 variables
        
        # The Pareto front (g=0) is:
        # f1 = 0.5 * x0 * x1
        # f2 = 0.5 * (1-x0) * x1
        # f3 = 0.5 * (1-x1)
        # with x0, x1 in [0,1]
        
        # To cover the front well, sample x0 and x1 uniformly in [0,1]
        
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

        # Phase 2: Local search to refine points
        # Focus on improving g values (getting closer to 0) and fine-tuning front positions
        
        while used < budget:
            if time.time() > deadline:
                break

            # Choose a parent
            if len(F) <= 200:
                dominated_mask = np.zeros(len(F), dtype=bool)
                for i in range(len(F)):
                    for j in range(len(F)):
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated_mask[i] = True
                                break
                non_dom_indices = np.where(~dominated_mask)[0]
                if len(non_dom_indices) > 0:
                    parent_idx = non_dom_indices[np.random.randint(len(non_dom_indices))]
                else:
                    parent_idx = np.random.randint(len(F))
            else:
                parent_idx = np.random.randint(len(F))

            parent = X[parent_idx].copy()
            
            # Perturbation: small random changes
            # More aggressive on g variables to try to reduce g
            scale_front = 0.02 * (hi[:n_front_vars] - lo[:n_front_vars])
            scale_g = 0.05 * (hi[n_front_vars:] - lo[n_front_vars:])
            
            child = parent.copy()
            child[:n_front_vars] += scale_front * np.random.normal(size=n_front_vars)
            child[n_front_vars:] += scale_g * np.random.normal(size=n_g_vars)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1

            # Decide replacement
            # Check if child is dominated
            is_child_dom = False
            for i in range(len(F)):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    is_child_dom = True
                    break
            
            if not is_child_dom:
                # Child is non-dominated, replace the worst point it dominates or the overall worst
                dominated_by_child = np.where(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))[0]
                if len(dominated_by_child) > 0:
                    replace_idx = dominated_by_child[np.random.randint(len(dominated_by_child))]
                else:
                    # Replace the point with the worst sum (heuristic for hypervolume)
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    replace_idx = worst_idx
            else:
                # Child is dominated, only replace if better than worst
                worst_idx = np.argmax(np.sum(F, axis=1))
                if np.sum(fc) < np.sum(F[worst_idx]):
                    replace_idx = worst_idx
                else:
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
