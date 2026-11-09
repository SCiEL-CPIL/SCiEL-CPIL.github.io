"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin to avoid timeout (300s limit, target 290s)
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 with M=3: first M-1=2 variables determine the shape (position on front)
        # The remaining N-M+1=11 variables determine the distance to the front.
        shape_dims = 2
        dist_dims = self.dim - shape_dims
        
        evals_used = 0
        
        def eval_point(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            evals_used += 1
            return f

        # Phase 1: Optimize the distance variables.
        # In DTLZ4, the distance variables are identical in their contribution to g.
        # We assume the optimal value is the same for all distance dimensions.
        # We search for the optimal scalar value v such that x_dist = [v, v, ...] minimizes g.
        
        # Initial guess based on DTLZ4 transformation x^20 = 0.5
        best_v = 0.5 ** (1.0 / 20.0)
        best_val = float('inf')
        
        # Coarse search
        search_range = np.linspace(0.8, 1.0, 10)
        for v in search_range:
            if time.time() > deadline or evals_used >= self.budget:
                break
            x_test = np.concatenate(([0.5, 0.5], np.full(dist_dims, v)))
            f_test = eval_point(x_test)
            if f_test is None:
                break
            val = np.sum(f_test)
            if val < best_val:
                best_val = val
                best_v = v
        
        # Fine search around best_v
        for step in [0.01, 0.001, 0.0001]:
            for delta in [-step, 0, step]:
                if time.time() > deadline or evals_used >= self.budget:
                    break
                v_cand = best_v + delta
                if v_cand < lo[shape_dims] or v_cand > hi[shape_dims]:
                    continue
                x_test = np.concatenate(([0.5, 0.5], np.full(dist_dims, v_cand)))
                f_test = eval_point(x_test)
                if f_test is None:
                    break
                val = np.sum(f_test)
                if val < best_val:
                    best_val = val
                    best_v = v_cand
        
        # Phase 2: Sample the shape variables (x0, x1) to cover the Pareto front.
        remaining_budget = self.budget - evals_used
        if remaining_budget <= 0:
            # If no budget left, return the single point we have
            x_final = np.concatenate(([0.5, 0.5], np.full(dist_dims, best_v)))
            return [x_final]
        
        points = []
        
        # Use a grid for the first part to ensure uniform coverage
        # Approximate sqrt(remaining_budget * 0.5) for grid size
        grid_size = max(1, int(np.sqrt(remaining_budget * 0.3)))
        
        # Add grid points
        grid_x0 = np.linspace(0.0, 1.0, grid_size)
        grid_x1 = np.linspace(0.0, 1.0, grid_size)
        for x0 in grid_x0:
            for x1 in grid_x1:
                points.append((x0, x1))
                
        # Fill remaining budget with random points
        num_random = remaining_budget - len(points)
        if num_random > 0:
            rand_x0 = np.random.rand(num_random)
            rand_x1 = np.random.rand(num_random)
            for i in range(num_random):
                points.append((rand_x0[i], rand_x1[i]))
        
        # Truncate if we generated too many (shouldn't happen if calc is correct)
        points = points[:remaining_budget]
        
        all_x = []
        all_f = []
        
        for x0, x1 in points:
            if time.time() > deadline or evals_used >= self.budget:
                break
            x = np.concatenate(([x0, x1], np.full(dist_dims, best_v)))
            f = eval_point(x)
            if f is None:
                break
            all_x.append(x)
            all_f.append(f)
            
        # If we didn't use all budget, we could do a local polish, 
        # but with DTLZ4 structure, the fixed dist is usually sufficient.
        
        # Filter non-dominated points
        if not all_f:
            return []
            
        F = np.array(all_f)
        X = np.array(all_x)
        
        non_dom_indices = []
        n = len(F)
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                non_dom_indices.append(i)
        
        return X[non_dom_indices]
# EVOLVE-BLOCK-END
