"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a deterministic grid over the first 3 variables (which directly determine
    the Pareto front) with remaining variables set to lower bound, followed by
    a directed local search phase to refine the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9: the first 3 variables determine the Pareto front shape.
        # Variables 3-11 (indices) affect distance to the front.
        # Setting them to lower bound minimizes distance penalty.
        # Budget is 200.
        # Grid size g=5 gives 125 points.
        # Remaining 75 points for local search.
        
        g = 5
        grid_points = []
        for i in range(g):
            for j in range(g):
                for k in range(g):
                    x = np.zeros(self.dim)
                    x[0] = lo[0] + (hi[0] - lo[0]) * i / (g - 1)
                    x[1] = lo[1] + (hi[1] - lo[1]) * j / (g - 1)
                    x[2] = lo[2] + (hi[2] - lo[2]) * k / (g - 1)
                    # Set distance variables to lower bound (minimizes distance)
                    for d in range(3, self.dim):
                        x[d] = lo[d]
                    grid_points.append(x)
        
        grid_points = np.array(grid_points)
        n_grid = len(grid_points)
        
        # Evaluate grid points
        F = np.zeros((n_grid, self.n_obj))
        for i in range(n_grid):
            if time.time() > deadline:
                break
            F[i] = problem(grid_points[i])
        
        used = n_grid
        
        # Keep track of all evaluated points
        X_all = [grid_points[i].copy() for i in range(n_grid)]
        F_all = [F[i].copy() for i in range(n_grid)]
        
        # Vectorized dominance check helper
        def get_nondominated_indices(F_arr):
            """Return indices of non-dominated points in F_arr (n x m array)."""
            n = len(F_arr)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                # Check if any other point dominates i
                # Point j dominates i if F[j] <= F[i] for all obj and F[j] < F[i] for at least one
                diff = F_arr - F_arr[i]  # shape (n, m)
                # j dominates i if diff[j] <= 0 for all m and diff[j] < 0 for at least one
                le_all = np.all(diff <= 0, axis=1)
                lt_any = np.any(diff < 0, axis=1)
                dom_mask = le_all & lt_any
                dom_mask[i] = False  # exclude self
                if np.any(dom_mask):
                    nondom[i] = False
            return np.where(nondom)[0]

        def is_dominated(f_new, F_arr):
            """Check if f_new is dominated by any point in F_arr."""
            diff = F_arr - f_new
            le_all = np.all(diff <= 0, axis=1)
            lt_any = np.any(diff < 0, axis=1)
            return np.any(le_all & lt_any)

        def dominates_point(f_new, f_old):
            """Check if f_new dominates f_old."""
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        # Local search phase: refine non-dominated points
        # For WFG9, the key is to refine the first 3 (shape) variables.
        # Variables 3-11 should stay at lower bound.
        # Use adaptive step sizes: start large, shrink as we converge.
        
        step_scale = 0.1  # fraction of range
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            F_arr = np.array(F_all)
            nondom_idx = get_nondominated_indices(F_arr)
            
            if len(nondom_idx) == 0:
                break
            
            # Pick a non-dominated point to refine (random selection)
            idx = nondom_idx[np.random.randint(len(nondom_idx))]
            
            x = X_all[idx].copy()
            f = F_all[idx].copy()
            
            # Pick a dimension to perturb
            # Focus on shape variables (0-2) most of the time
            if np.random.random() < 0.7:
                d = np.random.randint(3)
            else:
                d = np.random.randint(self.dim)
            
            # Determine direction and step
            if d < 3:
                # Shape variables: try both directions
                direction = np.random.choice([-1, 1])
                step_size = step_scale * (hi[d] - lo[d])
            else:
                # Distance variables: push towards lower bound
                if x[d] > lo[d]:
                    direction = -1
                    step_size = min(0.5 * (x[d] - lo[d]), 0.1 * (hi[d] - lo[d]))
                else:
                    direction = 1
                    step_size = 0.01 * (hi[d] - lo[d])
                
            x_new = x.copy()
            x_new[d] += direction * step_size
            x_new[d] = np.clip(x_new[d], lo[d], hi[d])
            
            f_new = problem(x_new)
            used += 1
            
            # Check if f_new is useful
            if dominates_point(f_new, f):
                # f_new dominates current point, replace it
                X_all[idx] = x_new
                F_all[idx] = f_new
            elif not is_dominated(f_new, F_arr):
                # f_new is not dominated by any existing point, add it
                X_all.append(x_new)
                F_all.append(f_new)
            
            # Gradually reduce step size for finer refinement
            if used % 10 == 0:
                step_scale *= 0.7
        
        # Final non-dominated set
        final_F = np.array(F_all)
        final_X = np.array(X_all)
        
        return final_X
# EVOLVE-BLOCK-END
