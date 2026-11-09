"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a deterministic grid over the first 3 variables (which directly determine
    the Pareto front) with remaining variables at midpoint, followed by a local
    search phase to refine the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9: the first 3 variables determine the Pareto front.
        # We'll use a grid over the first 3 variables and fix the rest at midpoint.
        # Budget is 200.
        # Grid size g=5 gives 125 points.
        # Remaining 75 points for local search.
        
        g = 5
        grid_points = []
        for i in range(g):
            for j in range(g):
                for k in range(g):
                    x = np.full(self.dim, 0.0)
                    x[0] = lo[0] + (hi[0] - lo[0]) * i / (g - 1)
                    x[1] = lo[1] + (hi[1] - lo[1]) * j / (g - 1)
                    x[2] = lo[2] + (hi[2] - lo[2]) * k / (g - 1)
                    for d in range(3, self.dim):
                        x[d] = mid[d]
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
        X_all = [grid_points[i] for i in range(n_grid)]
        F_all = [F[i] for i in range(n_grid)]
        
        # Local search phase: refine non-dominated points
        # We will perform coordinate descent on the non-dominated points.
        # For variables 0-2, we try steps in both directions.
        # For variables 3-11, we bias steps towards 0 (lower bound) as it typically minimizes distance in WFG.
        
        # Helper to check if a point is dominated by any in the archive
        def is_dominated_by_archive(f_new, F_archive):
            for f in F_archive:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        # Helper to check if a point is dominated by a specific point
        def is_dominated_by(f_new, f_old):
            return np.all(f_old <= f_new) and np.any(f_old < f_new)

        # Helper to check if f_new dominates f_old
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        # Local search loop
        # We have 75 evaluations left.
        # We will iterate through non-dominated points and try to improve them.
        
        # To make the search more effective, we can shuffle the order of non-dominated points
        # or just cycle through them.
        
        # Determine step size.
        # We want to cover the range.
        # Let's use a fixed small step for refinement.
        # step = 0.1 * (hi - lo)
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            nondom_idx = []
            for i in range(len(F_all)):
                dominated = False
                for j in range(len(F_all)):
                    if i != j:
                        if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                            dominated = True
                            break
                if not dominated:
                    nondom_idx.append(i)
            
            if not nondom_idx:
                break
            
            # Pick a non-dominated point to refine
            # We can pick the one with the best "potential" or just cycle.
            # Let's cycle through them.
            # To avoid getting stuck, we can pick randomly from the non-dominated set.
            idx = nondom_idx[np.random.randint(len(nondom_idx))]
            
            x = X_all[idx].copy()
            f = F_all[idx].copy()
            
            # Try a few random coordinate steps
            # We have budget to spend. Let's try one step per loop iteration.
            
            # Pick a random dimension
            d = np.random.randint(self.dim)
            
            # Determine direction
            if d < 3:
                # Shape variables: try both directions
                direction = np.random.choice([-1, 1])
                step_size = 0.1 * (hi[d] - lo[d])
            else:
                # Distance variables: bias towards lower bound (0)
                # If current value is > lo, try moving towards lo.
                # If current value is < hi, try moving towards hi?
                # Usually 0 is better. Let's try moving towards lo.
                if x[d] > lo[d]:
                    direction = -1
                else:
                    direction = 1
                step_size = 0.1 * (hi[d] - lo[d])
                
            x_new = x.copy()
            x_new[d] += direction * step_size
            x_new[d] = np.clip(x_new[d], lo[d], hi[d])
            
            f_new = problem(x_new)
            used += 1
            
            # Check if f_new is useful
            # If f_new dominates f, replace.
            if dominates(f_new, f):
                X_all[idx] = x_new
                F_all[idx] = f_new
            elif not is_dominated_by(f_new, f):
                # If f_new is not dominated by f, and not dominated by the rest of the archive, add it.
                # Check against archive
                dominated_by_archive = False
                for j in range(len(F_all)):
                    if j != idx:
                        if np.all(F_all[j] <= f_new) and np.any(F_all[j] < f_new):
                            dominated_by_archive = True
                            break
                if not dominated_by_archive:
                    X_all.append(x_new)
                    F_all.append(f_new)
        
        # Final non-dominated set
        final_F = np.array(F_all)
        final_X = np.array(X_all)
        
        return final_X
# EVOLVE-BLOCK-END
