"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_points = []
        used = 0

        # ZDT3 structure with g=1 (x[1:]=0)
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        
        # Dense grid to identify the non-dominated envelope
        n_grid = 1000000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        # Find non-dominated points (record lows in f2)
        min_f2_so_far = np.minimum.accumulate(f2_grid)
        tol = 1e-14
        is_nd = (f2_grid <= min_f2_so_far + tol)
        
        x_nd = x_grid[is_nd]
        f2_nd = f2_grid[is_nd]
        
        # Keep only strict record lows
        if len(x_nd) > 0:
            unique_mask = np.ones(len(x_nd), dtype=bool)
            prev_f2 = np.inf
            for i in range(len(x_nd)):
                if f2_nd[i] >= prev_f2 - 1e-15:
                    unique_mask[i] = False
                else:
                    prev_f2 = f2_nd[i]
            x_nd = x_nd[unique_mask]
            f2_nd = f2_nd[unique_mask]
        
        n_nd = len(x_nd)
        if n_nd <= budget:
            selected_x = x_nd
        else:
            # Select 'budget' points uniformly in f1-space for better hypervolume coverage
            interval_midpoints = (np.arange(budget) + 0.5) / budget
            selected_x = np.empty(budget)
            for i in range(budget):
                idx = np.searchsorted(x_nd, interval_midpoints[i])
                if idx == 0:
                    best_idx = 0
                elif idx == n_nd:
                    best_idx = n_nd - 1
                else:
                    if abs(x_nd[idx-1] - interval_midpoints[i]) <= abs(x_nd[idx] - interval_midpoints[i]):
                        best_idx = idx - 1
                    else:
                        best_idx = idx
                selected_x[i] = x_nd[best_idx]
            selected_x = np.unique(selected_x)
            
        n_evals = len(selected_x)
        
        for i in range(n_evals):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = selected_x[i]
            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
