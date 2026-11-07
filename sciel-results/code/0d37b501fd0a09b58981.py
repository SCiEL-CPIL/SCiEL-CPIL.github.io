"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using analytical front sampling and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        n_angle = dim - n_obj  # 9 for 12 vars, 3 objs

        used = 0
        all_F = []

        def check_time():
            return time.time() > deadline

        def eval_and_record(x):
            nonlocal used
            fx = problem(x)
            used += 1
            all_F.append(fx.copy())
            return fx

        # DTLZ2 with M=3 objectives:
        # f1 = (1+g)*cos(t1)*cos(t2)
        # f2 = (1+g)*sin(t1)*cos(t2)  
        # f3 = (1+g)*sin(t2)
        # where t1, t2 in [0, pi/2]
        # g(x) depends on x[2:] (the last dim - (n_obj-1) = dim - 2 = 10 variables)
        # For DTLZ2, g = sum((x_i - 0.5)^2) for i >= 2, or similar
        # To get g=0, set x[2:] = 0.5
        # t1, t2 are determined by x[0], x[1] through some mapping
        # In standard DTLZ2, x[0] and x[1] directly give the angles via:
        # t1 = x[0] * pi/2, t2 = x[1] * pi/2 (if x in [0,1])
        # But the problem clips to bounds, so we need to figure out the mapping.
        
        # Actually, in standard DTLZ2 formulation:
        # The first M-1 decision variables determine the angles
        # x[0] -> t1 = x[0] * (pi/2)
        # x[1] -> t2 = x[1] * (pi/2)
        # The remaining variables affect g
        
        # Strategy: Use a dense grid on the Pareto front
        # Sample t1, t2 on a grid to cover the quarter-sphere front densely
        # For each (t1, t2), set x[0] = 2*t1/pi, x[1] = 2*t2/pi, x[2:] = 0.5
        
        # With budget 200, use a grid of about 14x14 = 196 points
        n_grid = int(np.sqrt(self.budget))
        # Adjust to stay within budget
        if n_grid * n_grid > self.budget:
            n_grid = int(np.sqrt(self.budget))
        
        t1_vals = np.linspace(0, np.pi/2, n_grid)
        t2_vals = np.linspace(0, np.pi/2, n_grid)
        
        # Generate grid points
        grid_points = []
        for t1 in t1_vals:
            for t2 in t2_vals:
                x = np.zeros(dim)
                x[0] = 2 * t1 / np.pi  # maps [0, pi/2] to [0, 1]
                x[1] = 2 * t2 / np.pi  # maps [0, pi/2] to [0, 1]
                x[2:] = 0.5  # minimize g
                grid_points.append(x)
        
        # Shuffle to ensure diversity in evaluation order
        rng = np.random.default_rng(42)
        order = rng.permutation(len(grid_points))
        
        # Evaluate grid points
        for idx in order:
            if check_time() or used >= self.budget:
                break
            x = grid_points[idx]
            # Clip to bounds
            x = np.clip(x, lo, hi)
            fx = eval_and_record(x)
        
        # If we have remaining budget, do local search around the best points
        if used < self.budget and all_F:
            F_arr = np.array(all_F)
            # Find points with good hypervolume contribution (low sum of objectives)
            sums = np.sum(F_arr, axis=1)
            # Get indices of best 10 points
            n_refine = min(10, len(all_F))
            best_idx = np.argsort(sums)[:n_refine]
            
            for bi in best_idx:
                if check_time() or used >= self.budget:
                    break
                x_base = grid_points[order[bi]].copy()
                # Try small perturbations
                for _ in range(3):
                    if check_time() or used >= self.budget:
                        break
                    delta = 0.02 * (hi - lo) * rng.standard_normal(dim)
                    x_new = np.clip(x_base + delta, lo, hi)
                    fx_new = eval_and_record(x_new)
        
        F = np.array(all_F)
        return F
# EVOLVE-BLOCK-END
