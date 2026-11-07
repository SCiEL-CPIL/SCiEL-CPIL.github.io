import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # ZDT3: f1 = x[0], f2 depends on all
        # Sample x[0] uniformly, and for each, try to minimize f2
        # Use remaining budget for random perturbations around x_i=0.5
        
        n_grid = 20  # number of x[0] values to sample
        per_grid = max(1, (self.budget - n_grid) // n_grid)
        
        all_points = []
        all_F = []
        
        for i in range(n_grid):
            x0 = lo[0] + (hi[0] - lo[0]) * (i + 0.5) / n_grid
            # Base point: x[0]=x0, rest=0.5
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_points.append(x.copy())
            all_F.append(f)
            
            # Use remaining budget for this x0: random search in other dims
            for j in range(per_grid - 1):
                # Perturb random subset of variables
                x_try = x.copy()
                mask = rng.random(self.dim) < 0.3
                x_try[mask] = lo[mask] + (hi[mask] - lo[mask]) * rng.random(np.sum(mask))
                x_try[0] = x0  # keep x[0] fixed
                x_try = np.clip(x_try, lo, hi)
                f_try = problem(x_try)
                all_points.append(x_try)
                all_F.append(f_try)
        
        return all_points
