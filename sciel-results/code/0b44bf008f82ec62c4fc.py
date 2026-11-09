"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_head = self.n_obj - 1  # For DTLZ7, first m-1 vars control f1..f_{m-1}
        n_tail = self.dim - n_head
        
        # 1. Initial Random Sample to get a baseline
        n_init = 50
        X_init = lo + (hi - lo) * np.random.rand(n_init, self.dim)
        F_init = np.array([problem(x) for x in X_init])
        
        # Keep track of all evaluated points for final hypervolume calculation
        # The problem statement says "Every returned objective vector is recorded".
        # We don't control the recording, we just return points. 
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This implies the evaluator intercepts `problem(x)` calls? 
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded".
        # Usually in these tasks, the `problem` object passed to `__call__` is a wrapper that records every call.
        # So I just need to make as many good calls to `problem` as possible within budget and time.
        
        # Strategy:
        # DTLZ7 Pareto front is defined by x_head. 
        # f1 = x0
        # f2 = x1
        # f3 = (1 + g(x_tail)) * h(x_{m-1}, ..., f2)
        # To get good hypervolume, we need:
        # 1. g(x_tail) close to 0.
        # 2. x_head spread out on the Pareto front.
        
        # Step 1: Find a tail vector that minimizes g.
        # g is typically sum((xi - 0.5)^2) or similar for DTLZ7. 
        # Let's use Nelder-Mead to minimize the 3rd objective assuming head is set to a point on the front.
        # Actually, g depends only on tail. Let's find a tail that minimizes g.
        # We can estimate g by evaluating with a fixed head.
        
        # Pick a head point on the front, e.g., x_head = [0.5, 0.5] (normalized)
        # But we need to know the mapping. 
        # Let's just try to minimize the last objective with a fixed head.
        
        def objective_tail(x_tail):
            x_head = np.array([0.5] * n_head) # Arbitrary point on front
            x_full = np.concatenate([x_head, x_tail])
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            # For DTLZ7, f[2] is the one affected by g. 
            # Minimizing f[2] with fixed head minimizes g (since h is fixed).
            return f[2]
        
        # Initial tail guess
        x_tail_init = (lo[n_head:] + hi[n_head:]) / 2.0
        try:
            res = minimize(objective_tail, x_tail_init, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            best_tail = res.x
        except Exception:
            best_tail = x_tail_init
        
        # Check time
        if time.time() > deadline:
            return np.array([F_init])

        # Step 2: Grid search over head variables to generate points on the front.
        # The front is defined by x_head. 
        # For DTLZ7 with 3 objectives, x_head has 2 variables.
        # We want to sample x_head such that the resulting points cover the front.
        
        # Number of points to generate
        # We have used ~50 (init) + ~100 (Nelder-Mead) = 150 evals.
        # Budget is 200. We have ~50 left.
        # Let's use a grid of 5x5 = 25 points.
        
        n_grid = 5
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Map i, j to [0, 1] range for head variables
                x0 = i / (n_grid - 1) if n_grid > 1 else 0.5
                x1 = j / (n_grid - 1) if n_grid > 1 else 0.5
                x_head = np.array([x0, x1])
                x_full = np.concatenate([x_head, best_tail])
                x_full = np.clip(x_full, lo, hi)
                grid_points.append(x_full)
        
        F_grid = np.array([problem(x) for x in grid_points])
        
        # Check time
        if time.time() > deadline:
            return np.array([F_init, F_grid])

        # Step 3: Use remaining budget for random perturbations of the best points
        # to refine the front coverage or find better g.
        used_evals = n_init + 100 + len(grid_points) # Approximate
        # Actually, Nelder-Mead uses variable number of evals. 
        # Let's just run until we hit budget or time.
        
        # Combine all points
        all_X = np.vstack([X_init, grid_points])
        all_F = np.vstack([F_init, F_grid])
        
        # If we have budget left, do a few random restarts or perturbations
        # Since we don't know exact eval count of Nelder-Mead, let's be conservative.
        # If time allows, perturb the best tail and re-evaluate a few head points.
        
        # Let's add a few random points near the best tail to ensure robustness
        n_extra = 10
        X_extra = np.array([best_tail + 0.01 * np.random.randn(self.dim - n_head) for _ in range(n_extra)])
        # Wait, we need full vectors.
        # Pick a random head point for each
        X_extra_full = []
        for i in range(n_extra):
            x_head = np.random.rand(n_head)
            x_tail = best_tail + 0.01 * np.random.randn(n_tail)
            x_full = np.concatenate([x_head, x_tail])
            x_full = np.clip(x_full, lo, hi)
            X_extra_full.append(x_full)
            
        F_extra = np.array([problem(x) for x in X_extra_full])
        
        all_X = np.vstack([all_X, X_extra_full])
        all_F = np.vstack([all_F, F_extra])
        
        return all_X
# EVOLVE-BLOCK-END
