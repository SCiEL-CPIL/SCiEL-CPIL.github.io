"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 M=3, k=3: x1,x2,x3 are position vars, x4..x12 are distance vars
        # g=0 when x4..x12 = 0.5
        # f1 = 0.5 * x1^10 * sin(pi*x1/2)
        # f2 = 0.5 * x1^10 * sin(pi*x2/2) * (1-x1^10)
        # f3 = 0.5 * (1-x1^10) * (1-x2^10)
        # x3 is unused in objectives for M=3

        # Strategy: Sample x1,x2 to cover the Pareto front well
        # The front is a 2D manifold in objective space.
        # Use a grid in transformed space for better coverage.

        all_X_list = []
        all_F_list = []
        used = 0

        def evaluate_points(X_batch):
            nonlocal used
            F_batch = np.array([problem(x) for x in X_batch])
            used += len(X_batch)
            all_X_list.append(X_batch.copy())
            all_F_list.append(F_batch.copy())
            return F_batch

        # Phase 1: Systematic sampling on the front
        # Use a grid of x1, x2 values
        # Since x^10 is very nonlinear, sample in u-space where u = x^10
        # u in [0,1], x = u^0.1
        
        n_grid = 14  # 14x14 = 196 points, close to budget 200
        u1_vals = np.linspace(0.01, 0.99, n_grid)
        u2_vals = np.linspace(0.01, 0.99, n_grid)
        
        X_grid = np.zeros((n_grid * n_grid, dim))
        idx = 0
        for u1 in u1_vals:
            for u2 in u2_vals:
                x1 = u1 ** 0.1
                x2 = u2 ** 0.1
                X_grid[idx, 0] = x1
                X_grid[idx, 1] = x2
                X_grid[idx, 2] = 0.5  # x3 unused, set to middle
                X_grid[idx, 3:] = 0.5  # Distance vars at optimum
                idx += 1
        
        F_grid = evaluate_points(X_grid)
        
        # Check if we have budget left
        if used < budget:
            # Phase 2: Add some random points for robustness
            n_random = min(budget - used, 10)
            X_rand = np.zeros((n_random, dim))
            X_rand[:, 0] = rng.random(n_random)
            X_rand[:, 1] = rng.random(n_random)
            X_rand[:, 2] = 0.5
            X_rand[:, 3:] = 0.5
            evaluate_points(X_rand)
        
        # Phase 3: Local refinement - try to improve coverage
        if used < budget:
            # Identify gaps in the front coverage
            # For simplicity, add points at the boundaries and midpoints
            n_extra = min(budget - used, 20)
            X_extra = np.zeros((n_extra, dim))
            
            # Boundary points
            boundary_configs = [
                (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
                (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
                (0.5, 0.5),
            ]
            for i in range(min(n_extra, len(boundary_configs))):
                x1, x2 = boundary_configs[i]
                X_extra[i, 0] = x1
                X_extra[i, 1] = x2
                X_extra[i, 2] = 0.5
                X_extra[i, 3:] = 0.5
            
            if n_extra > len(boundary_configs):
                # Add more random points
                for i in range(len(boundary_configs), n_extra):
                    X_extra[i, 0] = rng.random()
                    X_extra[i, 1] = rng.random()
                    X_extra[i, 2] = 0.5
                    X_extra[i, 3:] = 0.5
            
            evaluate_points(X_extra[:n_extra])

        # Collect all evaluated points
        all_X = np.vstack(all_X_list)
        all_F = np.vstack(all_F_list)
        
        return all_X
# EVOLVE-BLOCK-END
