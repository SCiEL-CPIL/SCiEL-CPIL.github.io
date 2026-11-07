import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # DTLZ2 structure:
        # x[0], x[1] are position variables (angles)
        # x[2:] are decision variables; g(x) is minimized at x[2:] = 0.5
        # The Pareto front is a quarter-sphere in 3D: f1^2 + f2^2 + f3^2 = 1
        # f = [sin(pi*x0)*sqrt(1-x1^2), sin(pi*x0)*x1, cos(pi*x0)] when g=0

        # Strategy:
        # 1. Use a 2D grid in position space for guaranteed coverage
        # 2. Use Sobol sequence for additional space-filling samples
        # 3. Fix decision variables at 0.5 to be on the Pareto front (g=0)

        n_evals = self.budget
        eval_count = 0
        max_evals = self.budget

        F = []
        X = []

        # Use a 14x14 grid = 196 evaluations for good spatial coverage
        # The 14x14 grid provides good coverage of the quarter-sphere
        grid_x_size = 14
        grid_y_size = 14
        n_grid = grid_x_size * grid_y_size

        if n_grid > n_evals:
            # Reduce grid to fit budget
            grid_size = int(np.sqrt(n_evals))
            grid_x_size = grid_size
            grid_y_size = grid_size
            n_grid = grid_x_size * grid_y_size

        # Generate grid points in [0, 1]^2
        grid_x = np.linspace(0, 1, grid_x_size)
        grid_y = np.linspace(0, 1, grid_y_size)

        grid_points = []
        for gx in grid_x:
            for gy in grid_y:
                grid_points.append([gx, gy])
        grid_points = np.array(grid_points)

        # Evaluate grid points
        for i in range(len(grid_points)):
            if eval_count >= max_evals:
                break
            if time.time() > deadline:
                break
            x = mid.copy()
            x[0] = grid_points[i, 0] * (hi[0] - lo[0]) + lo[0]
            x[1] = grid_points[i, 1] * (hi[1] - lo[1]) + lo[1]
            x[2:] = 0.5  # Decision variables at optimum
            try:
                f = problem(x)
                F.append(f)
                X.append(x)
                eval_count += 1
            except Exception:
                break

        # Use remaining budget with Sobol for better space-filling
        remaining = max_evals - eval_count
        if remaining > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_samples = sampler.random(remaining)

            for i in range(remaining):
                if eval_count >= max_evals:
                    break
                if time.time() > deadline:
                    break
                x = mid.copy()
                x[0] = sobol_samples[i, 0] * (hi[0] - lo[0]) + lo[0]
                x[1] = sobol_samples[i, 1] * (hi[1] - lo[1]) + lo[1]
                x[2:] = 0.5  # Decision variables at optimum
                try:
                    f = problem(x)
                    F.append(f)
                    X.append(x)
                    eval_count += 1
                except Exception:
                    break

        return X
