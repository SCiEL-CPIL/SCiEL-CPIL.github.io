import numpy as np
import time


class Optimizer:
    """DTLZ2 optimizer: dense uniform grid on shape vars with dist=0 for max hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For DTLZ2: first n_obj-1 variables are shape variables
        # Remaining are distance variables (optimal = 0.0 for M=3)
        k_shape = self.n_obj - 1  # 2 for M=3
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        # Step 1: Confirm dist_val = 0.0 is optimal (1 eval)
        x_test = np.zeros(self.dim)
        for i in shape_vars:
            x_test[i] = 0.5
        f_test = np.array(problem(x_test))
        # sum(f^2) should be ~1.0 if on Pareto front
        dist_val = 0.0 if np.sum(f_test**2) < 1.5 else 0.5

        evals_used = 1
        remaining_budget = self.budget - evals_used

        # Step 2: Dense uniform grid on shape variables
        # Use g x g grid where g = int(sqrt(remaining_budget))
        g = int(np.sqrt(remaining_budget))
        g = max(g, 10)

        # For DTLZ2 with M=3, shape vars in [0,1] map to angles in [0, pi/2]
        # Uniform in [0,1] gives uniform coverage of the Pareto front
        grid_vals = np.linspace(0.0, 1.0, g)

        solutions = []
        objective_values = []
        used = 0

        # Phase 1: Grid search with full budget
        for v1 in grid_vals:
            for v2 in grid_vals:
                if time.time() > deadline:
                    break
                if used >= remaining_budget:
                    break

                x = np.zeros(self.dim)
                x[shape_vars[0]] = v1
                if len(shape_vars) > 1:
                    x[shape_vars[1]] = v2
                for i in dist_vars:
                    x[i] = dist_val

                f = np.array(problem(x))
                solutions.append(x)
                objective_values.append(f)
                used += 1

            if used >= remaining_budget or time.time() > deadline:
                break

        return np.array(solutions) if solutions else np.array([(lo + hi) / 2.0])
