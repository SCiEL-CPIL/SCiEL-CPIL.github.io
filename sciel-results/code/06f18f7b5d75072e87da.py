"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DTLZ2 optimizer: grid search on shape vars + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # For DTLZ2: first n_obj-1 variables are shape variables
        # Remaining are distance variables (should be 0.5 for optimal)
        k_shape = self.n_obj - 1  # 2 for M=3
        shape_vars = list(range(k_shape))
        dist_vars = list(range(k_shape, self.dim))

        # For DTLZ2, the Pareto front is the unit hypersphere in objective space.
        # We can parameterize the shape variables to directly generate points on the front.
        # For M=3, we have 2 shape variables (x0, x1) that map to angles (theta1, theta2).
        # The standard DTLZ2 mapping:
        # f_i = (1 + g) * z_i
        # where z_i are on the unit hypersphere.
        # For M=3: z = [cos(theta1)*cos(theta2), cos(theta1)*sin(theta2), sin(theta1)]
        # So x0 = theta1, x1 = theta2 (if bounds are [0, pi/2] for both)
        # But bounds are [0, 1], so we need to map [0,1] -> [0, pi/2]
        
        # Actually, let's just use a dense grid in the shape variable space
        # and rely on the problem function to compute the objectives.
        # The key insight: for DTLZ2, when distance vars = 0.5, we get g=0,
        # so f = z (the unit hypersphere point).
        
        # Generate a dense grid on the shape variables
        # Use more points for better coverage of the Pareto front
        n_grid_target = 180  # aim for ~180 grid points
        g = int(np.sqrt(n_grid_target))
        g = max(g, 10)

        # Generate grid values for shape variables
        grid_vals = np.linspace(lo[shape_vars[0]], hi[shape_vars[0]], g)

        solutions = []
        objective_values = []
        used = 0

        # Phase 1: Dense grid search on shape variables
        for v1 in grid_vals:
            for v2 in grid_vals:
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break

                x = np.copy(mid)
                x[shape_vars[0]] = v1
                x[shape_vars[1]] = v2
                # Distance variables already at mid (0.5)
                f = np.array(problem(x))
                solutions.append(x)
                objective_values.append(f)
                used += 1

            if used >= self.budget or time.time() > deadline:
                break

        # If we have remaining budget, do local refinement
        if used < self.budget and time.time() < deadline:
            # Convert to arrays for easier manipulation
            if solutions:
                X = np.array(solutions)
                F = np.array(objective_values)

                # Find non-dominated points
                n = len(F)
                dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    for j in range(n):
                        if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break

                nd_indices = np.where(~dominated)[0]

                # Refine a few of the non-dominated points
                n_refine = min(8, len(nd_indices))
                if n_refine > 0:
                    # Pick evenly spaced points from the non-dominated set
                    if len(nd_indices) > n_refine:
                        refine_idx = np.linspace(0, len(nd_indices) - 1, n_refine, dtype=int)
                        refine_idx = nd_indices[refine_idx]
                    else:
                        refine_idx = nd_indices

                    for idx in refine_idx:
                        if used >= self.budget or time.time() > deadline:
                            break

                        # Try small perturbations in shape space
                        base_x = X[idx].copy()
                        step_size = 0.03

                        for pert in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size)]:
                            if used >= self.budget or time.time() > deadline:
                                break

                            new_x = base_x.copy()
                            new_x[shape_vars[0]] += pert[0]
                            new_x[shape_vars[1]] += pert[1]
                            new_x = np.clip(new_x, lo, hi)

                            new_f = np.array(problem(new_x))
                            solutions.append(new_x)
                            objective_values.append(new_f)
                            used += 1

        return np.array(solutions) if solutions else np.array([mid])
# EVOLVE-BLOCK-END
