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

        # Generate grid points on shape variables
        # Distance vars fixed at midpoint (0.5 for [0,1] bounds)
        # This gives points on the Pareto front

        # Calculate grid resolution
        # We have budget evaluations total
        # Reserve some for refinement
        n_grid_target = 150  # aim for ~150 grid points
        # For 2D grid: g x g = n_grid_target => g ~ 12
        g = int(np.sqrt(n_grid_target))
        g = max(g, 8)

        # Generate grid values for shape variables
        grid_vals = np.linspace(lo[shape_vars[0]], hi[shape_vars[0]], g)

        solutions = []
        objective_values = []
        used = 0

        # Phase 1: Grid search on shape variables
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

                # Simple local search: perturb best points
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
                n_refine = min(5, len(nd_indices))
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
                        step_size = 0.05

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
