"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based multi-start with Nelder-Mead local optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # Phase 1: Deterministic 3^5 grid search (243 evaluations)
        # Grid points: -5, 0, 5 for each dimension
        grid_levels = np.array([-5.0, 0.0, 5.0])
        grid_points = []
        grid_values = []

        # Generate all 3^5 = 243 grid points
        for i in range(3 ** dim):
            point = []
            tmp = i
            for d in range(dim):
                point.append(grid_levels[tmp % 3])
                tmp //= 3
            grid_points.append(np.array(point, dtype=np.float64))

        # Evaluate all grid points
        for i, pt in enumerate(grid_points):
            if time.time() > deadline:
                break
            val = func(pt)
            grid_values.append((val, pt))

        # Sort by function value and take top 5 distinct basins
        grid_values.sort(key=lambda x: x[0])

        # Select top 5 points, ensuring some diversity
        # Use a simple approach: take the 5 best, but if they're too close, skip
        selected_starts = []
        selected_values = []
        min_distance = 1.0  # minimum distance between selected starts

        for val, pt in grid_values:
            if len(selected_starts) >= 5:
                break
            # Check if this point is sufficiently different from already selected
            is_diverse = True
            for existing_pt in selected_starts:
                if np.linalg.norm(pt - existing_pt) < min_distance:
                    is_diverse = False
                    break
            if is_diverse:
                selected_starts.append(pt)
                selected_values.append(val)

        # If we don't have 5 diverse points, fill with the best remaining
        if len(selected_starts) < 5:
            for val, pt in grid_values:
                if len(selected_starts) >= 5:
                    break
                if pt not in selected_starts:
                    selected_starts.append(pt)
                    selected_values.append(val)

        # Track best found so far
        best_x = selected_starts[0]
        best_f = selected_values[0]
        calls_made = len(grid_values)

        # Phase 2: Nelder-Mead local optimization from each starting point
        # Distribute remaining budget among the 5 starts
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # Use a custom Nelder-Mead implementation to avoid scipy overhead
        # and have better control over the number of function evaluations
        starts_per_opt = max(1, remaining_budget // len(selected_starts))

        for start_idx, (start_pt, start_val) in enumerate(zip(selected_starts, selected_values)):
            if time.time() > deadline:
                break

            # Nelder-Mead implementation
            x = start_pt.copy()
            fx = start_val

            # Initialize simplex
            simplex = [x.copy()]
            simplex_values = [fx]

            # Add perturbed points
            for d in range(dim):
                new_point = x.copy()
                perturbation = 0.1 if abs(x[d]) < 0.1 else 0.1 * abs(x[d])
                new_point[d] += perturbation
                new_point = np.clip(new_point, -5.0, 5.0)
                if new_point is not x:
                    f_new = func(new_point)
                    simplex.append(new_point)
                    simplex_values.append(f_new)
                    calls_made += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = new_point.copy()

            # Nelder-Mead iterations
            max_iterations = min(starts_per_opt, 100)
            for iteration in range(max_iterations):
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_values)
                simplex = [simplex[i] for i in order]
                simplex_values = [simplex_values[i] for i in order]

                # Check convergence
                if len(simplex) > 1:
                    spread = max(simplex_values) - min(simplex_values)
                    if spread < 1e-10:
                        break

                # Reflection
                centroid = np.mean([simplex[i] for i in range(len(simplex) - 1)], axis=0)
                x_reflect = centroid + 1.0 * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = func(x_reflect)
                calls_made += 1

                if f_reflect < simplex_values[0]:
                    # Expansion
                    x_expand = centroid + 2.0 * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = func(x_expand)
                    calls_made += 1
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_values[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_values[-1] = f_reflect
                elif f_reflect < simplex_values[-2]:
                    simplex[-1] = x_reflect
                    simplex_values[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = centroid + 0.5 * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = func(x_contract)
                    calls_made += 1
                    if f_contract < simplex_values[-1]:
                        simplex[-1] = x_contract
                        simplex_values[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_shrink = func(simplex[i])
                            calls_made += 1
                            simplex_values[i] = f_shrink

                # Update best
                if simplex_values[0] < best_f:
                    best_f = simplex_values[0]
                    best_x = simplex[0].copy()

                # If we've used most of the budget, stop
                if calls_made >= budget - 5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
