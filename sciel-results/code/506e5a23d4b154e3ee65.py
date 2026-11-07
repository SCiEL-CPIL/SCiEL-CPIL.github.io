"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimization with multi-start and adaptive polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution found overall
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_nm = 0.5 # shrink

        # Strategy:
        # 1. Multi-start with diverse initial points to explore the landscape
        # 2. Run Nelder-Mead from each start, keeping the best result
        # 3. Use remaining budget for adaptive local polishing

        # Budget allocation:
        # - 50% for multi-start exploration (multiple NM runs)
        # - 30% for refinement from best point
        # - 20% for final polishing

        explore_budget = int(budget * 0.5)
        refine_budget = int(budget * 0.3)
        polish_budget = budget - explore_budget - refine_budget

        # Multi-start: try several different initial centers
        num_starts = 5
        start_budget_per = explore_budget // num_starts

        # Generate diverse starting points
        start_points = []
        for i in range(num_starts):
            if i == 0:
                # First start: center of the box
                center = np.zeros(dim)
            elif i == 1:
                # Second start: random uniform
                center = rng.uniform(-4.0, 4.0, dim)
            else:
                # Subsequent starts: mix of random and near-center
                if i % 2 == 0:
                    center = rng.uniform(-3.0, 3.0, dim)
                else:
                    center = rng.normal(0, 1.5, dim)
            center = np.clip(center, -4.0, 4.0)
            start_points.append(center)

        # Run Nelder-Mead from each start point
        for start_idx, center in enumerate(start_points):
            if calls_made >= explore_budget:
                break

            # Allocate budget for this start
            if start_idx < num_starts - 1:
                nm_budget = min(start_budget_per, explore_budget - calls_made)
            else:
                nm_budget = explore_budget - calls_made

            if nm_budget < dim + 2:
                break

            # Create simplex around the start point
            step = 1.5  # Initial step size
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center.copy()
                simplex[i + 1][i] += step

            # Evaluate initial simplex
            fx_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= explore_budget:
                    break
                fx_values[i] = evaluate(simplex[i])

            if calls_made >= explore_budget:
                break

            # Nelder-Mead loop
            stagnation_count = 0
            last_best_f = best_f
            converged = False

            while calls_made < explore_budget:
                # Sort simplex
                order = np.argsort(fx_values)
                simplex = simplex[order]
                fx_values = fx_values[order]

                # Convergence check
                f_range = fx_values[-1] - fx_values[0]
                if f_range < 1e-7:
                    converged = True
                    break

                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if calls_made >= explore_budget:
                    break

                if f_r < fx_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)

                    if calls_made >= explore_budget:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        fx_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fx_values[-1] = f_r
                elif f_r < fx_values[-2]:
                    # Reflection
                    simplex[-1] = x_r
                    fx_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < fx_values[-1]:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)

                    if calls_made >= explore_budget:
                        break

                    if f_c < min(f_r, fx_values[-1]):
                        simplex[-1] = x_c
                        fx_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made < explore_budget:
                                fx_values[i] = evaluate(simplex[i])
                        if calls_made >= explore_budget:
                            break

                # Check for improvement
                if best_f < last_best_f - 1e-6:
                    stagnation_count = 0
                    last_best_f = best_f
                else:
                    stagnation_count += 1

                if stagnation_count > 15:
                    break

        # Refinement phase: run NM from best point with smaller steps
        if best_x is not None and calls_made < budget - polish_budget:
            refine_end = budget - polish_budget
            current_step = 0.5
            max_refine_restarts = 10

            for refine_restart in range(max_refine_restarts):
                if calls_made >= refine_end:
                    break

                # Initialize simplex around best point
                center = best_x.copy()
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = center
                for i in range(dim):
                    simplex[i + 1] = center.copy()
                    simplex[i + 1][i] += current_step

                # Evaluate initial simplex
                fx_values = np.zeros(dim + 1)
                for i in range(dim + 1):
                    if calls_made >= refine_end:
                        break
                    fx_values[i] = evaluate(simplex[i])

                if calls_made >= refine_end:
                    break

                # Nelder-Mead loop
                stagnation_count = 0
                last_best_f = best_f
                converged = False

                while calls_made < refine_end:
                    # Sort simplex
                    order = np.argsort(fx_values)
                    simplex = simplex[order]
                    fx_values = fx_values[order]

                    # Convergence check
                    f_range = fx_values[-1] - fx_values[0]
                    if f_range < 1e-8:
                        converged = True
                        break

                    # Centroid of best dim points
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5.0, 5.0)
                    f_r = evaluate(x_r)

                    if calls_made >= refine_end:
                        break

                    if f_r < fx_values[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5.0, 5.0)
                        f_e = evaluate(x_e)

                        if calls_made >= refine_end:
                            break

                        if f_e < f_r:
                            simplex[-1] = x_e
                            fx_values[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fx_values[-1] = f_r
                    elif f_r < fx_values[-2]:
                        # Reflection
                        simplex[-1] = x_r
                        fx_values[-1] = f_r
                    else:
                        # Contraction
                        if f_r < fx_values[-1]:
                            # Outer contraction
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            # Inner contraction
                            x_c = centroid - rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = evaluate(x_c)

                        if calls_made >= refine_end:
                            break

                        if f_c < min(f_r, fx_values[-1]):
                            simplex[-1] = x_c
                            fx_values[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                if calls_made < refine_end:
                                    fx_values[i] = evaluate(simplex[i])
                            if calls_made >= refine_end:
                                break

                    # Check for improvement
                    if best_f < last_best_f - 1e-7:
                        stagnation_count = 0
                        last_best_f = best_f
                    else:
                        stagnation_count += 1

                    if stagnation_count > 10:
                        break

                # Reduce step for next refinement restart
                if converged:
                    current_step *= 0.5
                else:
                    if current_step > 0.1:
                        current_step *= 0.7
                    else:
                        break

        # Final polishing: adaptive coordinate descent with shrinking steps
        if best_x is not None and calls_made < budget:
            step = 0.3
            # Use adaptive step: try larger steps first, then shrink
            while calls_made < budget and step > 1e-6:
                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    # Try positive step
                    x_try = best_x.copy()
                    x_try[i] += step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        improved = True
                        # Try to continue in same direction
                        while calls_made < budget:
                            x_try2 = best_x.copy()
                            x_try2[i] += step
                            x_try2 = np.clip(x_try2, -5.0, 5.0)
                            f_try2 = evaluate(x_try2)
                            if f_try2 < best_f:
                                best_x = x_try2
                                best_f = f_try2
                            else:
                                break

                    if calls_made >= budget:
                        break

                    # Try negative step
                    x_try = best_x.copy()
                    x_try[i] -= step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        improved = True
                        # Try to continue in same direction
                        while calls_made < budget:
                            x_try2 = best_x.copy()
                            x_try2[i] -= step
                            x_try2 = np.clip(x_try2, -5.0, 5.0)
                            f_try2 = evaluate(x_try2)
                            if f_try2 < best_f:
                                best_x = x_try2
                                best_f = f_try2
                            else:
                                break

                if not improved:
                    step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
