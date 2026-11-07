import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin from 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Generate diverse initial points
        n_initial = 25
        points = []
        values = []

        # Center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        points.append(x_center)
        values.append(f_center)

        # Vertices (8 corners for 5D)
        for i in range(1, min(8, n_initial - 1)):
            x_vert = np.array([5.0 if (i >> j) & 1 else -5.0 for j in range(dim)])
            f_vert = func(x_vert)
            points.append(x_vert)
            values.append(f_vert)

        # Random points to fill up to n_initial
        for _ in range(len(points), n_initial):
            x_rand = rng.uniform(-5, 5, dim)
            f_rand = func(x_rand)
            points.append(x_rand)
            values.append(f_rand)

        # Find top 5 best points
        sorted_indices = np.argsort(values)
        top_indices = sorted_indices[:5]

        best_x = points[top_indices[0]].copy()
        best_f = values[top_indices[0]]

        calls_used = len(points)
        remaining_budget = budget - calls_used

        # Phase 2: Multi-start Nelder-Mead with adaptive parameters
        nm_budget_per_start = max(20, remaining_budget // 5)

        for idx in top_indices:
            if time.time() > deadline - 5:
                break

            # Initialize Nelder-Mead simplex around the starting point
            x0 = points[idx].copy()
            fx0 = values[idx]

            # Build initial simplex
            simplex = [x0.copy()]
            simple_values = [fx0]

            # Add perturbed points
            step_size = 0.5
            for j in range(dim):
                x_pert = x0.copy()
                x_pert[j] += step_size
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = func(x_pert)
                simplex.append(x_pert)
                simple_values.append(f_pert)

            nm_calls = 1 + dim  # initial simplex evaluations
            nm_budget = nm_budget_per_start
            max_iter = nm_budget - nm_calls

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for iteration in range(max_iter):
                if time.time() > deadline - 2:
                    break

                if nm_calls >= nm_budget:
                    break

                # Sort simplex by function value
                order = np.argsort(simple_values)
                simplex = [simplex[i].copy() for i in order]
                simple_values = [simple_values[i] for i in order]

                # Check if we found a better global best
                if simple_values[0] < best_f:
                    best_f = simple_values[0]
                    best_x = simplex[0].copy()

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                nm_calls += 1

                if f_ref < simple_values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    nm_calls += 1

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simple_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simple_values[-1] = f_ref
                elif f_ref < simple_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simple_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simple_values[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)

                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    nm_calls += 1

                    if f_con <= f_ref:
                        simplex[-1] = x_con
                        simple_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_shrink = func(simplex[i])
                            nm_calls += 1
                            simple_values[i] = f_shrink

                        if nm_calls >= nm_budget:
                            break

        return best_x, best_f
