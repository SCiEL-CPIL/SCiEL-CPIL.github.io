"""Nelder-Mead simplex optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts for sharp ridge geometry."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Count function evaluations
        eval_count = [0]

        def f(x):
            eval_count[0] += 1
            return func(np.clip(x, -5, 5))

        # Initial simplex centered at origin (box center)
        # For sharp ridge, start with a reasonably sized simplex
        x_center = np.zeros(dim)
        f_center = f(x_center)

        # Build initial simplex
        simplex = [x_center.copy()]
        fvals = [f_center]

        # Initial step size - somewhat large to explore
        initial_step = 1.5
        for i in range(dim):
            x_i = x_center.copy()
            x_i[i] += initial_step
            simplex.append(x_i.copy())
            fvals.append(f(x_i))

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        best_x = simplex[0].copy()
        best_f = min(fvals)

        # Track best found
        if f_center < best_f:
            best_x = x_center.copy()
            best_f = f_center

        for i in range(dim):
            if fvals[i + 1] < best_f:
                best_f = fvals[i + 1]
                best_x = simplex[i + 1].copy()

        iteration = 0
        max_iterations = 5000

        while eval_count[0] < budget and iteration < max_iterations:
            # Check time deadline
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence: if the range of function values is tiny
            if fvals[-1] - fvals[0] < 1e-10:
                # Restart with a smaller step from the best point
                break

            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = f(x_r)

            if fvals[0] <= f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            elif f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                f_c = f(x_c)

                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])

            # Update best
            for i in range(dim + 1):
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            iteration += 1

            # Check if we've converged well enough to restart
            # If the simplex is very small, restart from best point with fresh simplex
            simplex_range = max(np.max(simplex, axis=0) - np.min(simplex, axis=0))
            if simplex_range < 1e-4:
                # Restart: create new simplex around best point
                if eval_count[0] >= budget:
                    break
                initial_step = max(simplex_range * 5, 0.1)
                simplex = [best_x.copy()]
                fvals = [best_f]
                for i in range(dim):
                    x_i = best_x.copy()
                    x_i[i] += initial_step
                    simplex.append(x_i.copy())
                    fvals.append(f(x_i))
                # Update best from new simplex
                for i in range(dim + 1):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = simplex[i].copy()
                iteration = 0  # reset iteration counter

        return best_x, best_f
# EVOLVE-BLOCK-END
