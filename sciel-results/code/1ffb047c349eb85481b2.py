"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid multi-start Nelder-Mead with adaptive global restart for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def time_left():
            return max_time - (time.time() - start_time)

        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time_left() <= 0:
                return None
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(start_x, max_calls=100, max_iter=200):
            """Run Nelder-Mead from start_x, updating best_x/best_f."""
            nonlocal calls, best_f, best_x

            if time_left() <= 0 or calls >= budget:
                return

            # Build initial simplex
            simplex = [start_x.copy()]
            base_sigma = 0.3
            for i in range(dim):
                pt = start_x.copy()
                pt[i] += base_sigma
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)

            # Evaluate simplex
            simplex_f = []
            for pt in simplex:
                if calls >= budget or time_left() <= 0:
                    break
                f = func(pt)
                calls += 1
                simplex_f.append(f)
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()

            if len(simplex_f) < dim + 1:
                return

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            nm_iter = 0
            while nm_iter < max_iter and calls < budget and time_left() > 0:
                nm_iter += 1

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if simplex_f[-1] - simplex_f[0] < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst_pt = simplex[-1]
                worst_f = simplex_f[-1]
                x_ref = centroid + alpha * (centroid - worst_pt)
                x_ref = np.clip(x_ref, -5, 5)

                if calls >= budget or time_left() <= 0:
                    break
                f_ref = func(x_ref)
                calls += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if calls >= budget or time_left() <= 0:
                        break
                    f_exp = func(x_exp)
                    calls += 1
                    if f_exp < f_ref:
                        x_new, f_new = x_exp, f_exp
                    else:
                        x_new, f_new = x_ref, f_ref
                    simplex[-1] = x_new
                    simplex_f[-1] = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                    if f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                else:
                    # Contraction
                    x_con = centroid + rho * (worst_pt - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if calls >= budget or time_left() <= 0:
                        break
                    f_con = func(x_con)
                    calls += 1
                    if f_con < worst_f:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                        if f_con < best_f:
                            best_f = f_con
                            best_x = x_con.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls >= budget or time_left() <= 0:
                                break
                            f_shrink = func(simplex[i])
                            calls += 1
                            simplex_f[i] = f_shrink
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()

        # Phase 1: Global exploration with diverse initial points
        # Use ~40% of budget for exploration, 60% for local refinement
        explore_budget = int(budget * 0.4)

        # Generate diverse initial points
        initial_points = []
        # Origin
        initial_points.append(np.zeros(dim))
        # Random points
        for i in range(10):
            if calls >= explore_budget or time_left() <= 0:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            initial_points.append(x)

        # Evaluate initial points
        point_fitness = []
        for x in initial_points:
            if calls >= explore_budget or time_left() <= 0:
                break
            fx = eval_point(x)
            if fx is not None:
                point_fitness.append((fx, x.copy()))

        # Sort by fitness
        point_fitness.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using Nelder-Mead from top candidates
        # Use remaining budget for local search
        local_budget = budget - calls

        # Select top candidates for local refinement
        num_local = min(len(point_fitness), 5)
        local_calls_per = max(30, local_budget // (num_local + 1))

        for idx in range(num_local):
            if calls >= budget or time_left() <= 0:
                break
            fx, x_start = point_fitness[idx]
            # Run Nelder-Mead from this starting point
            nelder_mead(x_start, max_calls=local_calls_per, max_iter=150)

        # Phase 3: If budget remains, do another round of local refinement
        # from the best point found
        if calls < budget and time_left() > 5:
            remaining = budget - calls
            if remaining > 20:
                nelder_mead(best_x.copy(), max_calls=remaining, max_iter=100)

        return best_x, best_f
# EVOLVE-BLOCK-END
