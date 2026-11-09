import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def nelder_mead(x0, max_calls):
            nonlocal calls, best_x, best_f

            # Initialize simplex
            delta = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += delta
                simplex.append(x_i)

            # Evaluate initial simplex
            simplex_f = []
            for x in simplex:
                if calls >= max_calls or time.time() >= deadline:
                    break
                fx = func(x)
                calls += 1
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            if len(simplex_f) < dim + 1:
                return

            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            stagnation_counter = 0
            prev_best_f = best_f

            while calls < max_calls and time.time() < deadline:
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check stagnation
                if simplex_f[0] < prev_best_f - 1e-10:
                    prev_best_f = simplex_f[0]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter > 50:
                    break

                # Check simplex size
                if np.linalg.norm(simplex[0] - simplex[-1]) < 1e-8:
                    break

                # Centroid of best d points
                centroid = np.mean(simplex[:dim], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_i = func(simplex[i])
                                calls += 1
                                simplex_f[i] = f_i
                                if f_i < best_f:
                                    best_f = f_i
                                    best_x = simplex[i].copy()
                                if calls >= max_calls or time.time() >= deadline:
                                    break
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_i = func(simplex[i])
                                calls += 1
                                simplex_f[i] = f_i
                                if f_i < best_f:
                                    best_f = f_i
                                    best_x = simplex[i].copy()
                                if calls >= max_calls or time.time() >= deadline:
                                    break
                            if calls >= max_calls or time.time() >= deadline:
                                break

        # Main loop: Multiple restarts
        # We allocate budget chunks to each restart
        # Lunacek is bi-modal, so we need to explore different regions.

        # First run: Random start
        x0 = rng.uniform(-4, 4, dim)
        alloc = 100
        nelder_mead(x0, min(calls + alloc, budget))

        # Subsequent runs
        while calls < budget and time.time() < deadline - 5:
            # Decide start point
            # 70% chance to try a completely new random point (to find the other mode)
            # 30% chance to perturb the current best (to refine)
            if rng.random() < 0.7:
                x0 = rng.uniform(-4, 4, dim)
            else:
                # Perturb best
                x0 = best_x + 0.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -4, 4)

            # Allocate remaining budget, but cap per run to allow multiple restarts
            # Let's try allocating 50 calls per run
            alloc = 50
            nelder_mead(x0, min(calls + alloc, budget))

        return best_x, best_f
