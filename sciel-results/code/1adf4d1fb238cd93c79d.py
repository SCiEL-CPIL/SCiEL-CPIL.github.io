"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive scaling and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        min_sigma = 1e-10

        def time_ok():
            return time.time() < deadline

        # Initialize simplex
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        # Build initial simplex around x0
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] = x0[i] + 1.0
            simplex.append(xi)

        # Evaluate initial simplex
        simplex_f = [f0]
        for i in range(1, dim + 1):
            if not time_ok():
                break
            simplex_f.append(func(simplex[i]))
            calls += 1
            if simplex_f[-1] < best_f:
                best_f = simplex_f[-1]
                best_x = simplex[i].copy()

        # Nelder-Mead iterations
        while calls < budget and time_ok():
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Check for convergence
            f_range = simplex_f[-1] - simplex_f[0]
            if f_range < 1e-14:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

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
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
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
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
                    sigma *= 0.5
                    if sigma < min_sigma:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
