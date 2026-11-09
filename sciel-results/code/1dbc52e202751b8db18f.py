"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with time guard for BBOB-style f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        n = self.dim
        budget = self.budget

        # Initialize at origin (standard center for BBOB)
        x0 = np.zeros(n)
        f0 = func(x0)
        calls = 1
        best_f = f0
        best_x = x0.copy()

        if calls >= budget:
            return best_x, best_f

        # Build initial simplex: x0 plus n unit-step vertices
        alpha = 0.5  # initial simplex spread
        simplex = [x0.copy()]
        simplex_f = [f0]

        for i in range(n):
            if time.time() > deadline:
                break
            v = x0.copy()
            v[i] += alpha
            v = np.clip(v, -5, 5)
            fv = func(v)
            calls += 1
            simplex.append(v.copy())
            simplex_f.append(fv)
            if fv < best_f:
                best_f = fv
                best_x = v.copy()

        if calls >= budget or len(simplex) < n + 1:
            return best_x, best_f

        # Nelder-Mead parameters
        gamma = 1.0   # reflection
        rho = 2.0     # expansion
        sigma_c = 0.5 # contraction
        xi = 0.5      # shrink

        def sort_simplex():
            nonlocal simplex, simplex_f
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

        sort_simplex()

        while calls < budget:
            if time.time() > deadline:
                break

            sort_simplex()
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + gamma * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()

            if f_r < simplex_f[0]:
                # Try expansion
                x_e = centroid + rho * (x_r - centroid)
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
                    x_c = centroid + sigma_c * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - sigma_c * (x_r - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()

                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + xi * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if time.time() > deadline:
                            break
                        f_s = func(simplex[i])
                        calls += 1
                        simplex_f[i] = f_s
                        if f_s < best_f:
                            best_f = f_s
                            best_x = simplex[i].copy()
                    if calls >= budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
