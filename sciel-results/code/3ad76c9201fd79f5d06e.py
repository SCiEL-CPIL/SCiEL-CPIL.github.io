"""Nelder-Mead simplex optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, alpha=1.0, gamma=2.0, rho=0.5, max_iter=100):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return

            # Initialize simplex: x0 plus dim points offset by 0.5 along each axis
            simplex = [x0.copy()]
            for i in range(dim):
                pt = x0.copy()
                pt[i] += 0.5 if x0[i] < 4.5 else -0.5
                pt[i] = np.clip(pt[i], -5.0, 5.0)
                simplex.append(pt)

            # Evaluate all simplex points
            simplex_f = [eval_func(pt) for pt in simplex]

            for iteration in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    return

                # Sort by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                f_best = simplex_f[0]
                f_worst = simplex_f[-1]
                f_second_worst = simplex_f[-2]

                # Check convergence
                if abs(f_worst - f_best) < 1e-10:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = eval_func(x_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = eval_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < f_second_worst:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = eval_func(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_f[i] = eval_func(simplex[i])
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = eval_func(x_c)
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_f[i] = eval_func(simplex[i])

        # Multi-start: several diverse starting points
        starts = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
            np.array([2.5, -2.5, 2.5, -2.5, 2.5])[:dim],
            np.array([-2.5, 2.5, -2.5, 2.5, -2.5])[:dim],
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for start in starts:
            if calls_made >= budget - 10 or time.time() > deadline:
                break
            nelder_mead(start, max_iter=50)

        # If we still have budget, do a final refinement around the best point
        if best_x is not None and calls_made < budget - 5 and time.time() < deadline:
            nelder_mead(best_x.copy(), max_iter=30)

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
