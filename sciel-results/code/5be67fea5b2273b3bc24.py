"""Multi-Start Nelder-Mead optimizer for BBOB f11 (Discus)."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with strategic initialization for low-dimensional search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill

        n = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_fx = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_fx
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Strategic initialization - center and axis-aligned points
        init_points = [np.zeros(n)]
        for i in range(n):
            for delta in [-2.5, 2.5]:
                pt = np.zeros(n)
                pt[i] = delta
                init_points.append(pt)

        for pt in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            evaluate(pt)

        # Phase 2: Nelder-Mead from the current best
        def nelder_mead(start_x, max_iter=150):
            nonlocal calls_made, best_fx, best_x
            if calls_made >= budget or time.time() > deadline:
                return

            # Initialize simplex around start_x
            simplex = [start_x.copy()]
            for i in range(n):
                vertex = start_x.copy()
                vertex[i] += 0.3
                simplex.append(vertex)

            fvals = []
            for v in simplex:
                if calls_made >= budget or time.time() > deadline:
                    return
                fvals.append(evaluate(v))

            if len(fvals) < n + 1:
                return

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = evaluate(x_ref)

                if f_ref < fvals[0]:
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = evaluate(x_con)
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])

        if best_x is not None:
            nelder_mead(best_x, max_iter=200)

        return best_x, best_fx
