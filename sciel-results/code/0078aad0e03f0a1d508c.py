"""Rewritten program: Nelder-Mead simplex optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts for robust optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        eps = 1e-8

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls_used += 1
            return f

        # Initial simplex: start near origin with reasonable spread
        # For Schaffers F7, the landscape has valleys; we want a good initial spread
        base = np.zeros(dim)
        scale = 1.0
        simplex = [base.copy()]
        for i in range(dim):
            v = base.copy()
            v[i] = scale
            simplex.append(v)

        # Evaluate initial simplex
        fvals = np.array([evaluate(x) for x in simplex])

        # Track best
        idx_min = np.argmin(fvals)
        best_x = simplex[idx_min].copy()
        best_f = fvals[idx_min]

        # Adaptive Nelder-Mead with restarts
        # We'll run multiple simplex iterations, restarting if stuck
        max_iter_per_run = 150
        num_restarts = budget // max_iter_per_run + 1

        for restart in range(num_restarts):
            if calls_used >= budget:
                break

            if restart == 0:
                # Use the existing simplex
                pass
            else:
                # Restart: new simplex centered at best_x with smaller scale
                center = best_x.copy()
                scale = max(0.1, 0.5 * (1.0 / (restart + 1)))
                simplex = [center.copy()]
                for i in range(dim):
                    v = center.copy()
                    v[i] = center[i] + scale
                    simplex.append(v)
                fvals = np.array([evaluate(x) for x in simplex])
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = simplex[idx_min].copy()

            # Nelder-Mead iterations
            for it in range(max_iter_per_run):
                if calls_used >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Check convergence
                if np.std(fvals) < eps * (1 + np.abs(fvals[0])):
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = evaluate(x_ref)

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = evaluate(x_con)
                        if f_con <= f_ref:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fvals[i] = evaluate(simplex[i])
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = evaluate(x_con)
                        if f_con < fvals[-1]:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                fvals[i] = evaluate(simplex[i])

                # Update best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = simplex[idx_min].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
