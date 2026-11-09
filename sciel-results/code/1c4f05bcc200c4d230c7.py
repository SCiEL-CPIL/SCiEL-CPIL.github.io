"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS warm-up + Nelder-Mead with adaptive step sizing for Sphere minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        calls_made = 0

        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Latin Hypercube Sampling for initial coverage
        n_lhs = 60
        lhs_points = []
        for i in range(n_lhs):
            # For each dimension, divide into n_lhs equal intervals and sample one point
            point = np.zeros(dim)
            for d in range(dim):
                interval = i / n_lhs
                point[d] = -5.0 + (interval + rng.random()) * (10.0 / n_lhs)
            lhs_points.append(point)

        best_x = None
        best_f = float('inf')

        for p in lhs_points:
            fx = eval_point(p)
            if fx is None:
                break

        # Phase 2: Nelder-Mead with adaptive step sizing
        # Start from the best LHS point
        x = best_x.copy()
        f_x = best_f

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        # Initial simplex: center + scaled unit vectors
        step = 0.5
        simplex = [x.copy()]
        for i in range(dim):
            v = x.copy()
            v[i] += step
            simplex.append(v)

        # Evaluate simplex
        simplex_f = []
        for v in simplex:
            fx = eval_point(v)
            if fx is None:
                break
            simplex_f.append(fx)

        # Main Nelder-Mead loop with adaptive step sizing
        no_improve = 0
        best_f_nm = best_f

        while calls_made < budget and time.time() < deadline:
            if len(simplex_f) < dim + 1:
                break

            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Check for convergence
            if simplex_f[0] < best_f_nm - 1e-10:
                best_f_nm = simplex_f[0]
                no_improve = 0
            else:
                no_improve += 1

            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = eval_point(x_r)
            if f_r is None:
                break

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = eval_point(x_e)
                if f_e is None:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Reflection accepted
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = eval_point(x_c)
                if f_c is None:
                    break
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        f_s = eval_point(simplex[i])
                        if f_s is None:
                            break
                        simplex_f[i] = f_s
                    else:
                        # Adaptive step: shrink the step size for new simplex
                        step *= 0.5
                        if step < 1e-4:
                            break
                        # Rebuild simplex with smaller step
                        simplex = [simplex[0].copy()]
                        simplex_f = [simplex_f[0]]
                        for i in range(dim):
                            v = simplex[0].copy()
                            v[i] += step
                            simplex.append(v)
                            fx = eval_point(v)
                            if fx is None:
                                break
                            simplex_f.append(fx)
                        if len(simplex_f) < dim + 1:
                            break

            # If no improvement for many iterations, restart with new step
            if no_improve > 50:
                step = max(step * 2, 0.1)
                simplex = [best_x.copy()]
                simplex_f = [best_f]
                for i in range(dim):
                    v = best_x.copy()
                    v[i] += step
                    simplex.append(v)
                    fx = eval_point(v)
                    if fx is None:
                        break
                    simplex_f.append(fx)
                no_improve = 0
                if len(simplex_f) < dim + 1:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
