"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-restart strategy and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def eval_point(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-restart Nelder-Mead
        num_restarts = 6
        budget_per_restart = max(40, budget // num_restarts)

        # Different initial centers for restarts
        rng = np.random.default_rng(12345)
        centers = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]) * 2.0
        ]

        for restart in range(num_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            center = centers[restart]

            # Initialize simplex
            alpha = 0.5  # step size for initial simplex
            simplex = []
            fvals = []

            # First vertex: center
            x0 = np.clip(center, -5, 5)
            f0 = eval_point(x0)
            if f0 is None:
                break
            simplex.append(x0)
            fvals.append(f0)

            # Other vertices: perturb each dimension
            for i in range(dim):
                x_i = x0.copy()
                step = alpha if i == 0 else alpha * 0.5
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                f_i = eval_point(x_i)
                if f_i is None:
                    break
                simplex.append(x_i)
                fvals.append(f_i)

            if len(simplex) < 2:
                continue

            # Nelder-Mead parameters
            alpha_ref = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5  # contraction
            sigma_shrink = 0.5  # shrink
            max_iter = budget_per_restart
            stall_count = 0
            stall_threshold = 30

            for it in range(max_iter):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by fitness
                sorted_idx = np.argsort(fvals)
                simplex = [simplex[i] for i in sorted_idx]
                fvals = [fvals[i] for i in sorted_idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_x = simplex[-1]
                worst_f = fvals[-1]

                # Reflection
                x_ref = centroid + alpha_ref * (centroid - worst_x)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_point(x_ref)
                if f_ref is None:
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - worst_x)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_point(x_exp)
                    if f_exp is None:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                    stall_count = 0
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                    stall_count = 0
                else:
                    # Contraction
                    if f_ref < worst_f:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (worst_x - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_point(x_con)
                    if f_con is None:
                        break
                    if f_con < min(f_ref, worst_f):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                        stall_count = 0
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = eval_point(simplex[i])
                            if f_i is None:
                                break
                            fvals[i] = f_i
                        stall_count += 1

                # Check for convergence
                if stall_count > stall_threshold:
                    break

                # Check if simplex is small enough
                spread = np.max(np.abs(np.array(simplex) - np.mean(simplex, axis=0)))
                if spread < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
