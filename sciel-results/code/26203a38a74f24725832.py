"""Nelder-Mead simplex optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 290  # safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        # Helper to evaluate with budget tracking
        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        # Initial simplex: center + small perturbations
        # Start at origin (center of box) for f13 which has a ridge structure
        base = np.zeros(dim)
        simplex = [base.copy()]
        for i in range(dim):
            v = base.copy()
            v[i] = 1.0
            simplex.append(v)

        # Evaluate initial simplex
        fvals = np.array([eval_func(x) for x in simplex])
        simplex = np.array(simplex)

        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        # Nelder-Mead iterations
        max_iters = budget - (dim + 1)  # remaining budget for iterations
        for iteration in range(max_iters):
            # Check deadline
            if time.time() > self.deadline:
                break
            # Check budget
            if calls >= budget:
                break

            # Compute centroid excluding worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = eval_func(x_ref)
            if calls >= budget:
                break

            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = eval_func(x_exp)
                if calls >= budget:
                    break
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
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_ref - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = eval_func(x_con)
                if calls >= budget:
                    break

                if f_con <= min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = eval_func(simplex[i])
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            # Re-sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check for convergence (simplex is very small)
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

            # If simplex has collapsed, restart with a larger perturbation
            simplex_size = np.max(np.linalg.norm(simplex - simplex[0], axis=1))
            if simplex_size < 1e-4 and calls < budget:
                # Restart around current best with larger step
                step = 2.0
                simplex = [simplex[0].copy()]
                for i in range(dim):
                    v = simplex[0].copy()
                    v[i] += step
                    v = np.clip(v, -5.0, 5.0)
                    simplex.append(v)
                fvals = np.array([eval_func(x) for x in simplex])
                if calls >= budget:
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

        return best_x, best_f
# EVOLVE-BLOCK-END
