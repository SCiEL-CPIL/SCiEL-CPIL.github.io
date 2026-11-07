"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with LHS initialization for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Latin Hypercube sampling for diverse initial points
        def lhs_sample(n, dim, rng):
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = (perm + rng.uniform(size=n)) / n
            points = points * 10.0 - 5.0  # map to [-5, 5]
            return points

        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_evals, deadline, time_check_interval=100):
            dim = len(x0)
            # Initial simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            # Add other vertices
            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if x0[i] == 0 else 0.5 * abs(x0[i])
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(func(xi))

            eval_count = 1 + dim
            best_x = x0.copy()
            best_f = min(fvals)
            best_f = fvals[np.argmin(fvals)]
            best_idx = np.argmin(fvals)
            best_x = simplex[best_idx].copy()

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            while eval_count < max_evals:
                # Time check
                if time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                eval_count += 1

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    eval_count += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    eval_count += 1

                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            eval_count += 1
                            if eval_count >= max_evals:
                                break

                # Track best
                current_best = min(fvals)
                if current_best < best_f:
                    best_f = current_best
                    best_idx = np.argmin(fvals)
                    best_x = simplex[best_idx].copy()

                if eval_count >= max_evals:
                    break

            return best_x, best_f, eval_count

        # Multi-start strategy
        n_starts = 10
        lxs = lhs_sample(n_starts, dim, rng)

        # Allocate budget: give more to better starts
        # First, do a quick evaluation of all LHS points
        lhs_fvals = []
        for x in lxs:
            fx = func(x)
            lhs_fvals.append(fx)

        evals_used = n_starts
        remaining = budget - evals_used

        # Sort by fitness and allocate remaining budget
        order = np.argsort(lhs_fvals)

        # Give 60% to best, 25% to second, 15% to third, rest split
        allocs = []
        if remaining > 0:
            a1 = int(remaining * 0.5)
            a2 = int(remaining * 0.25)
            a3 = int(remaining * 0.15)
            a4 = remaining - a1 - a2 - a3
            allocs = [a1, a2, a3, a4]
        else:
            allocs = [0, 0, 0, 0]

        best_x = lxs[order[0]].copy()
        best_f = lhs_fvals[order[0]]

        for i, idx in enumerate(order):
            if i >= len(allocs):
                break
            alloc = allocs[i]
            if alloc <= 0 or time.time() > deadline:
                break

            x0 = lxs[idx].copy()
            x_opt, f_opt, used = nelder_mead(func, x0, alloc, deadline)
            evals_used += used

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

            if evals_used >= budget or time.time() > deadline:
                break

        # If we still have budget, do a final local refinement around best
        if evals_used < budget and time.time() < deadline:
            remaining2 = budget - evals_used
            if remaining2 > 10:
                x_refined, f_refined, _ = nelder_mead(func, best_x, remaining2, deadline)
                if f_refined < best_f:
                    best_f = f_refined
                    best_x = x_refined.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
