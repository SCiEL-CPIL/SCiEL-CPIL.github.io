"""Optimizer for BBOB f7 (Step ellipsoidal) using Nelder-Mead with LHS initialization
and adaptive restarts. NM navigates flat plateaus better than gradient methods."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with Latin Hypercube init and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(7)

        best_x = None
        best_f = np.inf
        calls = 0

        def ok():
            return time.time() < deadline and calls < budget

        # Latin Hypercube sampling for initial diverse points
        def lhs_sample(n, d, lo=-4.5, hi=4.5):
            """Latin Hypercube sampling in [lo, hi]^d."""
            pts = np.zeros((n, d))
            for j in range(d):
                edges = np.linspace(0, 1, n + 1)
                u = rng.uniform(edges[:-1], edges[1:])
                order = rng.permutation(n)
                pts[:, j] = u[order]
            pts = lo + (hi - lo) * pts
            return pts

        # Nelder-Mead implementation
        def nelder_mead(func, x0, budget_left, deadline, rng):
            """Run NM simplex, return (best_x, best_f, calls_used)."""
            n = dim
            # Initialize simplex: x0 + alpha * e_i
            alpha = 0.5
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += alpha

            # Evaluate initial simplex
            fvals = np.zeros(n + 1)
            local_calls = 0
            for i in range(n + 1):
                if not (time.time() < deadline and calls + local_calls < budget):
                    break
                fvals[i] = func(simplex[i])
                local_calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # NM parameters
            alpha_r = 1.0   # reflection
            alpha_e = 2.0   # expansion
            alpha_c = 0.5   # contraction
            alpha_sh = 0.5  # shrink
            max_iter = 200
            tol = 1e-10

            iter_count = 0
            while iter_count < max_iter:
                if not (time.time() < deadline and calls + local_calls < budget):
                    break
                iter_count += 1

                # Sort simplex
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Check convergence
                if fvals[-1] - fvals[0] < tol:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha_r * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                local_calls += 1
                if fr < fvals[-1]:
                    if fr < fvals[0]:
                        # Try expansion
                        xe = centroid + alpha_e * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = func(xe)
                        local_calls += 1
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + alpha_c * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    local_calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + alpha_sh * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        for i in range(1, n + 1):
                            fvals[i] = func(simplex[i])
                            local_calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Track best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f, local_calls

        # Phase 1: LHS initialization - evaluate diverse points
        n_lhs = 15
        lhs_points = lhs_sample(n_lhs, dim)
        lhs_fvals = []
        for i in range(n_lhs):
            if not ok():
                break
            f = func(lhs_points[i])
            calls += 1
            lhs_fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = lhs_points[i].copy()

        # Phase 2: Run NM from top few LHS points
        # Sort LHS by fitness
        if len(lhs_fvals) > 0:
            order = np.argsort(lhs_fvals)
            # Try NM from best, 2nd best, 3rd best
            for rank in range(min(3, len(order))):
                if not ok():
                    break
                x0 = lhs_points[order[rank]].copy()
                _, _, used = nelder_mead(func, x0, budget - calls, deadline, rng)
                calls += used
                if not ok():
                    break

        # Phase 3: If still have budget, do more LHS + NM cycles
        remaining_cycles = 3
        for cycle in range(remaining_cycles):
            if not ok():
                break
            # Check if we've made good progress
            # Do a smaller LHS around best_x
            n_local = 5
            local_points = best_x + rng.standard_normal((n_local, dim)) * 1.0
            local_points = np.clip(local_points, -4.5, 4.5)
            local_fvals = []
            for i in range(n_local):
                if not ok():
                    break
                f = func(local_points[i])
                calls += 1
                local_fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = local_points[i].copy()

            if local_fvals:
                order = np.argsort(local_fvals)
                x0 = local_points[order[0]].copy()
                _, _, used = nelder_mead(func, x0, budget - calls, deadline, rng)
                calls += used

        # Phase 4: Final polish - one more NM from best
        if ok() and best_x is not None:
            _, _, used = nelder_mead(func, best_x.copy(), budget - calls, deadline, rng)
            calls += used

        return best_x, best_f
# EVOLVE-BLOCK-END
