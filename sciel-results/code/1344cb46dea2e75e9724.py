"""Coordinate-wise ternary search + Nelder-Mead refinement for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Two-phase optimizer for BBOB f20 (Schwefel) on [-5, 5]^5.
    Phase 1: Coordinate-wise ternary search to estimate local centers.
    Phase 2: Nelder-Mead refinement around the best point found.
    Explicit deadline guard to stay well within the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_fx = float('inf')
        calls_used = 0

        def try_func(x):
            nonlocal calls_used, best_fx, best_x
            if calls_used >= total_budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls_used < total_budget

        rng = np.random.default_rng(42)

        # Phase 1: Coordinate-wise ternary search from multiple random starts
        # Use 60% of budget for this phase
        phase1_budget = int(total_budget * 0.6)
        n_starts = 5
        budget_per_start = int(phase1_budget / n_starts)

        candidates = []

        for s in range(n_starts):
            if not budget_ok() or not time_ok():
                break

            # Random starting point
            x = rng.uniform(-5, 5, dim)
            fx = try_func(x)

            # Coordinate-wise ternary search for each dimension
            for d in range(dim):
                if not budget_ok() or not time_ok():
                    break

                # Ternary search on dimension d
                lo = -5.0
                hi = 5.0
                # Limit iterations per dimension to save budget
                max_iter_per_dim = min(15, max(1, budget_per_start // (dim * 3)))

                for _ in range(max_iter_per_dim):
                    if not budget_ok() or not time_ok():
                        break

                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0

                    x1 = x.copy()
                    x1[d] = m1
                    f1 = try_func(x1)

                    x2 = x.copy()
                    x2[d] = m2
                    f2 = try_func(x2)

                    if f1 < f2:
                        hi = m2
                    else:
                        lo = m1

                    # Update x if the new midpoint is better
                    mid = (lo + hi) / 2.0
                    x_new = x.copy()
                    x_new[d] = mid
                    f_new = try_func(x_new)
                    if f_new < fx:
                        x = x_new
                        fx = f_new

            candidates.append((x.copy(), fx))

        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[1])

        # Phase 2: Nelder-Mead refinement around the best candidates
        remaining_budget = total_budget - calls_used
        n_refine = min(3, len(candidates))
        budget_per_refine = int(remaining_budget / n_refine) if n_refine > 0 else 0

        for i in range(n_refine):
            if not budget_ok() or not time_ok():
                break

            x_start = candidates[i][0].copy()
            fx_start = candidates[i][1]

            # Initialize simplex
            simplex = [x_start.copy()]
            f_vals = [fx_start]

            # Create other vertices with small perturbations
            for j in range(dim):
                if not budget_ok() or not time_ok():
                    break
                x_j = x_start.copy()
                step = 0.5 if x_start[j] < 0 else -0.5
                x_j[j] += step
                x_j = np.clip(x_j, -5, 5)
                f_j = try_func(x_j)
                simplex.append(x_j)
                f_vals.append(f_j)

            if len(simplex) < dim + 1:
                continue

            # Nelder-Mead iterations
            max_iter = 30
            for _ in range(max_iter):
                if not budget_ok() or not time_ok():
                    break

                # Sort simplex by function value
                idx = np.argsort(f_vals)
                simplex = [simplex[i] for i in idx]
                f_vals = [f_vals[i] for i in idx]

                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = try_func(x_ref)

                if f_ref < f_vals[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = try_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = try_func(x_con)

                    if f_con < f_vals[-1]:
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for k in range(1, len(simplex)):
                            if not budget_ok() or not time_ok():
                                break
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            simplex[k] = np.clip(simplex[k], -5, 5)
                            f_vals[k] = try_func(simplex[k])

        return best_x, best_fx
# EVOLVE-BLOCK-END
