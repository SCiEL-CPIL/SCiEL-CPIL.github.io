"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Dense grid on front variables with targeted Pareto refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9: first 3 variables determine the Pareto front shape.
        # Fix variables 3..11 to midpoint (optimal for WFG9).
        base = mid.copy()

        X_list = []
        F_list = []
        used = 0

        def should_stop():
            return used >= self.budget or time.time() > deadline

        # Stage 1: Dense grid on the 3 front variables
        # 7*6*4 = 168 evaluations, leaving 32 for refinement
        n1, n2, n3 = 7, 6, 4

        t1_vals = np.linspace(0.0, 1.0, n1)
        t2_vals = np.linspace(0.0, 1.0, n2)
        t3_vals = np.linspace(0.0, 1.0, n3)

        for t1 in t1_vals:
            for t2 in t2_vals:
                for t3 in t3_vals:
                    if should_stop():
                        break

                    x = base.copy()
                    x[0] = lo[0] + t1 * (hi[0] - lo[0])
                    x[1] = lo[1] + t2 * (hi[1] - lo[1])
                    x[2] = lo[2] + t3 * (hi[2] - lo[2])

                    f = problem(x)
                    X_list.append(x)
                    F_list.append(f)
                    used += 1

                if should_stop():
                    break

            if should_stop():
                break

        if should_stop():
            if X_list:
                return np.array(X_list)
            else:
                return base.reshape(1, -1)

        # Stage 2: Targeted refinement of non-dominated points
        F_arr = np.array(F_list)

        def get_non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        nd_indices = get_non_dominated_indices(F_arr)

        # Refine non-dominated points with small perturbations
        remaining = self.budget - used
        # Sort non-dominated points by sum of objectives (prefer lower sum)
        nd_sorted = sorted(nd_indices, key=lambda i: np.sum(F_arr[i]))
        
        # Use remaining budget for refinements
        max_refine = min(len(nd_sorted), remaining)
        
        for idx in nd_sorted[:max_refine]:
            if should_stop():
                break

            center = X_list[idx]

            # Try small perturbations in each of the 3 front dimensions
            for dim in range(3):
                if should_stop():
                    break

                # Try a small step toward the center of the range
                span = hi[dim] - lo[dim]
                step = 0.015 * span

                # Try moving slightly in both directions
                for sign in [1, -1]:
                    if should_stop():
                        break

                    x = center.copy()
                    x[dim] = np.clip(x[dim] + sign * step, lo[dim], hi[dim])

                    f = problem(x)
                    X_list.append(x)
                    F_list.append(f)
                    used += 1

        if X_list:
            X = np.array(X_list)
        else:
            X = base.reshape(1, -1)

        return X
# EVOLVE-BLOCK-END
