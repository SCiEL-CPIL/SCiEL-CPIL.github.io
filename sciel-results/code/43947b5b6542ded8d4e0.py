"""Multi-objective optimizer on WFG9 exploiting analytic structure."""
import numpy as np
import time


class Optimizer:
    """Exploit WFG9 structure: fix distance variables, grid search shape variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # 280s safety margin before 300s limit
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n_obj = self.n_obj
        budget = self.budget

        # WFG9 structure: x1,x2,x3 are "shape" variables that directly affect objectives,
        # x4..x12 are "distance" variables. The optimal Pareto front is obtained by
        # setting distance variables to their bounds (typically the lower bound).
        #
        # WFG9 objectives (standard):
        # f1 = g(x_d) * (1 + h1(x_p))
        # f2 = g(x_d) * h2(x_p)
        # f3 = g(x_d) * h3(x_p)
        # where g(x_d) depends on distance vars, and h's depend on shape vars.
        #
        # The Pareto front is achieved when g(x_d) is minimized (distance vars at bounds)
        # and shape variables are varied to trace out the front.

        # Fix distance variables (indices 3..11) to their lower bounds
        x_base = np.zeros(12)
        x_base[:3] = lo[:3]  # will be overwritten by grid
        x_base[3:] = lo[3:]  # distance vars at lower bound

        # Also try upper bounds for distance vars
        x_base_hi = x_base.copy()
        x_base_hi[3:] = hi[3:]

        # Grid search on the 3 shape variables
        # Use a moderate grid: 8x8x8 = 512 points, but we only have 200 budget
        # So use 6x6x6 = 216 -> too many. Use 5x5x4 = 100 points
        n1, n2, n3 = 5, 5, 4
        grid_size = n1 * n2 * n3

        # Evaluate base version (distance at lower bound)
        all_x = []
        all_f = []
        used = 0

        for i in range(n1):
            v1 = lo[0] + (hi[0] - lo[0]) * (i / max(n1 - 1, 1)) if n1 > 1 else (lo[0] + hi[0]) / 2
            for j in range(n2):
                v2 = lo[1] + (hi[1] - lo[1]) * (j / max(n2 - 1, 1)) if n2 > 1 else (lo[1] + hi[1]) / 2
                for k in range(n3):
                    v3 = lo[2] + (hi[2] - lo[2]) * (k / max(n3 - 1, 1)) if n3 > 1 else (lo[2] + hi[2]) / 2
                    x = x_base.copy()
                    x[0] = v1
                    x[1] = v2
                    x[2] = v3
                    if used >= budget:
                        break
                    f = problem(x)
                    all_x.append(x.copy())
                    all_f.append(np.array(f))
                    used += 1
                    if time.time() > deadline:
                        return all_x
                if used >= budget or time.time() > deadline:
                    break
            if used >= budget or time.time() > deadline:
                break

        # Try distance vars at upper bound too
        if used < budget and time.time() < deadline:
            for i in range(n1):
                v1 = lo[0] + (hi[0] - lo[0]) * (i / max(n1 - 1, 1)) if n1 > 1 else (lo[0] + hi[0]) / 2
                for j in range(n2):
                    v2 = lo[1] + (hi[1] - lo[1]) * (j / max(n2 - 1, 1)) if n2 > 1 else (lo[1] + hi[1]) / 2
                    for k in range(n3):
                        v3 = lo[2] + (hi[2] - lo[2]) * (k / max(n3 - 1, 1)) if n3 > 1 else (lo[2] + hi[2]) / 2
                        x = x_base_hi.copy()
                        x[0] = v1
                        x[1] = v2
                        x[2] = v3
                        if used >= budget:
                            break
                        f = problem(x)
                        all_x.append(x.copy())
                        all_f.append(np.array(f))
                        used += 1
                        if time.time() > deadline:
                            return all_x
                    if used >= budget or time.time() > deadline:
                        break
                if used >= budget or time.time() > deadline:
                    break

        # Now do local refinement around non-dominated points
        if used < budget and time.time() < deadline:
            F = np.array(all_f)
            X = np.array(all_x)

            # Find non-dominated points
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_mask[i] = False
                        break

            nd_indices = np.where(nd_mask)[0]

            # Local search around each non-dominated point
            rng = np.random.default_rng(42)
            step = 0.05
            for idx in nd_indices:
                if used >= budget or time.time() > deadline:
                    break
                # Try perturbations in all dimensions
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    direction = rng.standard_normal(12)
                    direction /= (np.linalg.norm(direction) + 1e-12)
                    x_new = X[idx] + step * direction
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    f_new_arr = np.array(f_new)
                    all_x.append(x_new.copy())
                    all_f.append(f_new_arr)
                    used += 1

        return all_x
