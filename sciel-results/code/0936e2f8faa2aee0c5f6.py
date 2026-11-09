# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 using structured sweep + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_F.append(tuple(f))
            return f

        # Phase 1: Structured deterministic sweep along key WFG9 variables
        # WFG9 structure: f1 = f1(x1,x2), f2 = f2(x3,x4), f3 = f3(x5,x6)
        # The remaining variables affect the shape but the main trade-offs are in pairs
        # We'll do a grid sweep over the first 6 variables (3 pairs) and fix the rest at midpoints

        # First, fix variables 6-11 at their midpoints
        mid = (lo + hi) / 2.0
        fixed_tail = mid[6:].copy()

        # Grid sizes for each of the 6 key variables
        # We have budget 200. Let's use ~80 for the sweep, ~120 for refinement
        # 3 pairs, each with a 2D grid. Let's do 4x4 per pair = 16 per pair = 48 total
        # Or 5x5 per pair = 25 per pair = 75 total
        grid_size = 5  # 5x5 per pair
        n_per_pair = grid_size * grid_size
        total_sweep = 3 * n_per_pair

        # But we also need to vary the tail variables somewhat
        # Let's do a two-level sweep:
        # Level 1: sweep the 6 key variables with tail at midpoint
        # Level 2: for the best points, refine by varying tail variables

        for pair in range(3):
            for i in range(grid_size):
                for j in range(grid_size):
                    if used >= budget:
                        break
                    x = mid.copy()
                    x[2 * pair] = lo[2 * pair] + (hi[2 * pair] - lo[2 * pair]) * i / (grid_size - 1)
                    x[2 * pair + 1] = lo[2 * pair + 1] + (hi[2 * pair + 1] - lo[2 * pair + 1]) * j / (grid_size - 1)
                    eval_point(x)

        # Phase 2: Local refinement using Gaussian perturbations around best points
        # Find non-dominated points from the sweep
        if all_F:
            F_arr = np.array(all_F)
            # Simple non-dominated filter
            dominated = np.zeros(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            non_dominated_idx = np.where(~dominated)[0]

            # Take up to 20 best non-dominated points for refinement
            if len(non_dominated_idx) > 20:
                # Keep the most spread out ones (simple: just take first 20)
                non_dominated_idx = non_dominated_idx[:20]

            # For each non-dominated point, do local Gaussian refinement
            remaining_budget = budget - used
            if remaining_budget > 0:
                n_refine_points = min(len(non_dominated_idx), 15)
                evals_per_point = max(1, remaining_budget // max(1, n_refine_points))

                # We need to track the actual x values, but we only stored F
                # Let's re-evaluate: we need to store x as well
                # Actually, let's redo this: store x and F together
                pass

        # Let's redo with proper tracking
        all_F = []
        all_X = []
        used = 0

        def eval_point2(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_F.append(tuple(f))
            all_X.append(x.copy())
            return f

        # Phase 1: Structured sweep
        mid = (lo + hi) / 2.0
        grid_size = 5

        for pair in range(3):
            for i in range(grid_size):
                for j in range(grid_size):
                    if used >= budget:
                        break
                    x = mid.copy()
                    x[2 * pair] = lo[2 * pair] + (hi[2 * pair] - lo[2 * pair]) * i / (grid_size - 1)
                    x[2 * pair + 1] = lo[2 * pair + 1] + (hi[2 * pair + 1] - lo[2 * pair + 1]) * j / (grid_size - 1)
                    eval_point2(x)

        # Phase 2: Local refinement
        if all_F and used < budget:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Find non-dominated points
            n_pts = len(F_arr)
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            nd_idx = np.where(~dominated)[0]

            if len(nd_idx) > 0:
                # Sort by sum of objectives to prioritize
                sums = np.sum(F_arr[nd_idx], axis=1)
                nd_idx = nd_idx[np.argsort(sums)]

                # Take top candidates for refinement
                n_refine = min(len(nd_idx), 12)
                remaining = budget - used
                if remaining > 0:
                    # For each candidate, try a few random perturbations
                    n_trials_per_point = max(1, remaining // n_refine)
                    for idx in nd_idx[:n_refine]:
                        if used >= budget:
                            break
                        base_x = X_arr[idx]
                        for _ in range(n_trials_per_point):
                            if used >= budget:
                                break
                            # Gaussian perturbation with decreasing scale
                            scale = 0.05 * (hi - lo)
                            perturbation = np.random.randn(dim) * scale
                            new_x = base_x + perturbation
                            new_x = np.clip(new_x, lo, hi)
                            eval_point2(new_x)

        return all_F

# EVOLVE-BLOCK-END
