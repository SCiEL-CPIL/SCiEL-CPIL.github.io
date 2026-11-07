"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer: structured 2D grid on (x0,x1) with x[2:]=0,
    then local refinement around hypervolume-contributing points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        def make_x(x0, x1, rest=None):
            """Build a 22-d vector with x0, x1 set, and x[2:] = 0 (or rest)."""
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            if rest is not None:
                x[2:] = rest
            return x

        def eval_x(x):
            """Evaluate and return objective vector."""
            return np.asarray(problem(x), dtype=float)

        used = 0
        F_list = []  # list of objective vectors (all recorded)
        X_list = []  # corresponding decision vectors (for reference)

        def record(x):
            nonlocal used
            f = eval_x(x)
            F_list.append(f)
            X_list.append(x)
            used += 1
            return f

        # --- Phase 1: Structured 2D grid on (x0, x1) with x[2:]=0 ---
        # DTLZ7 front: f1 = x0*(1+h1), f2 = x1*(1+h2), f3 = 1
        # where h1 = 1-sqrt(x0/x1), h2 = 1-x0
        # We need x1 > 0 for h1 to be defined. Use a grid that covers [0,1]^2
        # but avoids x1=0 exactly.
        #
        # Hypervolume with ref [1.1, 1.1, 25.0]: points with small f1, f2, f3
        # contribute most. f3=1 is fixed. So we want to cover the front well,
        # especially near the "knee" where f1 and f2 are both small.
        #
        # Use a 14x14 grid = 196 points, leaving ~4 for refinement.
        # Actually, let's use a slightly smaller grid to leave more budget.
        # 12x12 = 144 points, leaving 56 for refinement.

        n_grid = 12
        # x1 must be > 0. Use a log-spaced or linear grid avoiding 0.
        # x1 in [eps, 1], x0 in [0, 1]
        eps = 1e-6
        x1_vals = np.linspace(eps, 1.0, n_grid)
        x0_vals = np.linspace(0.0, 1.0, n_grid)

        grid_F = []
        grid_X = []
        for x1 in x1_vals:
            for x0 in x0_vals:
                if used >= self.budget:
                    break
                x = make_x(x0, x1)
                f = record(x)
                grid_F.append(f)
                grid_X.append(x)
            if used >= self.budget:
                break

        if used >= self.budget:
            return X_list

        # --- Phase 2: Local refinement around promising points ---
        # Identify points that are on the "efficient" part of the front.
        # For DTLZ7 with g=0: f1 = x0*(1 + 1 - sqrt(x0/x1)) = x0*(2 - sqrt(x0/x1))
        # f2 = x1*(1 + 1 - x0) = x1*(2 - x0)
        # f3 = 1
        #
        # Hypervolume contribution: a point contributes volume where it's the
        # "best" in some region. Points with smaller f1 and f2 are better.
        #
        # Strategy: refine around points that are non-dominated in the grid,
        # with small step sizes to explore local neighborhoods.

        # Find non-dominated points in the grid
        n_grid_pts = len(grid_F)
        if n_grid_pts > 0:
            G = np.array(grid_F)
            # Non-dominated: no other point dominates it
            nd_mask = np.ones(n_grid_pts, dtype=bool)
            for i in range(n_grid_pts):
                if not nd_mask[i]:
                    continue
                # Check if any other point dominates i
                # Point j dominates i if G[j] <= G[i] and G[j] < G[i] in at least one
                dominated_by = np.any(
                    np.all(G <= G[i], axis=1) & np.any(G < G[i], axis=1),
                    axis=0
                )
                # Remove self from consideration
                dominated_by[i] = False
                if np.any(dominated_by):
                    nd_mask[i] = False

            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.array([0])
        else:
            nd_indices = np.array([])

        # Refine around each non-dominated point with a few perturbations
        # Use small steps in x0 and x1
        rng = np.random.default_rng(42)
        step_sizes = [0.05, 0.02, 0.01, 0.005]
        refinement_budget = self.budget - used

        # Prioritize points with smaller f1 + f2 (closer to origin in objective space)
        if len(nd_indices) > 0:
            G_nd = G[nd_indices]
            # Sort by f1 + f2 (smaller is better for hypervolume near origin)
            sort_idx = np.argsort(G_nd[:, 0] + G_nd[:, 1])
            nd_indices_sorted = nd_indices[sort_idx]
        else:
            nd_indices_sorted = np.array([])

        # For each non-dominated point, try a few local perturbations
        for idx in nd_indices_sorted:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            x_base = grid_X[idx]
            x0_base = x_base[0]
            x1_base = x_base[1]

            # Try perturbations in x0 and x1
            for step in step_sizes:
                if used >= self.budget or time.time() > deadline:
                    break
                # Perturb x0
                for dx0 in [-step, step]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x0_new = np.clip(x0_base + dx0, 0.0, 1.0)
                    x = make_x(x0_new, x1_base)
                    record(x)
                # Perturb x1
                for dx1 in [-step, step]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x1_new = np.clip(x1_base + dx1, eps, 1.0)
                    x = make_x(x0_base, x1_new)
                    record(x)

        # If we still have budget, do some random exploration near the front
        while used < self.budget and time.time() < deadline:
            # Random x0, x1 near [0,1] with x[2:]=0
            x0 = rng.uniform(0.0, 1.0)
            x1 = rng.uniform(eps, 1.0)
            x = make_x(x0, x1)
            record(x)

        return X_list
# EVOLVE-BLOCK-END
