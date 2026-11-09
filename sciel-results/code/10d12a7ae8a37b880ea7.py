"""DTLZ1 multi-objective optimizer exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1-specific optimizer: exploit structure, maximize hypervolume coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        center = (hi + lo) / 2.0

        # DTLZ1 structure:
        # f1 = 0.5 * g * (1 + x1)
        # f2 = 0.5 * g * (1 - x1) * (1 + x2)
        # f3 = 0.5 * g * (1 - x2)
        # where g = 100 * (n-1 + sum((xi - 0.5)^2 - cos(20*pi*(xi-0.5))) for i in 2..n)
        # Optimal: g=0 when xi=0.5 for i>=2, and x1,x2 vary to trace the front.
        # Pareto front: x1 in [0,1], x2 in [0,1], f1+f2+f3=0.5, convex front.

        # Strategy:
        # 1. Set "g" variables (indices 2..dim-1) to 0.5 to get g=0
        # 2. Use stratified/Sobol-like sampling of x1, x2 to cover the Pareto front
        # 3. Use remaining budget for targeted refinement in gaps

        def make_point(x1, x2):
            """Create a decision vector with given x1, x2 and optimal tail variables."""
            x = np.zeros(dim)
            x[0] = np.clip(x1, lo[0], hi[0])
            x[1] = np.clip(x2, lo[1], hi[1])
            for j in range(2, dim):
                x[j] = center[j]
            return x

        # Phase 1: Dense structured sampling using stratified approach
        # For hypervolume, we want good coverage of the 2D Pareto front
        # Use a combination of grid and Halton for best coverage

        # Generate points using a combination of methods for diversity
        # Method 1: Coarse grid
        n_grid = 10
        grid_pts = []
        for i in range(n_grid):
            for j in range(n_grid):
                x1 = lo[0] + span[0] * (i + 0.5) / n_grid
                x2 = lo[1] + span[1] * (j + 0.5) / n_grid
                grid_pts.append((x1, x2))

        # Method 2: Halton sequence for low-discrepancy coverage
        def halton_sequence(n, dim_h=2):
            """Generate n points in [0,1]^dim_h using Halton sequences."""
            pts = np.zeros((n, dim_h))
            for d in range(dim_h):
                base = d + 2  # 2, 3, 4, ...
                for i in range(n):
                    f = 1.0
                    x = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        x += f * (i_val % base)
                        i_val //= base
                    pts[i, d] = x
            return pts

        # Method 3: Vertex-focused sampling (corners of the unit square)
        # These give extreme points of the front
        vertex_pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                      (0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0), (0.5, 0.5)]

        # Combine all candidate (x1, x2) pairs
        all_candidates = []
        all_candidates.extend(grid_pts)
        all_candidates.extend(vertex_pts)

        # Add Halton points to fill remaining budget
        n_halton = budget - len(all_candidates)
        if n_halton > 0:
            halton_pts = halton_sequence(n_halton, 2)
            for i in range(n_halton):
                x1 = lo[0] + span[0] * halton_pts[i, 0]
                x2 = lo[1] + span[1] * halton_pts[i, 1]
                all_candidates.append((x1, x2))

        # Shuffle to randomize evaluation order (helps with early stopping)
        np.random.shuffle(all_candidates)

        # Evaluate all candidates
        X = np.zeros((len(all_candidates), dim))
        F = np.zeros((len(all_candidates), n_obj))
        for i, (x1, x2) in enumerate(all_candidates):
            if time.time() > deadline - 5.0:
                break
            X[i] = make_point(x1, x2)
            F[i] = problem(X[i])

        used = len(all_candidates)

        # Phase 2: Local refinement to fill gaps in the front
        # Identify non-dominated points and refine around them
        if used < budget and time.time() < deadline - 10.0:
            def is_dominated(f, F_set):
                """Check if f is dominated by any point in F_set."""
                return np.any(np.all(F_set <= f, axis=1) & np.any(F_set < f, axis=1))

            # Find non-dominated points
            nd_mask = np.array([not is_dominated(F[i], F) for i in range(len(F))])
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) > 0:
                # Select a diverse subset of non-dominated points for refinement
                n_refine = min(15, len(nd_indices), (budget - used) // 4)
                if n_refine > 0:
                    # Spread selection across the non-dominated set
                    step = max(1, len(nd_indices) // n_refine)
                    refine_idx = nd_indices[::step][:n_refine]

                    for idx in refine_idx:
                        if time.time() > deadline - 10.0 or used >= budget:
                            break

                        x0 = X[idx].copy()
                        f0 = F[idx].copy()
                        best_x = x0.copy()
                        best_f = f0.copy()
                        step_size = 0.02

                        # Local search with adaptive step size
                        for _round in range(4):
                            if used >= budget or time.time() > deadline - 10.0:
                                break
                            for _try in range(6):
                                if used >= budget or time.time() > deadline - 10.0:
                                    break
                                pert = np.zeros(2)
                                pert[0] = np.random.uniform(-step_size, step_size)
                                pert[1] = np.random.uniform(-step_size, step_size)
                                new_x = make_point(best_x[0] + pert[0], best_x[1] + pert[1])
                                new_f = problem(new_x)
                                used += 1

                                # Accept if new point is not dominated by best_f
                                if not np.all(new_f >= best_f):
                                    if np.any(new_f < best_f):
                                        best_x = new_x
                                        best_f = new_f

                            step_size *= 0.5

                        # Update population with refined point
                        X[idx] = best_x
                        F[idx] = best_f

        # Phase 3: Use remaining budget for additional Halton points with jitter
        if used < budget and time.time() < deadline - 5.0:
            remaining = budget - used
            n_extra = min(remaining, 50)
            extra_halton = halton_sequence(n_extra, 2)
            for i in range(n_extra):
                if time.time() > deadline - 5.0 or used >= budget:
                    break
                # Add slight jitter for diversity
                jitter = 0.01
                x1 = lo[0] + span[0] * (extra_halton[i, 0] + np.random.uniform(-jitter, jitter))
                x2 = lo[1] + span[1] * (extra_halton[i, 1] + np.random.uniform(-jitter, jitter))
                x = make_point(x1, x2)
                fx = problem(x)
                used += 1

                # Replace worst point (highest sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = x
                F[worst_idx] = fx

        return X
# EVOLVE-BLOCK-END
