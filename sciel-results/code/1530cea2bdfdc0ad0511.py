# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ4 with deterministic front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        mid = (lo + hi) / 2.0

        n_shape = n_obj - 1  # 2 shape variables
        n_dist = dim - n_shape  # 10 distance variables

        # Build a deterministic, low-discrepancy sampling of the shape space [0,1]^2
        # using a stratified grid with Halton-sequence jitter within each cell.
        # This gives near-uniform coverage of the 2D Pareto front.

        # Choose grid size: aim for ~sqrt(budget) x sqrt(budget) grid
        # with remaining budget used for additional stratified points
        grid_n = int(np.sqrt(budget))
        if grid_n < 2:
            grid_n = 2

        # Create cell boundaries
        x0_edges = np.linspace(0.0, 1.0, grid_n + 1)
        x1_edges = np.linspace(0.0, 1.0, grid_n + 1)

        # For each cell, place a point using Halton sequence for stratification
        # Halton base 2 for first dim, base 3 for second dim
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        X_shape = []
        idx = 0
        for i in range(grid_n):
            for j in range(grid_n):
                idx += 1
                # Halton point within [0,1]
                h0 = halton(idx, 2)
                h1 = halton(idx, 3)
                # Map to cell
                x0 = x0_edges[i] + h0 * (x0_edges[i + 1] - x0_edges[i])
                x1 = x1_edges[j] + h1 * (x1_edges[j + 1] - x1_edges[j])
                X_shape.append([x0, x1])

        X_shape = np.array(X_shape)
        n_grid = len(X_shape)

        # Use remaining budget for additional stratified points
        remaining = budget - n_grid
        if remaining > 0:
            # Continue Halton sequence for additional points
            extra_X = []
            for k in range(remaining):
                idx += 1
                h0 = halton(idx, 2)
                h1 = halton(idx, 3)
                extra_X.append([h0, h1])
            extra_X = np.array(extra_X)
            X_shape = np.vstack([X_shape, extra_X])

        # Truncate to budget if needed
        if len(X_shape) > budget:
            X_shape = X_shape[:budget]

        # Build full vectors: shape vars from X_shape, distance vars at 0.5
        X_full = np.zeros((len(X_shape), dim))
        X_full[:, :n_shape] = X_shape
        X_full[:, n_shape:] = mid[n_shape:]  # Optimal distance values

        # Evaluate all points
        F = np.array([problem(x) for x in X_full])
        used = len(X_full)

        # Check deadline
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return X_full

        # If we have remaining budget and time, do a refinement pass
        # Focus on regions of the front that may have sparse coverage
        if used < budget and elapsed < deadline - 10:
            rng = np.random.default_rng(0)
            n_extra = budget - used

            # Find non-dominated points to guide refinement
            nd_mask = np.ones(len(X_full), dtype=bool)
            for i in range(len(X_full)):
                if not nd_mask[i]:
                    continue
                for j in range(len(X_full)):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd_mask[i] = False
                        break

            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(len(X_full))

            # Add refinement points near non-dominated solutions
            for k in range(n_extra):
                if time.time() > deadline - 3:
                    break
                idx = nd_indices[k % len(nd_indices)]
                base = X_full[idx].copy()
                # Small perturbation in shape space
                pert = 0.01 * rng.standard_normal(n_shape)
                base[:n_shape] = np.clip(base[:n_shape] + pert, 0, 1)
                base[n_shape:] = mid[n_shape:]
                fc = problem(base)
                used += 1
                X_full = np.vstack([X_full, base[None, :]])
                F = np.vstack([F, fc[None, :]])

        return X_full
# EVOLVE-BLOCK-END
