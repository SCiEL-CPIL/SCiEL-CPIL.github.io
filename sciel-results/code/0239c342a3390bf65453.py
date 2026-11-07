"""Deterministic hierarchical grid search exploiting Katsuura fractal structure."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hierarchical grid-based search for Katsuura function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Track best point and function value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid 3^5 = 243 evaluations over full domain
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)

        # Grid sizes to try in sequence
        grid_sizes = [3, 5, 7, 9, 11]

        for gsize in grid_sizes:
            n_points = gsize ** dim
            if calls_made + n_points > budget:
                # Can't afford this full grid, break
                break

            # Create grid
            axes = [np.linspace(lo[i], hi[i], gsize) for i in range(dim)]
            grids = np.meshgrid(*axes, indexing='ij')

            # Evaluate all grid points
            for idx in np.ndindex(*([gsize] * dim)):
                x = np.array([grids[i][idx] for i in range(dim)])
                evaluate(x)

            if calls_made >= budget:
                break

            if best_x is None:
                break

            # Refine around the best point
            # Shrink the search region
            shrink = 0.5
            new_lo = best_x - (hi - lo) * shrink / 2
            new_hi = best_x + (hi - lo) * shrink / 2
            new_lo = np.clip(new_lo, -5.0, 5.0)
            new_hi = np.clip(new_hi, 5.0, 5.0)

            # Ensure new_lo < new_hi
            for i in range(dim):
                if new_lo[i] >= new_hi[i]:
                    new_lo[i] = best_x[i] - 0.1
                    new_hi[i] = best_x[i] + 0.1

            lo = new_lo
            hi = new_hi

        # Phase 2: Use remaining budget for local refinement
        if best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            local_x = best_x.copy()
            local_f = best_f

            # Adaptive step size local search with deterministic directions
            step = 0.05
            rng = np.random.default_rng(123)

            # Pre-compute a set of deterministic directions
            directions = []
            # Coordinate directions
            for coord in range(dim):
                for sign in [-1, 1]:
                    d = np.zeros(dim)
                    d[coord] = sign
                    directions.append(d)

            # Diagonal directions
            for _ in range(8):
                d = np.ones(dim)
                for coord in range(dim):
                    if _ % 2 == 0:
                        d[coord] = (-1) ** (coord + _ + 1)
                    else:
                        d[coord] = (-1) ** (coord + 1)
                directions.append(d)

            dir_idx = 0
            stall_count = 0
            max_stall = 30

            while calls_made < budget:
                # Try current direction
                move = step * directions[dir_idx % len(directions)]
                y = np.clip(local_x + move, -5.0, 5.0)
                fy = func(y)
                calls_made += 1

                if fy < local_f:
                    local_x = y
                    local_f = fy
                    step *= 1.15
                    stall_count = 0
                    if fy < best_f:
                        best_f = fy
                        best_x = local_x.copy()
                else:
                    step *= 0.75
                    stall_count += 1

                dir_idx += 1

                if step < 1e-9:
                    step = 0.01
                if stall_count > max_stall:
                    step = 0.03
                    stall_count = 0
                    dir_idx = (dir_idx + 5) % len(directions)

        return best_x, best_f
# EVOLVE-BLOCK-END
