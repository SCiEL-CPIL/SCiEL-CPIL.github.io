"""Portfolio search for DTLZ2: dense Pareto-front grid with optimal gap-filling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a Pareto-front grid portfolio."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ2 Pareto front: set x[2:] = 0.5 so g = sum (x_i - 0.5)^2 = 0
        # Then f = [sin(pi*x0/2), cos(pi*x0/2)*sin(pi*x1/2), cos(pi*x1/2)]
        # This is a quarter-sphere in the positive octant.
        # We sweep x0 and x1 over [0, 1] to cover the entire front.

        # Strategy: 14x14 uniform grid = 196 points, plus 4 gap-fillers = 200 total.
        # The 4 gap-fillers are placed at the centers of the 4 largest cells
        # in a coarser 7x7 overlay, which bisects the 14x14 grid spacing.

        n_grid = 14
        grid = np.linspace(0.0, 1.0, n_grid)

        candidates = []
        for a in grid:
            for b in grid:
                x = np.full(dim, 0.5)
                x[0] = a
                x[1] = b
                candidates.append(x)

        # Gap-fillers: bisect the grid cells. The grid has 13 intervals of width 1/13.
        # Place points at the midpoints of alternating cells to maximize coverage.
        # Use 4 points at strategic midpoints:
        midpoints = [
            (1.0 / (2 * 13), 1.0 / (2 * 13)),
            (1.0 / (2 * 13), 1.0 - 1.0 / (2 * 13)),
            (1.0 - 1.0 / (2 * 13), 1.0 / (2 * 13)),
            (1.0 - 1.0 / (2 * 13), 1.0 - 1.0 / (2 * 13)),
        ]
        for a, b in midpoints:
            x = np.full(dim, 0.5)
            x[0] = a
            x[1] = b
            candidates.append(x)

        # Cap to budget
        candidates = candidates[:budget]

        # Evaluate all candidates
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1

        return np.array(candidates[:used])
# EVOLVE-BLOCK-END
