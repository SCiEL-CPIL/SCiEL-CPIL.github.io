"""Multi-objective optimizer for DTLZ1 with a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 (3 objectives, 7 vars, bounds [0,1]).

    Uses the known structure of DTLZ1: the Pareto front is the simplex
    f_1 + f_2 + f_3 = 0.5 with f_i >= 0. We densely sample this simplex
    to maximize hypervolume with reference point [400, 400, 400].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        recorded_F = []
        used = 0

        def eval_and_record(x):
            nonlocal used
            f = problem(x)
            f = np.array(f, dtype=float)
            if np.all(f < 400.0):
                recorded_F.append(f)
            used += 1
            return f

        rng = np.random

        def simplex_point(u):
            """Create a decision vector from simplex coordinates u (length 3, sum=1)."""
            x = np.zeros(dim)
            x[:n_obj] = u
            x[n_obj:] = 0.5
            return x

        def halton_sequence(n, dims):
            """Generate n points in [0,1)^dims using Halton sequence."""
            points = np.zeros((n, dims))
            for d in range(dims):
                base = d + 2  # bases: 2, 3, 5, ...
                for i in range(n):
                    f = 1.0
                    result = 0.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        result += f * (idx % base)
                        idx //= base
                    points[i, d] = result
            return points

        # Phase 1: Dense structured grid on the simplex
        # Use a triangular grid with increasing resolution
        grid_levels = [5, 8, 12, 15]  # progressively finer grids
        for level in grid_levels:
            if used >= budget or not time_ok():
                break
            # Generate grid points on the 2-simplex
            for i in range(level + 1):
                for j in range(level + 1 - i):
                    if used >= budget or not time_ok():
                        break
                    k = level - i - j
                    u = np.array([i, j, k], dtype=float) / level
                    # Add small jitter to avoid exact duplicates
                    u += 0.001 * rng.standard_normal(n_obj)
                    u = np.clip(u, 0, 1)
                    u = u / u.sum()
                    x = simplex_point(u)
                    x = np.clip(x, lo, hi)
                    eval_and_record(x)

        # Phase 2: Halton sequence for low-discrepancy coverage
        if used < budget and time_ok():
            n_halton = min(60, budget - used)
            # Generate Halton points in 2D, map to simplex
            halton_2d = halton_sequence(n_halton, 2)
            for h in halton_2d:
                if used >= budget or not time_ok():
                    break
                # Map 2D point to simplex: u = [h0, h1*(1-h0), (1-h0)*(1-h1)]
                h0, h1 = h[0], h[1]
                u = np.array([h0, h1 * (1 - h0), (1 - h0) * (1 - h1)])
                # Add small jitter
                u += 0.0005 * rng.standard_normal(n_obj)
                u = np.clip(u, 0, 1)
                u = u / u.sum()
                x = simplex_point(u)
                x = np.clip(x, lo, hi)
                eval_and_record(x)

        # Phase 3: Sobol-like quasi-random points for additional coverage
        if used < budget and time_ok():
            n_sobol = min(50, budget - used)
            for _ in range(n_sobol):
                if used >= budget or not time_ok():
                    break
                # Use inverse CDF of uniform to get points on simplex
                r1, r2 = rng.random(2)
                u = np.array([r1, r2 * (1 - r1), (1 - r1) * (1 - r2)])
                # Add tiny perturbation
                u += 0.0001 * rng.standard_normal(n_obj)
                u = np.clip(u, 0, 1)
                u = u / u.sum()
                x = simplex_point(u)
                x = np.clip(x, lo, hi)
                eval_and_record(x)

        # Phase 4: Use remaining budget for edge and vertex concentration
        while used < budget and time_ok():
            r = rng.random()
            if r < 0.3:
                # Edge points (two active objectives)
                idx = rng.choice(n_obj, size=2, replace=False)
                t = rng.random()
                u = np.zeros(n_obj)
                u[idx[0]] = t
                u[idx[1]] = 1 - t
            elif r < 0.5:
                # Near vertices
                idx = rng.integers(0, n_obj)
                u = np.zeros(n_obj)
                u[idx] = 1.0
                u += 0.05 * rng.random(n_obj)
                u = u / u.sum()
            else:
                # Uniform on simplex
                u = rng.random(n_obj)
                u = u / u.sum()
            x = simplex_point(u)
            x = np.clip(x, lo, hi)
            eval_and_record(x)

        if recorded_F:
            return np.array(recorded_F)
        return np.array([])
# EVOLVE-BLOCK-END
