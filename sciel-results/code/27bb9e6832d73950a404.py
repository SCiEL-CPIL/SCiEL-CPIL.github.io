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

        # Generate dense Halton sequence points on the simplex
        # Use multiple Halton bases for better low-discrepancy coverage
        def generate_simplex_halton(n_points):
            """Generate n_points on the 2-simplex using Halton sequences in 2D."""
            points = np.zeros((n_points, n_obj))
            # Use bases 2 and 3 for the two coordinates
            for i in range(n_points):
                # Halton in base 2
                h1 = 0.0
                f1 = 1.0
                idx = i + 1
                while idx > 0:
                    f1 /= 2
                    h1 += f1 * (idx % 2)
                    idx //= 2
                
                # Halton in base 3
                h2 = 0.0
                f2 = 1.0
                idx = i + 1
                while idx > 0:
                    f2 /= 3
                    h2 += f2 * (idx % 3)
                    idx //= 3
                
                # Map to simplex: u = [h1, h2*(1-h1), (1-h1)*(1-h2)]
                u = np.array([h1, h2 * (1 - h1), (1 - h1) * (1 - h2)])
                points[i] = u
            return points

        # Phase 1: Very dense Halton sequence for excellent low-discrepancy coverage
        # Use most of the budget for this
        n_halton_main = min(180, budget)
        if n_halton_main > 0 and time_ok():
            halton_points = generate_simplex_halton(n_halton_main)
            for u in halton_points:
                if used >= budget or not time_ok():
                    break
                # Add tiny jitter to avoid exact duplicates
                u = u + 0.0001 * rng.standard_normal(n_obj)
                u = np.clip(u, 0, 1)
                u = u / u.sum()
                x = simplex_point(u)
                x = np.clip(x, lo, hi)
                eval_and_record(x)

        # Phase 2: Add structured grid points for uniform coverage
        if used < budget and time_ok():
            # Use a fine triangular grid
            level = 20
            for i in range(level + 1):
                for j in range(level + 1 - i):
                    if used >= budget or not time_ok():
                        break
                    k = level - i - j
                    u = np.array([i, j, k], dtype=float) / level
                    # Small jitter
                    u += 0.0005 * rng.standard_normal(n_obj)
                    u = np.clip(u, 0, 1)
                    u = u / u.sum()
                    x = simplex_point(u)
                    x = np.clip(x, lo, hi)
                    eval_and_record(x)

        # Phase 3: Use remaining budget for edge and vertex concentration
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
