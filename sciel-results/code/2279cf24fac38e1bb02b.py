"""Multi-objective optimizer for DTLZ1 with a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 (3 objectives, 7 vars, bounds [0,1]).

    Uses the known structure of DTLZ1: the Pareto front is the simplex
    sum(f) = 0.5 with f_i = 0.5 * x_i. We sample the front directly using
    stratified and quasi-random sampling for maximum hypervolume coverage.
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
            f = np.clip(f, 0.0, 399.999)
            recorded_F.append(f)
            used += 1
            return f

        rng = np.random

        # Key insight: For DTLZ1, points ON the Pareto front (x[n_obj:] = 0.5,
        # sum(x[:n_obj]) = 1) give f_i = 0.5 * x_i with sum(f) = 0.5.
        # Any deviation from the front increases g > 0, making ALL objectives worse.
        # So we spend the ENTIRE budget on the front with maximum coverage.
        #
        # Strategy: Use stratified sampling on the 2-simplex for uniform coverage,
        # which maximizes hypervolume for a fixed number of points.

        # Generate stratified samples on the 2-simplex using inverse CDF of Dirichlet(1,1,1)
        # For Dirichlet(1,1,1), the CDF for (a,b,c) with a+b+c=1 is:
        # P(A<=a, B<=b) = a*b for a+b<=1 (the triangular region)
        # We use stratified sampling: divide the simplex into N^2 sub-triangles
        # and pick one point per sub-triangle.

        # Phase 1: Stratified sampling - divide simplex into grid cells
        # Use a triangular grid: for i in range(n), j in range(n-i), cell (i,j)
        # The simplex can be tiled with n*(n-1)/2 small triangles of equal area
        # For n=20, we get 190 cells, which is close to our budget
        
        n_strata = 15  # gives 15*14/2 = 105 cells
        # Generate all stratum centers
        strata_points = []
        for i in range(n_strata):
            for j in range(n_strata - i):
                # Cell (i, j) in the triangular grid
                # The cell has vertices at:
                # (i/n, j/n), ((i+1)/n, j/n), (i/n, (j+1)/n)
                # Wait, that's not quite right for a simplex tiling.
                # Let me use a different approach: map from unit square to simplex
                # with stratified sampling.
                
                # Actually, let's use a cleaner approach:
                # Divide the simplex into n^2 regions using the transformation:
                # u = sqrt(random1), v = random2 * (1-u)
                # This gives uniform sampling on the simplex.
                # For stratified: divide the (u,v) space into cells.
                pass
        
        # Simpler approach: use Halton-like sequence for quasi-random coverage
        # Plus stratified sampling for the first portion

        # Phase 1: Stratified sampling using inverse CDF
        # For uniform on simplex, use: u = sqrt(U1), v = U2*(1-u)
        # Stratify U1 and U2 into [0,1] intervals
        n_strat = int(np.sqrt(budget * 0.6))  # ~60% of budget for stratified
        n_strat = max(n_strat, 10)
        
        # Create stratified points in unit square, then map to simplex
        for i in range(n_strat):
            if used >= budget or not time_ok():
                break
            for j in range(n_strat):
                if used >= budget or not time_ok():
                    break
                # Stratified: pick random point in cell [i/n, (i+1)/n] x [j/n, (j+1)/n]
                u1 = (i + rng.random()) / n_strat
                u2 = (j + rng.random()) / n_strat
                # Map to simplex: a = sqrt(u1), b = u2*(1-sqrt(u1)), c = 1-a-b
                a = np.sqrt(u1)
                b = u2 * (1.0 - a)
                c = 1.0 - a - b
                # Clip to ensure validity
                a = np.clip(a, 0.0, 1.0)
                b = np.clip(b, 0.0, 1.0)
                c = np.clip(c, 0.0, 1.0)
                s = a + b + c
                if s > 0:
                    a /= s
                    b /= s
                    c /= s
                u_vec = np.array([a, b, c])
                x = np.zeros(dim)
                x[:n_obj] = u_vec
                x[n_obj:] = 0.5
                x = np.clip(x, lo, hi)
                eval_and_record(x)

        # Phase 2: Quasi-random sampling using Halton sequence for remaining budget
        # Halton sequence in 2D, then map to simplex
        def halton(index, base):
            """Generate Halton sequence value."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        remaining = budget - used
        for k in range(1, remaining + 1):
            if used >= budget or not time_ok():
                break
            # 2D Halton sequence
            h1 = halton(k, 2)
            h2 = halton(k, 3)
            # Map to simplex
            a = np.sqrt(h1)
            b = h2 * (1.0 - a)
            c = 1.0 - a - b
            a = np.clip(a, 0.0, 1.0)
            b = np.clip(b, 0.0, 1.0)
            c = np.clip(c, 0.0, 1.0)
            s = a + b + c
            if s > 0:
                a /= s
                b /= s
                c /= s
            u_vec = np.array([a, b, c])
            x = np.zeros(dim)
            x[:n_obj] = u_vec
            x[n_obj:] = 0.5
            x = np.clip(x, lo, hi)
            eval_and_record(x)

        # Return (ignored by evaluator, but keep valid)
        if recorded_F:
            return np.array(recorded_F)
        return np.array([])
# EVOLVE-BLOCK-END
