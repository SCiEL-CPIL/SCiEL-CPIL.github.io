"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT4: x0 in [-5,5], x1..x9 in [0,1]
        # Pareto front: f1=1 for x1..x9 in [0,1], f2=1+sin(10*pi*x0)
        # f2 ranges from 0 to 2, reference is [11, 400] so all valid points count

        # Strategy:
        # 1. Dense sampling of x0 with x1..x9 in safe zone
        # 2. Local refinement around promising x0 values

        points = []  # Store (x, f) pairs

        def evaluate(x):
            nonlocal points
            f = problem(x)
            points.append((x.copy(), f))
            return f

        # Phase 1: Dense grid on x0 (100 points)
        n_grid = 100
        x0_grid = np.linspace(lo[0], hi[0], n_grid)
        for x0 in x0_grid:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            x[1:] = 0.5  # Safe middle for x1..x9
            evaluate(x)

        # Phase 2: Random sampling with local polish
        n_random = min(80, budget - len(points))
        for _ in range(n_random):
            if time.time() > deadline or len(points) >= budget:
                break

            x = np.zeros(dim)
            x[0] = rng.uniform(lo[0], hi[0])
            x[1:] = rng.uniform(0.1, 0.9, dim - 1)
            f = evaluate(x)

            # Local polish on x0
            step = 0.05
            for _ in range(3):
                if len(points) >= budget or time.time() > deadline:
                    break
                for delta in [-step, step]:
                    if len(points) >= budget:
                        break
                    new_x = x.copy()
                    new_x[0] = np.clip(x[0] + delta, lo[0], hi[0])
                    new_f = problem(new_x)
                    points.append((new_x.copy(), new_f))
                    if new_f[0] <= f[0] and new_f[1] <= f[1] and \
                       (new_f[0] < f[0] or new_f[1] < f[1]):
                        x = new_x
                        f = new_f
                step *= 0.5

        # Phase 3: Fine-tune best regions
        if len(points) < budget:
            # Find points with low f2 (close to 0 or 2) and refine
            sorted_pts = sorted(points, key=lambda p: p[1][1])
            for _, f in sorted_pts[:10]:
                if len(points) >= budget or time.time() > deadline:
                    break
                # Try to find nearby x0 that gives even lower f2
                x0_base = None
                for x, ff in points:
                    if ff[1] == f[1]:
                        x0_base = x[0]
                        break
                if x0_base is None:
                    continue
                for delta in [-0.01, 0.01, -0.02, 0.02]:
                    if len(points) >= budget:
                        break
                    new_x = np.zeros(dim)
                    new_x[0] = np.clip(x0_base + delta, lo[0], hi[0])
                    new_x[1:] = 0.5
                    evaluate(new_x)

        return np.array([p[0] for p in points])
# EVOLVE-BLOCK-END
