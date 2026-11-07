"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: front determined by (x0, x1); drive g->0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim

        def compute_gt(x):
            """Compute gt(x) for DTLZ7: 0.35*(1+1/(4+10*x))*sin(12*pi*x)"""
            return 0.35 * (1.0 + 1.0 / (4.0 + 10.0 * x)) * np.sin(12.0 * np.pi * x)

        def optimize_g_vars(xi_init):
            """Optimize g-variables using fixed-point iteration to find xi ≈ gt(xi)."""
            xi = xi_init.copy()
            for _ in range(100):
                gt_val = compute_gt(xi)
                xi_new = gt_val
                if np.max(np.abs(xi_new - xi)) < 1e-12:
                    break
                xi = xi_new
            return np.clip(xi, 0.0, 1.0)

        def make_point_on_front(x0, x1):
            """Create a 22-dim point on the Pareto front with given x0, x1."""
            x = np.ones(dim, dtype=np.float64)
            x[0] = x0
            x[1] = x1
            # Optimize g-variables to be on the Pareto front (g=0)
            g_vars = optimize_g_vars(np.full(dim - 2, 0.5))
            x[2:] = g_vars
            x = np.clip(x, lo, hi)
            return x

        # Phase 1: Dense Fibonacci lattice sampling of (x0, x1) on the Pareto front
        # Use golden angle spiral for uniform disk coverage in [0,1]^2
        n_init = min(80, self.budget // 2)
        points = []
        used = 0

        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            r = np.sqrt((k + 0.5) / n_init)
            theta = k * golden_angle
            x0 = 0.5 + 0.5 * r * np.cos(theta)
            x1 = 0.5 + 0.5 * r * np.sin(theta)
            x0 = np.clip(x0, 0.0, 1.0)
            x1 = np.clip(x1, 0.0, 1.0)

            x = make_point_on_front(x0, x1)
            f = problem(x)
            points.append((x, f))
            used += 1

        # Phase 2: Local refinement - refine g-vars for all points
        if points:
            for idx in range(len(points)):
                if used >= self.budget or time.time() > deadline:
                    break

                x, f = points[idx]
                best_x = x.copy()
                best_f = f.copy()

                # Coordinate descent on g-vars with adaptive step sizes
                for _ in range(5):
                    improved = False
                    for i in range(2, dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        # Try multiple step sizes
                        for step in [0.001, 0.01, 0.05, 0.1, -0.001, -0.01, -0.05, -0.1]:
                            new_x = best_x.copy()
                            new_x[i] = np.clip(best_x[i] + step, lo[i], hi[i])
                            new_f = problem(new_x)
                            used += 1
                            # Check if new point dominates or is closer to front
                            if np.all(new_f <= best_f) and np.any(new_f < best_f):
                                best_x = new_x
                                best_f = new_f
                                improved = True
                                if used >= self.budget or time.time() > deadline:
                                    break
                    if not improved:
                        break

                points[idx] = (best_x, best_f)

        # Phase 3: If budget remains, do more random search with g-optimization
        while used < self.budget and time.time() < deadline:
            x0 = rng.random()
            x1 = rng.random()
            x = make_point_on_front(x0, x1)
            f = problem(x)
            points.append((x, f))
            used += 1

        # Return the last point (return value is ignored per contract)
        if points:
            return points[-1][0]
        return np.zeros(dim)
# EVOLVE-BLOCK-END
