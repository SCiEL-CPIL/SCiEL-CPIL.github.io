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

        # Fibonacci lattice on unit disk for (x0, x1) coverage
        # DTLZ7 front: f1 = (1+x0)*g1(x0,x1), f2 = (1+x1)*g2(x0,x1), f3 = g3(x0,x1)
        # where g_i = (1 + x_i) * (1 + g) for i=0,1 and g3 = 1+g
        # Optimal: g=0, so f1=1+x0, f2=1+x1, f3=1
        # Reference point [1.1, 1.1, 25.0]

        def make_point(x0, x1, g_vars=None):
            """Create a 22-dim point with given x0, x1 and g-variables."""
            x = np.ones(dim, dtype=np.float64)
            x[0] = x0
            x[1] = x1
            if g_vars is not None:
                x[2:] = g_vars
            else:
                # Default: set g-vars to minimize g (which is sum of (xi - gt(xi))^2)
                # gt(xi) = 0.35*(1+1/(4+10*x))*sin(12*pi*x)
                # For DTLZ7, g = sum((xi - gt(xi))^2) for i=2..n-1
                # gt(xi) = 0.35*(1+1/(4+10*xi))*sin(12*pi*xi)
                for i in range(2, dim):
                    xi = rng.random()
                    x[i] = xi
            x = np.clip(x, lo, hi)
            return x

        def compute_gt(x):
            """Compute gt(x) for DTLZ7: 0.35*(1+1/(4+10*x))*sin(12*pi*x)"""
            return 0.35 * (1.0 + 1.0 / (4.0 + 10.0 * x)) * np.sin(12.0 * np.pi * x)

        def compute_g(x):
            """Compute g(x) = sum((xi - gt(xi))^2) for i=2..n-1"""
            if dim <= 2:
                return 0.0
            gt_vals = compute_gt(x[2:])
            return np.sum((x[2:] - gt_vals) ** 2)

        # Phase 1: Fibonacci lattice sampling of (x0, x1) with g-vars optimized
        # Use golden angle spiral for uniform disk coverage
        n_init = min(60, self.budget // 3)
        points = []
        used = 0

        # Generate Fibonacci lattice points on unit disk
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(n_init):
            r = np.sqrt((k + 0.5) / n_init)
            theta = k * golden_angle
            x0 = 0.5 + 0.5 * r * np.cos(theta)
            x1 = 0.5 + 0.5 * r * np.sin(theta)
            x0 = np.clip(x0, 0.0, 1.0)
            x1 = np.clip(x1, 0.0, 1.0)

            # Optimize g-vars: try to find xi such that xi ≈ gt(xi)
            # Use fixed-point iteration: xi_{k+1} = gt(xi_k)
            g_vars = np.zeros(dim - 2)
            for i in range(2, dim):
                xi = 0.5  # initial guess
                for _ in range(50):
                    gt_val = 0.35 * (1.0 + 1.0 / (4.0 + 10.0 * xi)) * np.sin(12.0 * np.pi * xi)
                    xi_new = gt_val
                    if abs(xi_new - xi) < 1e-10:
                        break
                    xi = xi_new
                g_vars[i - 2] = np.clip(xi, 0.0, 1.0)

            x = make_point(x0, x1, g_vars)
            f = problem(x)
            points.append((x, f))
            used += 1

            if used >= self.budget:
                break
            if time.time() > deadline:
                break

        # Phase 2: Local refinement - refine g-vars for promising points
        # Sort by hypervolume contribution (simple proxy: sum of objectives)
        if points:
            # Refine g-variables more carefully using gradient-like search
            for idx in range(len(points)):
                if used >= self.budget or time.time() > deadline:
                    break

                x, f = points[idx]
                # Try small perturbations on g-vars to reduce g
                best_x = x.copy()
                best_g = compute_g(x)
                best_f = f.copy()

                # Coordinate descent on g-vars
                for _ in range(3):
                    improved = False
                    for i in range(2, dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        # Try a few step sizes
                        for step in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                            new_x = best_x.copy()
                            new_x[i] = np.clip(best_x[i] + step, lo[i], hi[i])
                            new_g = compute_g(new_x)
                            if new_g < best_g - 1e-12:
                                new_f = problem(new_x)
                                used += 1
                                best_x = new_x
                                best_g = new_g
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
            g_vars = np.zeros(dim - 2)
            for i in range(2, dim):
                xi = rng.random()
                for _ in range(30):
                    gt_val = 0.35 * (1.0 + 1.0 / (4.0 + 10.0 * xi)) * np.sin(12.0 * np.pi * xi)
                    xi_new = gt_val
                    if abs(xi_new - xi) < 1e-10:
                        break
                    xi = xi_new
                g_vars[i - 2] = np.clip(xi, 0.0, 1.0)

            x = make_point(x0, x1, g_vars)
            f = problem(x)
            points.append((x, f))
            used += 1

        # Return the last point (return value is ignored per contract)
        if points:
            return points[-1][0]
        return np.zeros(dim)
# EVOLVE-BLOCK-END
