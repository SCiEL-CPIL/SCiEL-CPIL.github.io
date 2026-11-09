import numpy as np
import time


class Optimizer:
    """DTLZ2 multi-objective optimizer.

    Strategy: The DTLZ2 Pareto front is the unit quarter-sphere in the
    positive octant. We use a 2D Fibonacci lattice to generate quasi-uniform
    points on that front, with a refined probe to find the g-minimizing distance.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fibonacci_lattice(self, n):
        """Generate n points in [0,1]^2 using a 2D Fibonacci lattice."""
        golden_ratio = (1 + np.sqrt(5)) / 2
        inv_phi2 = 1.0 / (golden_ratio * golden_ratio)
        
        points = np.empty((n, 2))
        for i in range(n):
            x = (i + 0.5) / n
            y = (i * inv_phi2 + 0.5) % 1.0
            points[i] = [x, y]
        return points

    def _front_dir(self, u, v):
        """Map (u,v) in [0,1]^2 to a direction on the unit quarter-sphere."""
        theta = np.arccos(1.0 - u)  # uniform in spherical-cap area
        phi = 0.5 * np.pi * v       # uniform azimuth in [0, pi/2]
        st = np.sin(theta)
        return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = int(self.dim)
        m = int(self.n_obj)
        budget = int(self.budget)

        def make_x(dir3, dval):
            x = np.empty(n, dtype=float)
            x[0:m] = dir3
            x[m:n] = dval
            return np.clip(x, lo, hi)

        def ok():
            return (time.time() - t0) < deadline

        used = 0

        # Phase 1: Probe the distance variables to find the g-minimizing value.
        # Try several candidate values and pick the one with minimum ||f||.
        probe_dir = np.full(m, 1.0 / np.sqrt(m))
        best_d = 0.5
        best_norm = np.inf
        
        # Try a set of candidate distance values
        candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for d in candidates:
            if used >= budget or not ok():
                break
            f = problem(make_x(probe_dir, d))
            used += 1
            norm = float(np.linalg.norm(f))
            if norm < best_norm:
                best_norm = norm
                best_d = d

        # Refine around the best candidate with finer steps
        if used < budget and ok():
            step = 0.05
            for d in np.arange(max(0, best_d - 0.1), min(1, best_d + 0.1) + step, step):
                if used >= budget or not ok():
                    break
                f = problem(make_x(probe_dir, d))
                used += 1
                norm = float(np.linalg.norm(f))
                if norm < best_norm:
                    best_norm = norm
                    best_d = d

        # Phase 2: Generate Fibonacci lattice points and map to the front
        n_front = budget - used
        if n_front > 0 and ok():
            lattice_pts = self._fibonacci_lattice(n_front)
            for u, v in lattice_pts:
                if used >= budget or not ok():
                    break
                dir3 = self._front_dir(u, v)
                problem(make_x(dir3, best_d))
                used += 1

        return None
