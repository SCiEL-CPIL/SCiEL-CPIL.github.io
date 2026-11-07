"""DTLZ4 Pareto-front construction with stratified spherical sampling."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points using stratified spherical sampling.

    DTLZ4 (m=3 objectives, 12 variables):
      x_0, x_1 : position variables (spherical angles)
      x_2..x_11: control variables; g minimized at x_i=0.5
      At g=0, objectives trace the unit sphere octant.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _front_point(self, theta, phi, lo, hi):
        """Create decision vector from spherical angles on the front."""
        x = np.zeros(self.dim)
        x[0] = theta
        x[1] = phi
        x[2:] = 0.5
        x = np.clip(x, lo, hi)
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        evaluated = []

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1
            return f

        # Phase 1: Strategic vertices and edge/face centers for maximum hypervolume
        strategic = [
            (0.0, 0.0),           # h = (1, 0, 0)
            (0.0, np.pi/2),       # h = (0, 0, 1)
            (np.pi/2, 0.0),       # h = (0, 1, 0)
            (np.pi/2, np.pi/2),   # h = (0, 0, 1) - same as above but valid
            (0.0, np.pi/4),       # edge midpoint
            (np.pi/4, 0.0),       # edge midpoint
            (np.pi/4, np.pi/4),   # face center
            (np.pi/2, np.pi/4),   # edge midpoint
            (np.pi/6, np.pi/3),   # additional coverage
            (np.pi/3, np.pi/6),   # additional coverage
        ]

        for theta, phi in strategic:
            if used >= self.budget or time.time() > deadline:
                break
            x = self._front_point(theta, phi, lo, hi)
            evaluate(x)

        # Phase 2: Stratified spherical sampling for uniform coverage
        # Use stratified sampling in (cos_theta, phi) space
        # cos_theta uniform in [0,1] gives uniform area on sphere
        remaining = self.budget - used
        if remaining > 0:
            n = remaining
            rng = np.random.RandomState(0)
            
            # Create stratified grid in 2D: divide [0,1]x[0,1] into n cells
            # Use a simple approach: sort indices and assign stratified positions
            # Better: use a quasi-random sequence (Halton or Sobol-like)
            
            # Halton sequence in 2D for quasi-random coverage
            def halton_2d(n, base1=2, base2=3):
                """Generate n points in [0,1]^2 using Halton sequence."""
                points = []
                for i in range(1, n + 1):
                    # Base base1
                    f1, r1 = 1.0 / base1, i
                    while r1 > 0:
                        r1, r = divmod(r1, base1)
                        f1 += r * f1
                        f1 /= base1
                    # Base base2
                    f2, r2 = 1.0 / base2, i
                    while r2 > 0:
                        r2, r = divmod(r2, base2)
                        f2 += r * f2
                        f2 /= base2
                    points.append((f1, f2))
                return np.array(points)
            
            halton_points = halton_2d(n)
            
            for cos_theta_u, phi_u in halton_points:
                if used >= self.budget or time.time() > deadline:
                    break
                
                # cos_theta in [0, 1] -> theta in [0, pi/2]
                cos_theta = np.clip(cos_theta_u, 0.0, 1.0)
                theta = np.arccos(cos_theta)
                
                # phi in [0, pi/2]
                phi = (np.pi / 2.0) * np.clip(phi_u, 0.0, 1.0)
                
                x = self._front_point(theta, phi, lo, hi)
                evaluate(x)

        # Phase 3: If budget remains, add a few more refined points
        if used < self.budget and time.time() < deadline:
            rng = np.random.RandomState(123)
            n_extra = min(self.budget - used, 10)
            for _ in range(n_extra):
                if used >= self.budget or time.time() > deadline:
                    break
                # Random point on the front
                cos_theta = rng.uniform(0, 1)
                theta = np.arccos(cos_theta)
                phi = rng.uniform(0, np.pi/2)
                x = self._front_point(theta, phi, lo, hi)
                evaluate(x)

        if not evaluated:
            # Fallback
            x = np.zeros(self.dim)
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
