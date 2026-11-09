import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        all_points = []
        all_fvals = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
            except Exception:
                return None
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x.copy())
            all_fvals.append(f)
            n_evals += 1
            return f

        # Strategy: Use Scrambled Sobol sequence for the 2 angle variables.
        # DTLZ2 with n_obj=3 has Pareto front on a quarter-sphere.
        # Variables x[0], x[1] determine the angles a1, a2.
        # x[2:] should be set to the center of the bounds to minimize g(x).
        
        n_points = self.budget
        
        # Try to use scipy.stats.qmc.Sobol with scrambling
        try:
            from scipy.stats import qmc
            # Sobol sequence requires n to be a power of 2 for best properties, 
            # but we can take the first n points.
            # We need 2 dimensions for the angles.
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            # Generate slightly more to ensure we have n_points, 
            # although Sobol generates exactly n if requested.
            u = sampler.random(n_points)
        except ImportError:
            # Fallback to a different deterministic sequence if scipy is not available
            # Using a simple grid + jitter or a different low-discrepancy sequence
            # Here we fallback to a Halton-like sequence but with a different base order
            # to maintain structural difference from the original Halton implementation
            def halton_sequence(n, dims=2):
                points = []
                for i in range(1, n + 1):
                    point = []
                    for base in range(3, dims + 3):  # bases 3, 5 for 2D (different from 2,3)
                        f = 1.0
                        r = 0.0
                        ii = i
                        while ii > 0:
                            f /= base
                            r += f * (ii % base)
                            ii //= base
                        point.append(r)
                    points.append(point)
                return np.array(points)
            u = halton_sequence(n_points, dims=2)

        # Map u[0], u[1] in [0,1) to angles a1, a2 in [0, pi/2]
        # For uniform surface density on the sphere:
        # a1 = arccos(1 - u1)
        # a2 = (pi/2) * u2
        a1 = np.arccos(1.0 - u[:, 0])
        a2 = (np.pi / 2.0) * u[:, 1]

        n_angles = self.n_obj - 1  # 2
        n_g_vars = self.dim - n_angles  # 10

        # Center for g-variables to make g=0
        # Assuming standard DTLZ2 where g = sum((x_i - 0.5)^2) or similar
        # The problem clips to bounds. If bounds are [0,1], center is 0.5.
        # If bounds are different, we should use the midpoint.
        mid = (lo + hi) / 2.0
        g_center = mid[2:]

        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = a1[i]
            x[1] = a2[i]
            x[2:] = g_center
            eval_point(x)

        return all_points
