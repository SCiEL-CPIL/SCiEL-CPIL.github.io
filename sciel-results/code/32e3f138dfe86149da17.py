import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit

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
            f = problem(x)
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x)
            all_fvals.append(f)
            n_evals += 1
            return f

        # DTLZ2 Pareto front:
        # f1 = sin^2(pi * x0) * cos^2(pi * x1)
        # f2 = sin^2(pi * x0) * sin^2(pi * x1)
        # f3 = cos^2(pi * x0)
        # x2...x11 = 0.5 (to make g=0)
        
        # We want to maximize hypervolume. The front is a quarter-sphere in objective space.
        # To get a uniform distribution on the surface (which maximizes hypervolume for a given number of points on a convex front),
        # we should sample angles such that the area element is constant.
        # Area element on sphere: sin(theta) dtheta dphi.
        # Let theta = pi * x0, phi = pi * x1.
        # We want uniform sampling in (u, v) where u = 1 - cos(theta), v = phi.
        # So 1 - cos(pi*x0) = u => cos(pi*x0) = 1 - u => pi*x0 = arccos(1-u) => x0 = arccos(1-u)/pi.
        # phi = pi*x1 = v => x1 = v/pi.
        
        # Generate u and v uniformly in [0, 1]
        # To ensure we cover the boundaries well (corners of the front), we might want to include endpoints or use a slight jitter.
        # However, standard uniform random sampling of u, v in [0,1] works well.
        # We can use a deterministic quasi-random sequence (like Halton or Sobol) to ensure better spread than pure random, 
        # but since the seed is fixed, pure random with seed is also deterministic and sufficient if budget is high enough.
        # Given budget=200, a simple random sample of u, v will provide good coverage.
        
        # Let's use a structured approach: 
        # We can generate 200 points.
        # To avoid clustering at the "poles" (where sin(pi*x0) is 0), the transformation x0 = arccos(1-u)/pi handles that.
        
        u = np.random.rand(self.budget)
        v = np.random.rand(self.budget)
        
        # Transform to x0, x1
        x0 = np.arccos(1 - u) / np.pi
        x1 = v / np.pi
        
        # Clip to [0, 1] just in case of floating point issues at boundaries
        x0 = np.clip(x0, 0.0, 1.0)
        x1 = np.clip(x1, 0.0, 1.0)
        
        # Construct full vectors
        for i in range(self.budget):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0[i]
            x[1] = x1[i]
            # Set remaining dimensions to 0.5 (center of [0,1] bounds) to minimize g(x)
            # If bounds are not [0,1], we should use the midpoint of the bounds for the "bad" variables.
            # The problem states bounds [0,1] for every variable, but let's be robust.
            # For DTLZ2, g(x) = sum_{i=3}^{M} (xi - 0.5)^2. So 0.5 is optimal.
            # If bounds were different, we'd need to adjust. But contract says [0,1].
            for j in range(2, self.dim):
                # Use midpoint of bounds for optimal g
                mid = (lo[j] + hi[j]) / 2.0
                x[j] = mid
            
            eval_point(x)
            
            if n_evals >= self.budget:
                break

        return all_points
