# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic grid-based sampling with Sobol sequence for WFG4.

    Uses a low-discrepancy Sobol sequence to systematically cover the
    12-dimensional search space, maximizing probability of hitting diverse
    regions of the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0

        dim = self.dim
        n_obj = self.n_obj

        lo = np.empty(dim)
        hi = np.empty(dim)
        for i, (l, h) in enumerate(bounds):
            lo[i] = l
            hi[i] = h
        span = hi - lo

        # Generate Sobol sequence points
        # scipy.stats.qmc.Sobol requires dimension <= 21
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        
        # We need enough points; use next_power_of_two for efficiency
        n_points = self.budget
        # Sobol requires power of 2 for initial points, but we can generate more
        # and take the first n_points
        # Actually, let's just generate n_points worth
        # The Sobol generator can generate any number, but performance is best for powers of 2
        # Let's use the next power of 2 >= n_points for efficiency, then slice
        n_pow2 = 1
        while n_pow2 < n_points:
            n_pow2 *= 2
        
        # Generate Sobol points in [0,1]^d
        sobol_points = sampler.random(n_pow2)
        
        # Take first n_points
        sobol_points = sobol_points[:n_points]
        
        # Map to [lo, hi]
        X = lo + span * sobol_points
        
        # Evaluate all points
        F = np.empty((n_points, n_obj))
        used = 0
        
        for i in range(n_points):
            if time.time() > deadline - 5.0:
                break
            F[i] = problem(X[i])
            used += 1
        
        # Keep only the first 'used' evaluated points
        X = X[:used]
        F = F[:used]
        
        # Compute non-dominated set
        if used == 0:
            return np.empty((0, n_obj))
        
        # Vectorized dominance check
        n = F.shape[0]
        # le[i,j] = F[i] <= F[j] elementwise
        le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
        # lt[i,j] = F[i] < F[j] in at least one component
        lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
        # dom[i,j] = i dominates j
        dom = le & lt
        # dominated[j] = True if some i dominates j
        dominated = np.any(dom, axis=0)
        # Non-dominated indices
        nd_mask = ~dominated
        nd_F = F[nd_mask]
        
        return nd_F
# EVOLVE-BLOCK-END
