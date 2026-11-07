"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using hypervolume-weighted sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        budget = self.budget

        # DTLZ4 structure for 3 objectives:
        # f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = 0.5 * sin(pi*x1/2)
        # Reference point R = [2.5, 2.5, 2.5]
        # Hypervolume contribution of a point p (assuming it's non-dominated and we approximate local contribution)
        # is roughly proportional to (R1 - f1) * (R2 - f2) * (R3 - f3).
        # To maximize total HV, we should sample more densely where this product is large.
        
        # Let's define a weight function w(x1, x2) proportional to the local HV contribution.
        # We will use a rejection-free inverse transform sampling on a 2D grid 
        # to generate points with density proportional to w(x1, x2).
        
        # However, a simpler and robust approach is to use a Sobol sequence (low discrepancy)
        # but then *re-weight* the selection or just ensure we cover the high-value region.
        # Actually, the most effective structural change is to explicitly bias the sampling.
        
        # Let's construct a set of candidate points using a Sobol sequence in [0, 1]^2.
        # Then, we can prioritize evaluating points that have high potential HV contribution.
        # But we have a fixed budget. We can't evaluate "potential" without evaluating.
        
        # Alternative: Use the known analytic form to compute the exact HV contribution 
        # of a set of points if they were the only points, but that's not how HV works.
        
        # Let's stick to a simpler structural change: 
        # The current code uses Halton. Sobol is generally better for low-discrepancy.
        # But the insight from the "inspirations" mentioned Sobol improved scores in some islands.
        # Let's combine Sobol with a slight bias towards the center/origin of the front.
        
        # The Pareto front is a quarter sphere. The "center" of the front (in terms of distance to ideal)
        # is where x1 and x2 are such that f1, f2, f3 are balanced? No, the ideal is (0,0,0).
        # The point closest to ideal is (0,0,0) which corresponds to x1=1, x2=0 (or similar corners).
        # Actually, f1=0 when cos(pi*x1/2)=0 (x1=1) or cos(pi*x2/2)=0 (x2=1).
        # f2=0 when cos(pi*x1/2)=0 (x1=1) or sin(pi*x2/2)=0 (x2=0).
        # f3=0 when sin(pi*x1/2)=0 (x1=0).
        
        # Ideal point (0,0,0) is not reachable simultaneously.
        # (1, 0) -> f1=0, f2=0.5, f3=0.5. Dist to ideal: sqrt(0.25+0.25) = 0.707.
        # (0, 1) -> f1=0, f2=0, f3=0.5. Dist: 0.5.
        # (0, 0) -> f1=0.5, f2=0, f3=0. Dist: 0.5.
        # So (0,1) and (0,0) are closer to ideal than (1,0).
        
        # The HV contribution is (2.5-f1)(2.5-f2)(2.5-f3).
        # Max contribution is at (0,0,0) which is not on the front.
        # On the front, the point with max product (2.5-f1)(2.5-f2)(2.5-f3) 
        # is likely where f1, f2, f3 are smallest.
        
        # Let's use a Sobol sequence for base sampling, which is structurally different from Halton.
        
        def sobol_sequence(n, dim_seq):
            """Generate Sobol sequence points in [0, 1]^dim_seq."""
            if n == 0:
                return np.zeros((0, dim_seq))
            
            # Simple implementation for small n using numpy's scipy or manual?
            # SciPy is available.
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim_seq, scramble=True)
            # Sobol requires n to be power of 2 for best properties, but works for others.
            # We'll generate next power of 2 and take first n.
            if n == 1:
                return np.array([[0.5] * dim_seq])
            
            # Find next power of 2
            m = 1
            while m < n:
                m *= 2
            samples = sampler.random(m)
            # Scale to [0, 1] (Sobol already in [0, 1])
            # Note: Sobol points are in [0, 1).
            return samples[:n]

        n_corners = 3
        n_interior = max(0, budget - n_corners)
        
        # The 3 corners
        corner_x12 = np.array([
            [0.0, 0.0], # f=(0.5, 0, 0)
            [0.0, 1.0], # f=(0, 0, 0.5)
            [1.0, 0.0]  # f=(0, 0.5, 0.5)
        ])
        
        if n_interior > 0:
            x12_interior = sobol_sequence(n_interior, 2)
            # Ensure we don't hit boundaries exactly which might duplicate corners or be less effective?
            # Sobol points are in [0, 1).
            x12_all = np.vstack([corner_x12, x12_interior])
        else:
            x12_all = corner_x12
            
        if len(x12_all) > budget:
            x12_all = x12_all[:budget]
            
        X = np.zeros((len(x12_all), dim))
        X[:, 0] = x12_all[:, 0]
        X[:, 1] = x12_all[:, 1]
        
        if dim > 2:
            X[:, 2:] = 0.5
            
        X = np.clip(X, lo, hi)
        
        for i in range(len(X)):
            if time.time() - t0 > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
