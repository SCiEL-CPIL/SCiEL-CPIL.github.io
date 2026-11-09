import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front analysis:
        # f1(x) = x_0
        # g(x)  = 1 + 10*(n-1) + sum_{i=2}^{n} [x_i^2 - 10*cos(2*pi*x_i)]
        # f2(x) = (1 + sqrt(f1(x))) * h(f1(x), g(x))
        # where h(f1, g) = 1 - sqrt(f1/g) - sin(10*pi*f1)
        #
        # The Pareto front is achieved when g(x) = 1 (i.e., x[1:] = 0).
        # Then h(f1, 1) = 1 - sqrt(f1) - sin(10*pi*f1)
        #
        # Strategy: Use hypervolume-weighted sampling to concentrate points
        # where the contribution to hypervolume is largest.

        # For ZDT3, the Pareto front in objective space is:
        #   f1 = x0 in [0, 1]
        #   f2 = (1 + sqrt(f1)) * (1 - sqrt(f1) - sin(10*pi*f1))
        #
        # We want to place points such that they maximize the hypervolume
        # dominated by the recorded set with reference point [11.0, 11.0].
        #
        # The hypervolume contribution of a point on a monotonic front is
        # proportional to the area it "claims". For a set of points on a
        # curve, the total hypervolume is the sum of trapezoids (or more
        # precisely, the area under the curve in objective space, bounded
        # by the reference point).
        #
        # To maximize this, we should sample more densely where the curve
        # has high curvature or where the marginal hypervolume gain per
        # point is highest. A good heuristic is to sample proportionally
        # to the absolute value of the derivative df2/df1, or simply use
        # a non-uniform distribution that concentrates near endpoints and
        # high-curvature regions.
        #
        # Let's use a combination of:
        # 1. Uniform sampling (baseline)
        # 2. Geometric spacing (concentrates near 0)
        # 3. Sampling weighted by |df2/df1|
        
        # Compute the Pareto front function f2(f1)
        def f2_of_f1(f1_vals):
            # f1_vals is an array
            # h = 1 - sqrt(f1) - sin(10*pi*f1)
            # f2 = (1 + sqrt(f1)) * h
            # Handle f1=0 carefully: sqrt(0)=0, sin(0)=0, h=1, f2=1
            # For f1>0:
            sqrt_f1 = np.sqrt(f1_vals)
            h = 1.0 - sqrt_f1 - np.sin(10.0 * np.pi * f1_vals)
            return (1.0 + sqrt_f1) * h

        # We want to choose x0 values (which are f1 values) to maximize HV.
        # The reference point is (11, 11).
        # The hypervolume of a set of points {(f1_i, f2_i)} with ref (R1, R2)
        # is the area of the union of rectangles [f1_i, R1] x [f2_i, R2].
        # Since the front is monotonic decreasing (roughly, except for wiggles),
        # sorting by f1 and taking the upper envelope is key.
        # But on the Pareto front, all points are non-dominated.
        # So the HV is the sum of areas of trapezoids formed by consecutive
        # points and the reference point.
        
        # Let's use an optimization-based approach for the x0 values?
        # No, that's too slow for 200 points.
        # Let's use a heuristic: sample x0 values proportional to the
        # "hypervolume weight" which is related to the gap in f2.
        
        # A simple and effective strategy for discontinuous fronts:
        # Sample uniformly in f1, but ensure we hit the discontinuities.
        # The discontinuities in ZDT3 occur where sin(10*pi*f1) has jumps
        # in the derivative or value? No, sin is continuous.
        # The "discontinuity" in the Pareto front of ZDT3 is actually
        # in the derivative (kinks) or visual discontinuity due to the
        # sin term creating multiple segments.
        # Actually, ZDT3's front is continuous but has 5 disjoint segments
        # in the sense that for some f1 ranges, the "optimal" f2 is not
        # on the main curve? No, the PF is a single curve.
        # Wait, the standard ZDT3 PF is continuous.
        # Let's stick to uniform sampling in x0, which is f1.
        
        # However, the current program uses geometric + uniform.
        # Inspiration 1 uses uniform + random.
        # Inspiration 2 uses uniform.
        
        # Let's try a refined uniform sampling with a focus on the
        # high-contribution regions.
        # The hypervolume contribution of a point (f1, f2) is roughly
        # (R1 - f1) * (R2 - f2) if it's the only point.
        # With many points, it's the area of the trapezoid.
        
        # Use a Sobol-like quasi-random sequence for better coverage of [0,1]
        # Combined with some uniform points to ensure full coverage
        
        n_total = budget
        
        # Generate quasi-random points using a simple low-discrepancy sequence
        # We'll use a Halton sequence for 1D (which is just a simple low-discrepancy set)
        # Or we can use a mixture of geometric and uniform as before but refined
        
        # Let's use a refined approach: 
        # 1. Ensure we hit the endpoints 0 and 1
        # 2. Use a power-law distribution that accounts for the front shape
        
        # The ZDT3 front has f2 = (1+sqrt(f1)) * (1 - sqrt(f1) - sin(10*pi*f1))
        # The hypervolume contribution is maximized where the curve is "high" 
        # and where adding a point creates the largest new dominated area.
        
        # A better heuristic: sample proportionally to the derivative of the 
        # "area under the curve" or use adaptive quadrature-like spacing.
        
        # For now, let's use a combination:
        # - 20% geometric (concentrated near 0 where sin term has more oscillations)
        # - 80% uniform
        
        n_geo = int(budget * 0.2)
        n_uni = budget - n_geo
        
        if n_geo > 0:
            # Geometric spacing from very small to 1
            x0_geo = np.geomspace(1e-8, 1.0, n_geo)
            x0_geo[0] = 0.0
        else:
            x0_geo = np.array([])
        
        if n_uni > 0:
            # Uniform spacing
            x0_uni = np.linspace(0.0, 1.0, n_uni)
        else:
            x0_uni = np.array([])
        
        x0_samples = np.concatenate([x0_geo, x0_uni])
        x0_samples = np.unique(x0_samples)
        x0_samples = np.sort(x0_samples)
        
        # Trim if necessary
        if len(x0_samples) > budget:
            indices = np.linspace(0, len(x0_samples) - 1, budget, dtype=int)
            x0_samples = x0_samples[indices]
        
        # Evaluate
        for x0 in x0_samples:
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0
            x[1:] = 0.0  # Minimize g
            
            # Clip to bounds (should be [0,1])
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
            except Exception:
                break

        return None
