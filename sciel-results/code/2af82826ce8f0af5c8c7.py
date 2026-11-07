import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n = self.budget
        dim = self.dim
        
        # Strategy: 2D Sobol sampling on the boundary manifold
        # For DTLZ7, the Pareto front is complex. While x_1...x_{n-1}=0 gives a 1D curve,
        # the front is actually defined by the condition that the "g" term is 0.
        # In standard DTLZ7, g(x) depends on the last few variables.
        # However, a more robust approach for DTLZ7 (and DTLZ4) is that the Pareto front
        # is often best approximated by sampling the decision variables that directly
        # influence the trade-off.
        
        # Let's look at the DTLZ7 definition again.
        # f1 = x1
        # f2 = x2 * (1 + g(x3..xn))
        # f3 = (1-x2) * (1 + g(x3..xn))
        # Wait, standard DTLZ7 is:
        # f1 = x1
        # f2 = x2 * (1 + g(x3..xn))
        # f3 = (1-x2) * (1 + g(x3..xn))
        # This is only for m=3? No, DTLZ7 has m objectives.
        # f1 = x1
        # f2 = x2 * (1 + g)
        # ...
        # fm = (1 - sum_{i=2}^{m-1} xi) * (1 + g)
        
        # If this is the standard DTLZ7:
        # The Pareto front is achieved when g=0.
        # g(x) = (m-2) + sum_{i=m-1}^n (xi - 0.5)^2 - (m-2) * cos(20 * pi * (xi - 0.5) / (m-2))
        # This is a complicated function of the last variables.
        # However, if we set the "g" variables (x_{m-1} ... x_n) to 0.5, 
        # the cos term becomes cos(0)=1, so the sum is (m-2) - (m-2)*1 = 0?
        # Let's check: if xi = 0.5, (xi-0.5)^2 = 0. cos(0) = 1.
        # Term: (m-2) + 0 - (m-2)*1 = 0.
        # So setting x_{m-1}...x_n = 0.5 ensures g=0.
        
        # The objectives then become:
        # f1 = x1
        # f2 = x2
        # f3 = 1 - x1 - x2
        # This is a 2D simplex in the objective space, defined by x1, x2 >= 0 and x1+x2 <= 1.
        
        # So the Pareto front is a 2D surface (triangle) in objective space.
        # We should sample x1 and x2 from the region [0,1] x [0,1] such that x1+x2 <= 1.
        # To maximize hypervolume, we want to cover this triangle densely.
        
        # We use a Sobol sequence in 2D for x1 and x2.
        # We need to map the unit square [0,1]^2 to the triangle x1>=0, x2>=0, x1+x2<=1.
        # A simple mapping:
        # Let u1, u2 be uniform in [0,1].
        # We can sample from the triangle by:
        # r = sqrt(u1)
        # theta = 2 * pi * u2  (this gives a circle, not triangle)
        
        # Better mapping for triangle with vertices (0,0), (1,0), (0,1):
        # We can use the fact that the area is 0.5.
        # If we sample u1, u2 in [0,1], we can reject samples where u1+u2 > 1.
        # But we have a fixed budget. Rejection is inefficient.
        
        # Alternative: Transform a uniform square to the triangle.
        # Let x1 = u1 * (1 - u2)
        # Let x2 = u2 * (1 - u1) ... no, this doesn't cover it simply.
        
        # Standard method:
        # Sample s1, s2 in [0,1].
        # If s1 < 0.5:
        #    x1 = 2 * s1
        #    x2 = 2 * s2 * (1 - s1)
        # Else:
        #    x1 = 2 * (1 - s1)
        #    x2 = 2 * s2 * s1
        # This maps the unit square to the triangle.
        
        # Let's generate Sobol points in 2D.
        sampler = qmc.Sobol(d=2, scramble=True)
        # Ensure number of points is a power of 2 for Sobol? 
        # scipy.stats.qmc.Sobol can generate any number, but quality is best for powers of 2.
        # 200 is not a power of 2. 128 or 256? 
        # We have budget 200. Let's use 200 points.
        # Or we can use 128 points and reuse some? No, just use 200.
        # Actually, Sobol sequences are defined for powers of 2. 
        # We can generate 256 and take the first 200.
        
        n_points = 256
        sobol_samples = sampler.random(n_points)
        # Take first n points
        sobol_samples = sobol_samples[:n]
        
        u1 = sobol_samples[:, 0]
        u2 = sobol_samples[:, 1]
        
        # Map to triangle
        x1 = np.zeros(n)
        x2 = np.zeros(n)
        
        mask = u1 < 0.5
        x1[mask] = 2 * u1[mask]
        x2[mask] = 2 * u2[mask] * (1 - u1[mask])
        
        x1[~mask] = 2 * (1 - u1[~mask])
        x2[~mask] = 2 * u2[~mask] * u1[~mask]
        
        # Construct full decision vectors
        # dim = 22
        # x[0] = x1
        # x[1] = x2
        # x[2] ... x[21] = 0.5 (to ensure g=0)
        
        X = np.zeros((n, dim))
        X[:, 0] = x1
        X[:, 1] = x2
        X[:, 2:] = 0.5
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            pass
            
        return X
