"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Strategy:
    1. Analytical reduction: For ZDT4, the Pareto front is achieved by setting 
       x_1...x_9 = 0 (minimizing g to 1). The front is f1 = x0, f2 = 2 - sqrt(x0) 
       for x0 in [0, 1].
    2. The problem reduces to choosing K points x0_i in [0, 1] to maximize the 
       hypervolume of the set of points (x0_i, 2 - sqrt(x0_i)) against reference 
       point (11, 400).
    3. Since the front is convex and monotonic, we can optimize the placement of 
       these K points. The hypervolume contribution is the sum of the areas of the 
       rectangles formed by each point and the reference point, minus overlaps. 
       However, a more direct approach for a convex front is to minimize the 
       discretization error of the integral approximation or use a specific 
       spacing that equalizes the marginal contribution.
      
      A known result for maximizing hypervolume on a convex front is that the 
      optimal points are clustered where the curvature is highest (near x=0). 
      We use a power-law spacing: x_i = (i/K)^p * range, with p > 1 to cluster 
      near 0. We can also fine-tune p or use a direct optimization of the grid.
      
      However, an even stronger approach is to directly compute the optimal grid 
      for the given budget. For K=200, we can solve for the optimal x_i by 
      minimizing the "gap" or using a fixed-point iteration.
      
      Let's try a simple but effective heuristic: 
      The marginal gain of a point at x is proportional to the area it "covers". 
      For a convex function, the optimal grid is non-uniform. 
      
      We will use a deterministic grid with power-law spacing. 
      x_i = L * (i/K)^alpha, for i=0..K-1.
      We can tune alpha. Empirically, alpha around 1.5 to 2.0 works well.
      
      Alternatively, we can use a "refinement" step: start with a coarse grid and 
      iteratively add points where the local hypervolume gain is highest. 
      But with K=200, a direct grid is fast enough.
      
      Let's implement a power-law grid with alpha=2.0 (quadratic spacing).
      This places points closer to x=0, where the curve is steeper.
      
      Wait, the current program uses "equal area" which is different.
      The inspiration says "power-law spacing ... outperforming uniform or adaptive bisection".
      So let's switch to power-law spacing.
      
      x_i = (i/K)^2 * 1.0 for i=0..199.
      This gives x_0=0, x_1=(1/200)^2=0.000025, ..., x_199=(199/200)^2 ~ 0.99.
      
      We should ensure we don't miss the endpoint x=1. 
      Actually, for hypervolume, the last point should be close to 1 to capture the full width.
      A standard technique is to use i=0..K-1 for x = (i/(K-1))^alpha, but this forces x_last=1.
      Or use i=1..K for x = (i/K)^alpha, which doesn't reach 1.
      Reaching 1 is important because f1=1 is the max x0.
      If we don't reach 1, we lose the rightmost part of the front.
      So we should probably use a grid that includes 0 and 1, or gets very close.
      
      Let's use x_i = (i / (K-1))^alpha for i=0..K-1.
      This ensures x_0=0 and x_{K-1}=1.
      With alpha=2.0, the points are:
      i=0: 0
      i=1: (1/199)^2 ~ 0.000025
      ...
      i=199: 1.0
      
      This seems like a strong structural change from the "equal area" bisection.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Check if 0 is within bounds for x1...x9
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if not zeros_ok:
            return self._fallback(problem, bounds, deadline)

        # 1. Determine x0 domain
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)

        if x0_lo >= x0_hi:
            x0_vals = np.array([lo[0]])
            X = np.zeros((1, self.dim))
            X[0, 0] = lo[0]
            F = [problem(X[0])]
            return X

        K = self.budget

        # 2. Generate points using power-law spacing
        # x_i = x0_lo + (x0_hi - x0_lo) * (i / (K-1))^alpha
        # alpha > 1 clusters points near x0_lo (which is 0 in standard ZDT4)
        # where the curvature of f2 = 2 - sqrt(x0) is highest.
        alpha = 2.0  # Quadratic spacing
        
        if K == 1:
            x0_vals = np.array([x0_lo])
        else:
            # Ensure we hit the endpoints
            i_vals = np.arange(K)
            # Scale to [0, 1]
            normalized = i_vals / (K - 1)
            # Apply power law
            x0_vals = x0_lo + (x0_hi - x0_lo) * (normalized ** alpha)
            
            # Clip to be safe
            x0_vals = np.clip(x0_vals, x0_lo, x0_hi)

        # 3. Construct X
        X = np.zeros((K, self.dim))
        X[:, 0] = x0_vals
        # X[:, 1:] = 0 is already set

        # 4. Evaluate
        F = []
        for i in range(K):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))

        return X

    def _fallback(self, problem, bounds, deadline):
        # Simple uniform sampling
        n_points = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        x0_lo = lo[0]
        x0_hi = hi[0]
        if x0_lo >= x0_hi:
            x0_vals = np.array([lo[0]])
        else:
            x0_vals = np.linspace(x0_lo, x0_hi, n_points)

        X = np.zeros((len(x0_vals), self.dim))
        X[:, 0] = x0_vals

        F = []
        for i in range(len(x0_vals)):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))

        return X

# EVOLVE-BLOCK-END
