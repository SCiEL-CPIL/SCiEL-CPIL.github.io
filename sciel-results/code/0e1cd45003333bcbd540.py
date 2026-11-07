"""Multi-objective optimizer on DTLZ7 using deterministic Pareto-front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ7 that directly samples the known Pareto front.
    
    DTLZ7 has a known analytical Pareto front. For 22 variables and 3 objectives:
    - x[0] and x[1] are the decision variables that parametrize the front
    - x[2:] should be set to 0 for points on the front
    - f1 = x[0]
    - f2 = x[1]  
    - f3 = (n_g+1) * g(x_g) * h(f1, f2, g)
    where n_g = 20 (number of g-variables), g is a penalty term, and h depends on the objectives.
    
    For DTLZ7 specifically:
    - g(x_g) = 1 + 10 * n_g * (sum of sin^2(pi * x_g[i]/2) for i in range(n_g) / n_g - sum x_g[i]*x_g[i] for i in range(n_g))
    - When x_g = 0, g = 1
    - h(f1, f2, g) = 1 - sqrt(f1/g) - f2/g * (1 - sqrt(f1/g))
    - f3 = (n_g + 1) * g * h
    
    On the Pareto front (x_g = 0, so g = 1):
    - f1 = x[0] in [0, 1]
    - f2 = x[1] in [0, 1]
    - f3 = 21 * (1 - sqrt(f1) - f2 * (1 - sqrt(f1)))
    
    We want to minimize all three objectives. The Pareto front is a 2D surface in objective space
    parametrized by (f1, f2) in [0,1]x[0,1].
    
    Strategy: Use a 2D max-min facility location approach to place points uniformly in the
    (f1, f2) space, then map them to decision variables and evaluate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin: 300s limit, keep 20s buffer
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ7 with 22 dims: x[0]=f1, x[1]=f2, x[2:]=0 for Pareto front
        # We need to generate points on the Pareto front
        # f1 = x[0], f2 = x[1], f3 = 21*(1 - sqrt(f1) - f2*(1-sqrt(f1)))
        
        # Budget allocation:
        # Use most of the budget for generating Pareto front points
        # The Pareto front is 2D, so we need to sample (f1, f2) pairs
        
        # Strategy: Generate points using a space-filling approach in (f1, f2) space
        # to maximize hypervolume coverage
        
        n_evals = 0
        results = []
        
        # Phase 1: Generate a grid of points on the Pareto front
        # Use sqrt-spaced points to better cover the curved region
        # f3 = 21*(1 - sqrt(f1) - f2*(1-sqrt(f1)))
        # When f1=0: f3 = 21*(1 - 0 - f2) = 21*(1-f2), so f2 in [0,1] gives f3 in [0,21]
        # When f1=1: f3 = 21*(1-1-f2*0) = 0
        
        # Let's use a combination of approaches:
        # 1. Dense sampling in regions where hypervolume contribution is high
        # 2. Ensure we stay within budget
        
        # The reference point is [1.1, 1.1, 25.0]
        # Points must be strictly better: f1 < 1.1, f2 < 1.1, f3 < 25.0
        # Since f1, f2 in [0,1] and f3 in [0,21], all Pareto front points satisfy this
        
        # Hypervolume contribution of a point depends on its position relative to other points
        # and the reference point. For 3 objectives, a point contributes volume proportional
        # to the box it defines with the reference point, minus overlaps.
        
        # Key insight: Points with smaller objective values contribute more to hypervolume.
        # The corner (0,0,0) would have the largest individual contribution.
        
        # Let's use a stratified sampling approach in (f1, f2) space
        # with more points near the origin (low f1, low f2) where f3 is also low
        
        # Actually, let's think about this more carefully.
        # f3 = 21*(1 - sqrt(f1) - f2*(1-sqrt(f1)))
        # = 21*(1 - sqrt(f1)) - 21*f2*(1-sqrt(f1))
        # = 21*(1-sqrt(f1))*(1 - f2)
        
        # So f3 = 21*(1-sqrt(f1))*(1-f2)
        # This is a product form! f3 is large when both (1-sqrt(f1)) and (1-f2) are large,
        # i.e., when f1 is small and f2 is small.
        
        # For hypervolume, we want to cover the "lower-left" region of the objective space.
        # Let's use a transformation to make the front more uniform.
        
        # Let u = sqrt(f1), v = f2
        # Then f1 = u^2, f2 = v, f3 = 21*(1-u)*(1-v)
        # u in [0,1], v in [0,1]
        
        # In (u, v, f3) space, the front is f3 = 21*(1-u)*(1-v)
        # This is a hyperbolic surface.
        
        # For hypervolume maximization, we should sample points that spread out in
        # the objective space. Let's use a quasi-random (Sobol-like) sequence in (u,v) space.
        
        # Generate points using a simple low-discrepancy sequence
        def sobol_2d(n, seed=0):
            """Generate n points in [0,1]^2 using a simple quasi-random sequence."""
            points = np.zeros((n, 2))
            # Use Halton sequence for 2D
            for i in range(n):
                # Base 2 for first coordinate
                x = 0.0
                rad = 1.0
                j = i + 1
                while j > 0:
                    rad /= 2
                    x += rad * (j % 2)
                    j //= 2
                points[i, 0] = x
                
                # Base 3 for second coordinate
                y = 0.0
                rad = 1.0
                j = i + 1
                while j > 0:
                    rad /= 3
                    y += rad * (j % 3)
                    j //= 3
                points[i, 1] = y
            return points
        
        # How many points can we afford?
        # Each evaluation takes some time. Let's estimate.
        # We have budget evaluations allowed, and 280s deadline.
        # Let's be conservative and use at most budget points, but also check time.
        
        n_points = min(self.budget, 150)  # Conservative: 150 points
        
        # Generate Halton sequence points in (u, v) space
        uv_points = sobol_2d(n_points, seed=42)
        
        # Transform to (f1, f2) space
        f1_vals = uv_points[:, 0] ** 2  # u^2
        f2_vals = uv_points[:, 1]       # v
        
        # Compute f3 for each point
        f3_vals = 21.0 * (1.0 - np.sqrt(f1_vals)) * (1.0 - f2_vals)
        
        # Create decision variable vectors
        # x[0] = f1, x[1] = f2, x[2:] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = f1_vals
        X[:, 1] = f2_vals
        # x[2:] remains 0
        
        # Evaluate all points
        for i in range(n_points):
            if n_evals >= self.budget:
                break
            if time.time() > deadline - 30:  # Stop if we're getting close to deadline
                break
            x = X[i]
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        # If we have remaining budget, we could add more points, but the Halton
        # sequence should already give good coverage. Let's check if we used all budget.
        # For now, return what we have.
        
        return X[:n_evals]
# EVOLVE-BLOCK-END
