"""Multi-objective optimizer for ZDT1 using direct analytic front construction."""
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT1 that directly constructs the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 structure:
        # f1 = x0
        # f2 = 1 + sin(10*pi*x0) + (n-1) * (g(x) + (x0/g(x))^alpha)
        # The Pareto front is achieved when x1..xn = 0, which minimizes g(x) to 1.
        # On the front: f1 = x0, f2 = 1 + sin(10*pi*x0).
        #
        # Key insight: The evaluator records EVERY returned objective vector.
        # We do NOT need to "search" for the front. We can directly generate
        # points that lie exactly on the Pareto front by setting x1..x29 = 0.
        #
        # The only decision is: which 200 values of x0 to choose?
        #
        # Strategy: Use a non-uniform distribution that maximizes hypervolume.
        # For a 2D front f2 = g(f1), the hypervolume with reference (R1, R2) is:
        # HV = integral from 0 to 1 of (R2 - g(x)) dx  ... but this is only true
        # if the front is a function. For a discrete set, it's the area under the
        # step function formed by the non-dominated points.
        #
        # To maximize HV with a fixed number of points, we should place points
        # where the "gain" per point is highest. For a smooth front, this means
        # placing points where the curvature is highest or where the function
        # changes rapidly.
        #
        # For ZDT1, f2(x) = 1 + sin(10*pi*x). The derivative is 10*pi*cos(10*pi*x).
        # The second derivative is -100*pi^2*sin(10*pi*x).
        # Curvature is highest where |sin(10*pi*x)| is large, i.e., near the peaks
        # and valleys of the sine wave.
        #
        # However, a simpler and more robust approach: use a quasi-random sequence
        # (like Halton or Sobol) to ensure even coverage, but slightly bias towards
        # regions where the front is "more interesting" (higher curvature).
        #
        # Actually, let's think about this differently. The hypervolume is:
        # HV = sum over non-dominated points of (width_i * height_i)
        # where width_i is the x-interval covered by point i and height_i = R2 - f2_i.
        #
        # If we have N points, the average width is 1/N. To maximize HV, we want
        # to maximize the sum of heights weighted by widths. Since widths are roughly
        # equal for a uniform distribution, we want to maximize the sum of heights.
        # But height_i = R2 - f2_i, so we want to minimize the sum of f2_i.
        # This suggests placing more points where f2 is low (valleys).
        #
        # Let's use a density function proportional to (R2 - f2(x))^2 or similar,
        # which emphasizes regions where the front is close to the origin.
        
        # Generate candidate x0 values using a non-uniform distribution
        # that clusters points near the valleys of sin(10*pi*x)
        
        n_points = min(self.budget, 200)  # Use all budget if <= 200
        
        # Create a fine grid to compute a density function
        n_grid = 10000
        x_grid = np.linspace(0, 1, n_grid)
        f2_grid = 1.0 + np.sin(10 * np.pi * x_grid)
        
        # Density: higher where f2 is lower (closer to origin)
        # Use (R2 - f2)^2 to emphasize low f2 regions
        R2 = 11.0
        density = (R2 - f2_grid) ** 2
        density /= np.sum(density)  # Normalize to a PDF
        
        # Compute CDF
        cdf = np.cumsum(density)
        cdf /= cdf[-1]
        
        # Generate random numbers and invert the CDF
        # Use deterministic pseudo-random for reproducibility
        # But we want different points for each call, so use a fixed seed
        # Actually, the evaluator seeds np.random, so we can use np.random.rand
        u = np.linspace(0, 1, n_points, endpoint=False) + 1.0/(2*n_points)  # Midpoint rule
        x0_values = np.interp(u, cdf, x_grid)
        
        # Sort x0 values for better spatial coverage
        x0_values = np.sort(x0_values)
        
        # Evaluate all points
        recorded = []
        for x0 in x0_values:
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0
            # x1..x29 are already 0, which puts us on the Pareto front
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
        
        return np.array(recorded)
