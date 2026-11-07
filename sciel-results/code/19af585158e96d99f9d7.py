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
        limit_time = 325.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        N = self.budget
        if N == 0:
            return np.array([])

        # Strategy based on WFG9 structure:
        # WFG9 has a disconnected Pareto front.
        # The first 2 variables (x1, x2) primarily determine the position on the front (shape/position).
        # The remaining 10 variables contribute to the "noise" or distance from the front.
        # To maximize hypervolume with a limited budget (200), we should:
        # 1. Create a dense grid over the first 2 variables to capture the shape of the front.
        # 2. Fix the remaining 10 variables to values that minimize the distance to the front.
        #    For WFG9, setting the "noise" variables to 0.5 (midpoint of normalized space) often yields points
        #    closer to the ideal front than random or boundary values, as the transformation functions
        #    (like t(x)) often have minima or stable points near the center for these types of benchmark functions.
        #    Alternatively, 0.0 might be better if the penalty is linear, but 0.5 is a safer heuristic for
        #    "neutral" variables in many WFG problems unless specific knowledge suggests otherwise.
        #    Let's try a mix or just 0.5.
        
        # Let's allocate the budget to a grid over the first 2 dimensions.
        # With N=200, a grid of ~14x14 = 196 points is feasible.
        # This ensures dense coverage of the 2D manifold that defines the front.
        
        dim1 = 2
        dim_rest = self.dim - dim1
        
        # Determine grid size
        # We want roughly sqrt(N) points per dimension in the first 2 dims.
        n_grid = int(np.sqrt(N))
        # Adjust to fit exactly N or close to it
        # Let's just generate N points using a grid logic or Sobol on 2 dims?
        # Grid is better for capturing the continuous front shape.
        
        # Generate grid points for first 2 dimensions
        # We will use N points.
        # Let's create a grid of size n_grid x n_grid.
        # If n_grid^2 > N, we reduce n_grid.
        while n_grid * n_grid > N:
            n_grid -= 1
        
        # Generate the grid
        x1_vals = np.linspace(0, 1, n_grid)
        x2_vals = np.linspace(0, 1, n_grid)
        
        # Create meshgrid
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        
        # Flatten
        X_grid = np.column_stack((X1.ravel(), X2.ravel()))
        
        # If we have extra budget, we can add some random points or refine the grid.
        # But for now, let's stick to the grid for the first 2 dims.
        # We need to fill up to N points.
        # If n_grid^2 < N, we have remaining points.
        # Let's just use the grid points and if we have space, maybe add some Sobol points for the rest?
        # Or just repeat? No, repeating is bad.
        # Let's adjust n_grid to be as close to N as possible without exceeding.
        # Actually, let's just use N points from a Sobol sequence on 2 dimensions?
        # Sobol on 2 dims is also good for space filling.
        # But a grid might be better for a 1D curve (front) if the front is 1D?
        # WFG9 front is 1D (defined by x1, x2 relationship? No, it's 3 objectives, so front is 2D surface).
        # Wait, 3 objectives -> Pareto front is 2D.
        # So a 2D grid is appropriate.
        
        # Let's use Sobol for the first 2 dimensions to get N points, which is more efficient than a regular grid
        # if the front is not axis-aligned, but Sobol is generally robust.
        # However, the "inspiration" mentioned "constrained deterministic sweep that varies only the first two decision variables".
        # This suggests a grid or line sweep.
        
        # Let's try a grid for the first 2 dimensions.
        # To get exactly N points, we can't always have a perfect square grid.
        # Let's use N points from a Sobol sequence on 2 dimensions. It's simpler and robust.
        # But the directive says "dense 1D curve". This implies the front might be effectively 1D in the relevant subspace?
        # Or maybe the first variable is the main driver?
        
        # Let's stick to the plan: Grid on first 2 dims.
        # To handle N not being a perfect square, we can just take the first N points from a larger grid generation?
        # Or use Sobol.
        
        # Let's use Sobol for the first 2 dimensions for simplicity and robustness.
        sampler = qmc.Sobol(d=2, scramble=True)
        X1 = sampler.random(N)
        
        # Create full X array
        X = np.zeros((N, self.dim))
        X[:, :2] = X1
        
        # Assign heuristic values to the remaining dimensions
        # Let's try 0.5 for all remaining dimensions.
        X[:, 2:] = 0.5
        
        # Map to bounds
        X = lo + X * span
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
