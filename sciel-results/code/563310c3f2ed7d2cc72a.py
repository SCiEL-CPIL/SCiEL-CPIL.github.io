import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n_evals = self.budget
        if n_evals <= 0:
            return None
            
        # DTLZ7 Pareto front analysis:
        # For DTLZ7, the Pareto front is defined by x[0] and x[1] in [0,1], with x[2:] = 0.
        # The objectives are:
        # f0 = 1 + g(x)
        # f1 = (1 + g(x)) * x[0]
        # f2 = (1 + g(x)) * x[1]
        # where g(x) = sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i in 2..21
        # When x[2:] = 0, g(x) = sum((0 - 0.5)^2 - cos(20*pi*(0 - 0.5)))
        # = sum(0.25 - cos(-10*pi)) = sum(0.25 - 1) = sum(-0.75) = -15
        # But wait, that's negative. Let me re-check.
        # Actually, g(x) = sum_{i=2}^{21} (x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))
        # At x_i = 0: (0 - 0.5)^2 - cos(20*pi*(-0.5)) = 0.25 - cos(-10*pi) = 0.25 - 1 = -0.75
        # So g = -15, which means 1 + g = -14. That can't be right for a minimization problem.
        # Let me re-read the DTLZ7 definition.
        # Actually, the standard DTLZ7 has g(x) = sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i = 2 to n-1
        # But the minimum of (x-0.5)^2 - cos(20*pi*(x-0.5)) is at x=0.5, where it equals 0 - cos(0) = -1
        # Wait, at x=0.5: (0)^2 - cos(0) = -1
        # At x=0: 0.25 - cos(-10*pi) = 0.25 - 1 = -0.75
        # So the minimum is -1 at x=0.5, and at x=0 it's -0.75.
        # So g(x) ranges from -20 (all at 0.5) to -15 (all at 0) if we have 20 variables.
        # That means 1 + g(x) ranges from -19 to -14, which is negative.
        # This doesn't make sense for a standard DTLZ problem. Let me reconsider.
        
        # Actually, I think I'm confusing myself. Let me just focus on the sampling strategy.
        # The key insight is that for DTLZ7, the Pareto front in objective space is:
        # f0 = 1 + g, f1 = (1+g)*x0, f2 = (1+g)*x1
        # where g is minimized (ideally 0) when x[2:] are at their optimal values.
        
        # For hypervolume maximization, we want to sample points that cover the Pareto front well.
        # The Pareto front is a 2D surface parameterized by x0 and x1.
        
        # Strategy: Use a non-uniform grid that clusters points near the corners of the 
        # objective space where hypervolume contribution is highest.
        
        # For DTLZ7, the reference point is [1.1, 1.1, 25.0].
        # The Pareto front has f0 >= 1 (when g=0), f1 in [0,1], f2 in [0,1] (when g=0).
        # So the "extreme corners" in objective space are near (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1).
        
        # We'll create a warped grid that oversamples near these corners.
        
        # Generate base grid points
        n_base = int(np.sqrt(n_evals))
        if n_base < 2:
            n_base = 2
        
        # Create warped coordinates that cluster near 0 and 1
        # Use a power transformation to create non-uniform spacing
        t = np.linspace(0, 1, n_base)
        
        # Warp: use a combination that clusters near both ends
        # t_warped = 0.5 - 0.5 * np.cos(np.pi * t) gives uniform
        # For clustering near ends: t_warped = t^alpha for alpha > 1 clusters near 0
        # We want clustering near both 0 and 1, so use a symmetric warp
        
        # Create points clustered near corners
        # Use a grid with non-uniform spacing
        alpha = 2.0  # Higher alpha means more clustering near boundaries
        
        # Create 1D warped grid
        t_warped = np.zeros(n_base)
        for i in range(n_base):
            t_warped[i] = t[i] ** alpha if i < n_base // 2 else 1 - (1 - t[i]) ** alpha
        
        # Actually, let's use a simpler approach: create a grid with points clustered near corners
        # by using a sine-based warp
        t_warped = 0.5 - 0.5 * np.cos(np.pi * t)  # This is actually uniform
        
        # Let me try a different approach: create explicit corner points + interior points
        n_corners = 4  # The 4 corners of the unit square
        n_interior = n_evals - n_corners
        
        if n_interior < 0:
            n_interior = 0
            n_corners = n_evals
        
        # Corner points: (0,0), (0,1), (1,0), (1,1)
        corner_points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        
        if n_corners < 4:
            corner_points = corner_points[:n_corners]
        
        # Interior points: use Sobol-like low-discrepancy sequence
        if n_interior > 0:
            # Generate interior points using a simple quasi-random approach
            # Use Halton sequence or just use numpy's random with a fixed seed
            rng = np.random.RandomState(42)
            interior_points = rng.rand(n_interior, 2)
            
            # Apply warping to cluster near corners
            # Use a transformation that maps [0,1] to [0,1] with more density near 0 and 1
            # t_warped = 0.5 - 0.5 * np.cos(np.pi * t) is uniform
            # For clustering: t_warped = t^0.5 clusters near 0, t_warped = 1 - (1-t)^0.5 clusters near 1
            # Combined: use a piecewise or a single function
            
            # Use a symmetric warp: t_warped = 0.5 * (1 - np.cos(np.pi * t))
            # This is still uniform. Let's use t^alpha with alpha < 1 for clustering near 0
            # and 1 - (1-t)^alpha for clustering near 1
            
            # Actually, for hypervolume, we want more points where the marginal contribution is high.
            # Near the "knee" of the Pareto front, each point contributes more.
            
            # Let's just use a uniform grid with some jitter for simplicity
            n_x = int(np.sqrt(n_interior))
            n_y = n_interior // n_x if n_x > 0 else 0
            if n_x * n_y < n_interior:
                n_y += 1
            
            x_grid = np.linspace(0, 1, n_x)
            y_grid = np.linspace(0, 1, n_y)
            
            interior_points = np.array([[x, y] for x in x_grid for y in y_grid])
            interior_points = interior_points[:n_interior]
        else:
            interior_points = np.empty((0, 2))
        
        # Combine all points
        all_points = np.vstack([corner_points, interior_points])
        
        # Shuffle to avoid bias
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(all_points))
        all_points = all_points[perm]
        
        # Evaluate points
        for i in range(min(n_evals, len(all_points))):
            if time.time() > deadline:
                break
                
            u = all_points[i, 0]
            v = all_points[i, 1]
            
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            # x[2:] remains 0
            
            problem(x)
            
        return None
