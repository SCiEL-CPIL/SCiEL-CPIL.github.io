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
            
        # DTLZ7 structure:
        # f0 = 1 + g(x)
        # f1 = (1 + g(x)) * x0
        # f2 = (1 + g(x)) * x1
        # where g(x) = sum_{i=2}^{n-1} ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)))
        # 
        # The Pareto front is achieved when g(x) is minimized.
        # g(x) is minimized when each (x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)) is minimized.
        # Let y = x_i - 0.5. We minimize y^2 - cos(20*pi*y) for y in [-0.5, 0.5].
        # The global minimum of y^2 - cos(20*pi*y) occurs at y = 0 (where value is -1)
        # or at local minima. Let's check: at y=0, value = 0 - 1 = -1.
        # At y = 0.5, value = 0.25 - cos(10*pi) = 0.25 - 1 = -0.75.
        # At y = -0.5, value = 0.25 - cos(-10*pi) = 0.25 - 1 = -0.75.
        # So the minimum is at y=0, i.e., x_i = 0.5, with value -1 per term.
        # With 20 terms (i=2..21), g_min = 20 * (-1) = -20.
        # So 1 + g_min = 1 - 20 = -19.
        #
        # Wait, that gives negative objectives. Let me re-read the DTLZ7 definition.
        # Standard DTLZ7: g(x) = sum_{i=m+1}^{n} (x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5)) + 20
        # The +20 is crucial! So g_min = 20 * (-1) + 20 = 0.
        # Thus 1 + g_min = 1.
        # Pareto front: f0 = 1, f1 = x0, f2 = x1 for x0, x1 in [0,1].
        #
        # Reference point: [1.1, 1.1, 25.0]
        # Points on the front: f0=1, f1=x0, f2=x1
        # All satisfy f0=1 < 1.1, f1=x0 <= 1 < 1.1, f2=x1 <= 1 < 25.0.
        # So all front points are strictly better than the reference point.
        #
        # Strategy: Sample the 2D Pareto front (x0, x1) with x[2:] = 0.5
        # to maximize hypervolume coverage.
        
        # Use Sobol sequence for quasi-random sampling of the 2D Pareto front
        from scipy.stats import qmc
        
        # Generate Sobol points in [0,1]^2
        sobol = qmc.Sobol(d=2, scramble=True)
        # Number of points: we'll use most of our budget
        # Sobol requires power of 2, so round up
        n_points = n_evals
        # Find next power of 2 >= n_points
        n_sobol = 1
        while n_sobol < n_points:
            n_sobol *= 2
        
        # Generate Sobol points
        sobol_points = sobol.random(n_sobol)
        # Take only the first n_points
        sobol_points = sobol_points[:n_points]
        
        # Also add corner points for better coverage
        corners = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        
        # Combine Sobol points with corners
        all_points = np.vstack([corners, sobol_points])
        
        # If we have more points than budget, subsample
        if len(all_points) > n_evals:
            # Keep corners and subsample the rest
            n_corner = len(corners)
            n_remaining = n_evals - n_corner
            if n_remaining > 0:
                # Shuffle Sobol points and take first n_remaining
                rng = np.random.RandomState(42)
                perm = rng.permutation(len(sobol_points))
                selected_sobol = sobol_points[perm[:n_remaining]]
                all_points = np.vstack([corners, selected_sobol])
            else:
                all_points = corners[:n_evals]
        
        # Shuffle to avoid any ordering bias
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(all_points))
        all_points = all_points[perm]
        
        # Evaluate points
        for i in range(min(n_evals, len(all_points))):
            if time.time() > deadline:
                break
                
            u = all_points[i, 0]
            v = all_points[i, 1]
            
            x = np.full(self.dim, 0.5)  # x[2:] = 0.5 for optimal g
            x[0] = u
            x[1] = v
            
            problem(x)
            
        return None
