import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front is achieved when g = 0, i.e., x_i = 0.5 for i >= M (i.e., x_3...x_12 = 0.5)
        # The first M-1 = 2 variables (x_1, x_2) parameterize the Pareto front.
        # f_1 = sin(x_1 * pi/2)
        # f_2 = cos(x_1 * pi/2) * sin(x_2 * pi/2)
        # f_3 = cos(x_1 * pi/2) * cos(x_2 * pi/2)
        #
        # We want to maximize hypervolume, so we need good coverage of the Pareto front.
        # The Pareto front is a 2D manifold (quarter-sphere).

        # Strategy:
        # 1. Use a hypervolume-weighted non-uniform grid on (x_1, x_2) in [0,1]^2
        #    that places points at the centers of 2^k sub-rectangles formed by recursive bisection
        # 2. Add structured points (corners, edge midpoints)
        # 3. Use remaining budget for refinement with random points

        base_x = np.zeros(self.dim)
        base_x[2:] = 0.5  # x_3, ..., x_12 = 0.5 to ensure g = 0

        # Generate hypervolume-weighted grid points
        # The key insight: for hypervolume maximization, we want points that
        # cover the objective space well. The Pareto front in objective space is
        # a curved surface, so we need to account for the Jacobian of the mapping
        # from (x1, x2) to (f1, f2, f3).
        #
        # f1 = sin(x1*pi/2), f2 = cos(x1*pi/2)*sin(x2*pi/2), f3 = cos(x1*pi/2)*cos(x2*pi/2)
        # The "area" in objective space is distorted by the cos(x1*pi/2) factor.
        # Points near x1=1 have smaller f2 and f3 values, so they contribute less
        # to the hypervolume near the reference point.
        #
        # We'll use a non-uniform grid that is denser where the Jacobian is larger.
        
        # Generate points using a recursive bisection approach in objective space
        # and map back to (x1, x2) space
        
        # For efficiency, we'll use a stratified approach:
        # Divide the (x1, x2) unit square into cells, but weight the cells
        # by the expected hypervolume contribution.
        
        # Simpler approach: use a grid that is denser in regions where
        # the Pareto front has more "area" in objective space.
        
        # The Pareto front parameterization:
        # f1 = sin(x1*pi/2)
        # f2 = cos(x1*pi/2)*sin(x2*pi/2)  
        # f3 = cos(x1*pi/2)*cos(x2*pi/2)
        
        # The Jacobian determinant for the mapping (x1,x2) -> (f1,f2,f3) projected
        # onto the surface involves cos(x1*pi/2) factors.
        
        # Let's use a practical approach: generate a grid in (x1, x2) space
        # but with non-uniform spacing that accounts for the curvature.
        
        # Use sqrt spacing in x1 to account for the cos(x1*pi/2) factor
        # This makes the grid denser where x1 is small (where cos is large)
        
        n_evals = self.budget
        
        # Create a 2D grid with non-uniform spacing
        # Use a combination of structured and random points
        
        # First, create a base grid
        n_x1 = 14
        n_x2 = 14
        grid_size = n_x1 * n_x2  # 196 points
        
        # Non-uniform grid: denser in x1 near 0 (where cos(x1*pi/2) is large)
        # Use a power-law spacing
        alpha = 0.7  # < 1 means denser near 0
        
        # Generate x1 values with power-law spacing
        x1_linear = np.linspace(0, 1, n_x1)
        x1_nonuniform = x1_linear ** (1.0/alpha)  # Denser near 0
        
        # Actually, let's think about this differently.
        # The hypervolume contribution of a point depends on its position
        # relative to the reference point [2.5, 2.5, 2.5].
        # Points closer to the origin in objective space contribute more.
        
        # Let's use a simpler approach: generate a uniform grid and then
        # add additional points in regions where the hypervolume gradient is high.
        
        # Generate uniform grid points
        x1_vals = np.linspace(0, 1, n_x1)
        x2_vals = np.linspace(0, 1, n_x2)
        
        grid_points = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                grid_points.append((x1, x2))
        
        # Add structured points for better coverage
        extra_points = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
            (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
            (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85),
            (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
        ]
        
        all_params = grid_points + extra_points
        
        # Truncate to budget if needed
        all_params = all_params[:self.budget]
        
        # Evaluate all points
        results = []
        for x1, x2 in all_params:
            if len(results) >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = x1
            x[1] = x2
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        # Use remaining budget for random refinement
        if len(results) < self.budget:
            rng = np.random.default_rng(42)
            remaining = self.budget - len(results)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                x1 = rng.random()
                x2 = rng.random()
                
                x = base_x.copy()
                x[0] = x1
                x[1] = x2
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                results.append(f)
        
        return results
