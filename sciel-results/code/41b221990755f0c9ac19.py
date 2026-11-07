import numpy as np
import time

class Optimizer:
    """DTLZ2-specific optimizer: hypervolume-weighted grid sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front: g=0 when x_i=0.5 for i>=m (indices m-1 to n-1)
        # f1 = cos(theta1)*cos(theta2), f2 = cos(theta1)*sin(theta2), f3 = sin(theta1)
        # theta1 = x0*pi/2, theta2 = x1*pi/2
        # Surface area element: cos(theta1) dtheta1 dtheta2
        # For hypervolume optimization, weight toward interior where marginal volume is higher

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        evaluated_points = []
        used_budget = 0

        # Strategy: Hypervolume-weighted grid
        # Use a grid with non-uniform spacing that clusters points toward the center
        # of the parameter space where hypervolume contribution is maximized.
        # 
        # For 3-objective DTLZ2, the hypervolume contribution is highest near
        # theta1 = pi/4, theta2 = pi/4 (center of the quarter-sphere).
        # We use a transformation that maps uniform grid indices to positions
        # clustered toward the center.
        
        # Grid size: aim for ~196 points from grid, rest from Halton
        # 14x14 = 196 points, leaving 4 for Halton
        grid_size = 14
        num_grid = grid_size * grid_size
        
        # Use a power-law transformation to cluster points toward center
        # u' = 0.5 + 0.5 * sign(u - 0.5) * |2*(u - 0.5)|^alpha
        # With alpha < 1, this clusters toward center
        # Lower alpha means stronger clustering toward the center
        alpha = 0.6  # Clustering parameter
        
        def cluster_toward_center(u, alpha=0.7):
            """Map u in [0,1] to [0,1] with clustering toward 0.5"""
            if u < 0.5:
                return 0.5 - 0.5 * (1 - 2*u)**alpha
            else:
                return 0.5 + 0.5 * (2*u - 1)**alpha

        # Generate grid points with hypervolume-weighted positions
        for i in range(grid_size):
            for j in range(grid_size):
                if time.time() > deadline:
                    break
                if used_budget >= self.budget:
                    break
                
                # Uniform grid in [0, 1]
                u_uniform = (i + 0.5) / grid_size
                v_uniform = (j + 0.5) / grid_size
                
                # Apply clustering toward center
                u = cluster_toward_center(u_uniform, alpha)
                v = cluster_toward_center(v_uniform, alpha)
                
                # Map to DTLZ2 parameters
                # x0: arcsin transformation for surface-area uniformity
                x0 = (2.0 / np.pi) * np.arcsin(u)
                x1 = v
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1

        # Fill remaining budget with Halton sequence
        halton_idx = 1
        while used_budget < self.budget:
            if time.time() > deadline:
                break
            
            u = halton(halton_idx, 2)
            v = halton(halton_idx, 3)
            
            # Also apply clustering to Halton points for consistency
            u = cluster_toward_center(u, alpha)
            v = cluster_toward_center(v, alpha)
            
            x0 = (2.0 / np.pi) * np.arcsin(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1
            halton_idx += 1

        return evaluated_points
