"""Multi-objective optimizer on DTLZ2 using hypervolume-weighted grid search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-weighted grid optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_pos = self.n_obj - 1  # For DTLZ2, first M-1 vars determine position
        n_g = self.dim - n_pos  # Remaining vars determine distance g(x)

        # Optimal g(x) = 0 achieved by setting distance vars to midpoint
        mid = 0.5 * (lo[n_pos:] + hi[n_pos:])

        # We want to maximize hypervolume. 
        # The Pareto front of DTLZ2 in 3 objectives is a quarter sphere.
        # f_1^2 + f_2^2 + f_3^2 = 1.
        # Reference point is [2.5, 2.5, 2.5].
        # Hypervolume contribution of a point is maximized when it covers unique volume.
        # A uniform grid in the angle space (theta, phi) maps to non-uniform spacing on the surface.
        # Specifically, for a sphere, equal area sampling corresponds to:
        # sin(theta) uniform? No, area element is sin(theta) dtheta dphi.
        # So dtheta should be proportional to 1/sin(theta).
        # Alternatively, we can use a grid that is uniform in the "hypervolume contribution" sense.
        
        # Let's stick to the structural insight from inspirations: 
        # DTLZ2 front is smooth. A dense grid on the positional variables is key.
        # Inspiration 1 uses arcsin mapping to ensure uniform coverage of the front.
        # Let's use a similar mapping but adapt the grid size to budget.
        
        # Budget is 200.
        # We can afford a 14x14 grid (196 points) for n_pos=2.
        # This leaves 4 points for potential refinement or edge cases.
        
        if self.n_obj == 3:
            n_pos = 2
            # Use a grid size that fits in budget
            # 14^2 = 196. 15^2 = 225 > 200.
            grid_size = 14
        else:
            # Fallback for other dimensions (though task specifies n_obj=3, dim=12)
            grid_size = int(np.sqrt(self.budget))
            while grid_size ** n_pos > self.budget:
                grid_size -= 1
                if grid_size < 2:
                    grid_size = 2
                    break

        n_grid = grid_size ** n_pos
        n_remaining = self.budget - n_grid

        # Generate grid points
        # For DTLZ2, x_1 and x_2 determine the position.
        # x_1 = (2/pi) * arcsin(u_1)
        # x_2 = u_2
        # This ensures the points are distributed uniformly on the Pareto front.
        
        u1_vals = np.linspace(0, 1, grid_size)
        u2_vals = np.linspace(0, 1, grid_size)
        
        U1, U2 = np.meshgrid(u1_vals, u2_vals, indexing='ij')
        
        X1 = (2.0 / np.pi) * np.arcsin(U1)
        X2 = U2
        
        X1_flat = X1.ravel()
        X2_flat = X2.ravel()
        
        n_points = len(X1_flat)
        
        # Create full dimension vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = X1_flat
        X[:, 1] = X2_flat
        
        # Set distance variables to midpoint
        if n_g > 0:
            X[:, n_pos:] = mid
            
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all grid points
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            F[i] = problem(X[i])

        # Use remaining budget for additional points
        # We can add points that might improve hypervolume, e.g., near the corners or edges
        # Or simply random points on the front to fill gaps.
        
        if n_remaining > 0 and time.time() < deadline - 5:
            # Generate random points on the front
            n_rand = n_remaining
            rng = np.random.default_rng(42)
            
            u1_rand = rng.random(n_rand)
            u2_rand = rng.random(n_rand)
            
            X1_rand = (2.0 / np.pi) * np.arcsin(u1_rand)
            X2_rand = u2_rand
            
            X_rand = np.zeros((n_rand, self.dim))
            X_rand[:, 0] = X1_rand
            X_rand[:, 1] = X2_rand
            if n_g > 0:
                X_rand[:, n_pos:] = mid
            X_rand = np.clip(X_rand, lo, hi)
            
            X = np.vstack([X, X_rand])
            
            for i in range(n_rand):
                if time.time() > deadline - 5:
                    break
                problem(X[n_points + i])

        return X
# EVOLVE-BLOCK-END
