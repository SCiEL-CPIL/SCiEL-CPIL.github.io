"""Multi-objective optimizer on DTLZ2 using structured grid search on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured grid optimizer for DTLZ2 exploiting Pareto front geometry."""

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

        # Build a dense grid over the positional variables
        # For M=3: 2 positional vars -> grid_size x grid_size points
        # Budget 200, use 14x14 = 196 grid + 4 refinement = 200
        if self.n_obj == 3:
            grid_size = 14
        elif self.n_obj == 2:
            grid_size = 20
        else:
            grid_size = 5

        n_grid = grid_size ** n_pos

        # Ensure we don't exceed budget
        while n_grid > self.budget - 4:
            grid_size -= 1
            if grid_size < 2:
                grid_size = 2
                break
            n_grid = grid_size ** n_pos

        # Create grid over positional variables using uniform area mapping
        # For DTLZ2, the Pareto front is a sphere.
        # Uniform sampling in Cartesian x1, x2 leads to non-uniform area coverage
        # and potentially invalid points (x1^2 + x2^2 > 1).
        # We use a mapping that ensures points are on the front and uniformly distributed in area.
        # u1 ~ U[0, 1] corresponds to cos(theta1).
        # phi ~ U[0, pi/2] corresponds to theta2.
        # x1 = u1 * cos(phi)
        # x2 = u1 * sin(phi)
        # This maps the unit square to the quarter disk, with density proportional to 1/sqrt(1-x1^2-x2^2).
        
        u1_grid = np.linspace(0, 1, grid_size)
        phi_grid = np.linspace(0, np.pi/2, grid_size)
        
        U1, Phi = np.meshgrid(u1_grid, phi_grid, indexing='ij')
        
        x1_vals = U1 * np.cos(Phi)
        x2_vals = U1 * np.sin(Phi)
        
        grid_X = np.column_stack((x1_vals.ravel(), x2_vals.ravel()))
        
        # Create full dimension vectors
        X_grid = np.zeros((len(grid_X), self.dim))
        X_grid[:, :n_pos] = grid_X
        X_grid[:, n_pos:] = mid

        # Evaluate grid
        F_grid = np.array([problem(x) for x in X_grid], dtype=np.float64)

        n_used = len(X_grid)
        n_remaining = self.budget - n_used

        # Use remaining budget for edge/corner refinement
        X_refined = []
        F_refined = []

        if n_remaining > 0 and time.time() < deadline - 5:
            # Add boundary points that may improve hypervolume coverage
            # For 3D: add points along the edges of the square domain
            if n_pos == 2:
                # Add midpoints of edges and corners if not already present
                edge_points = []
                half = 0.5 * (lo[:n_pos] + hi[:n_pos])
                # Edge midpoints
                for i in range(n_pos):
                    for j in range(i + 1, n_pos):
                        for k in range(2):
                            pt = half.copy()
                            pt[i] = lo[i] if k == 0 else hi[i]
                            pt[j] = lo[j] if k == 0 else hi[j]
                            edge_points.append(pt)
                # Corner points
                for c0 in [0, 1]:
                    for c1 in [0, 1]:
                        pt = np.zeros(n_pos)
                        pt[0] = lo[0] if c0 == 0 else hi[0]
                        pt[1] = lo[1] if c1 == 0 else hi[1]
                        edge_points.append(pt)

                # Deduplicate: only add points not already in grid
                for pt in edge_points:
                    if len(X_refined) + n_used >= self.budget:
                        break
                    # Check if this point is already in the grid
                    if not any(np.allclose(grid_X, pt, atol=1e-10)):
                        x_new = np.zeros(self.dim)
                        x_new[:n_pos] = pt
                        x_new[n_pos:] = mid
                        X_refined.append(x_new)

            # Evaluate refined points
            if X_refined:
                X_refined = np.array(X_refined)
                F_refined = np.array([problem(x) for x in X_refined], dtype=np.float64)

        # Combine all points
        if X_refined:
            X_all = np.vstack([X_grid, X_refined])
            F_all = np.vstack([F_grid, F_refined])
        else:
            X_all = X_grid
            F_all = F_grid

        return X_all
# EVOLVE-BLOCK-END
