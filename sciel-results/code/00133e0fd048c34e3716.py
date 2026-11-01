"""Rewritten program for 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def construct():
    """
    Place 50 equal non-overlapping circles in [-0.5, 0.5]^2 to maximize the
    minimum distance between centres and to the boundaries.
    
    We use differential evolution to optimize the centre positions.
    The objective is to maximize the minimum of:
      - half the distance between any two centres
      - distance from each centre to the nearest boundary
    
    We parameterize positions in [-0.5, 0.5] for each coordinate.
    """
    # Flatten positions into a single vector of length 2*N
    # x_coords = params[0:N], y_coords = params[N:2*N]
    
    def negative_min_radius(params):
        """
        Returns the negative of the minimum radius (distance to boundary or
        half-distance to another centre). We minimize this (i.e., maximize radius).
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Distance to boundaries for each circle
        # Square is [-0.5, 0.5]^2
        # Distance to left boundary: x_i - (-0.5) = x_i + 0.5
        # Distance to right boundary: 0.5 - x_i
        # Distance to bottom boundary: y_i + 0.5
        # Distance to top boundary: 0.5 - y_i
        dist_to_left = x + 0.5
        dist_to_right = 0.5 - x
        dist_to_bottom = y + 0.5
        dist_to_top = 0.5 - y
        
        # Minimum distance to boundary for each circle
        boundary_dists = np.minimum(
            np.minimum(dist_to_left, dist_to_right),
            np.minimum(dist_to_bottom, dist_to_top)
        )
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances
        # Compute all pairwise distances
        # For efficiency with N=50, we can use broadcasting
        dx = np.abs(x[:, None] - x[None, :])
        dy = np.abs(y[:, None] - y[None, :])
        dists = np.sqrt(dx**2 + dy**2)
        
        # Exclude diagonal (self-distance)
        np.fill_diagonal(dists, np.inf)
        
        # Minimum pairwise distance
        min_pairwise = np.min(dists)
        
        # The radius is the minimum of min_boundary and min_pairwise/2
        radius = min(min_boundary, min_pairwise / 2.0)
        
        # We want to maximize radius, so minimize -radius
        return -radius
    
    # Bounds for each coordinate: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use differential evolution
    # For N=50, 100 parameters, this might be slow. Let's use reasonable settings.
    result = differential_evolution(
        negative_min_radius,
        bounds=bounds,
        maxiter=200,
        popsize=15,
        tol=1e-8,
        seed=42,
        polish=False,  # Disable polish to save time; we can do a local search if needed
        init='sobol'
    )
    
    # Extract the best positions
    x = result.x[:N]
    y = result[N:2*N]
    
    # Construct the centre array
    centres = np.column_stack((x, y))
    
    return centres

# EVOLVE-BLOCK-END
