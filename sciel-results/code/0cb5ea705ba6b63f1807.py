"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Objective function: minimize the negative of the minimum clearance
    # clearance = min(min_pairwise_dist, min_boundary_dist * 2)
    # We want to maximize clearance, so we minimize -clearance.
    
    def objective(params):
        centers = params.reshape(N, 2)
        
        # Calculate pairwise distances
        # Using broadcasting for efficiency, though for N=50, loops are fine too.
        # dist[i, j] = distance between i and j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask out self-distances
        np.fill_diagonal(dists, np.inf)
        
        min_pairwise = np.min(dists)
        
        # Calculate boundary distances
        # Square is [-0.5, 0.5] x [-0.5, 0.5]
        # Distance to boundary for point (x,y) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
        x, y = centers[:, 0], centers[:, 1]
        boundary_dists = np.min(np.stack([0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y], axis=1), axis=1)
        
        # Effective distance for radius calculation
        # Radius r = min(min_pairwise / 2, min_boundary_dist)
        # To maximize r, we maximize min(min_pairwise, 2 * min_boundary_dist)
        # Let D = min(min_pairwise, 2 * min_boundary_dist)
        # We minimize -D
        
        min_boundary_effective = 2 * np.min(boundary_dists)
        
        D = min(min_pairwise, min_boundary_effective)
        
        return -D

    # Initialization: Hexagonal lattice
    # Estimate side length s for hex packing
    # Area per hex point ~ sqrt(3)/2 * s^2
    # 50 * sqrt(3)/2 * s^2 = 1  => s = sqrt(2 / (50*sqrt(3)))
    s = np.sqrt(2 / (N * np.sqrt(3)))
    
    # Generate hex grid points
    # We need enough points to choose from
    # Let's generate a grid that covers the square
    # x spacing = s
    # y spacing = s * sqrt(3) / 2
    # Offset every other row by s/2
    
    # Determine number of points needed in grid
    # Square side 1.
    # nx = int(1/s) + 2
    # ny = int(1 / (s * np.sqrt(3) / 2)) + 2
    
    nx = int(1 / s) + 3
    ny = int(1 / (s * np.sqrt(3) / 2)) + 3
    
    points = []
    for j in range(ny):
        y = (j + 0.5) * s * np.sqrt(3) / 2 - 0.5 # Centered
        offset = (j % 2) * s / 2
        for i in range(nx):
            x = (i + 0.5) * s - 0.5 + offset # Centered
            points.append([x, y])
            
    points = np.array(points)
    
    # Select N points closest to the center (0,0)
    dist_to_center = np.sum(points**2, axis=1)
    closest_indices = np.argsort(dist_to_center)[:N]
    initial_centers = points[closest_indices]
    
    # Jitter to break symmetry
    np.random.seed(42) # Reproducibility
    initial_centers += np.random.normal(0, s * 0.1, initial_centers.shape)
    
    # Flatten for optimizer
    x0 = initial_centers.flatten()
    
    # Optimize
    # Use Powell method as it is generally better for high-dimensional derivative-free
    # optimization than Nelder-Mead for this specific problem size.
    # If Powell is too slow, Nelder-Mead is an option, but Powell is usually faster.
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 2000, 'ftol': 1e-8, 'xtol': 1e-8})
        if res.success or res.fun < -0.07: # Fallback check
            final_centers = res.x.reshape(N, 2)
        else:
            final_centers = initial_centers
    except Exception:
        final_centers = initial_centers
        
    # Ensure points are strictly inside the square to avoid invalid states
    # Clip to slightly inside [-0.5, 0.5]
    # The optimizer might push points slightly out.
    # We should project them back or ensure the objective penalizes it heavily.
    # The objective function handles boundary distance, so if a point is outside,
    # boundary_dist becomes negative, making D negative, and -D positive.
    # The optimizer minimizes -D, so it will try to make D large (positive).
    # However, if a point is far outside, the distance to boundary is large negative?
    # No, 0.5 - x if x > 0.5 is negative.
    # min(0.5-x, ...) will be negative.
    # Then 2 * min_boundary_dist is negative.
    # D = min(min_pairwise, negative) = negative.
    # -D = positive.
    # The optimizer minimizes -D. It wants -D to be small (negative).
    # So it wants D to be large (positive).
    # So it will push points inside.
    
    # Just in case, clip to valid range
    final_centers = np.clip(final_centers, -0.499, 0.499)
    
    return final_centers

# EVOLVE-BLOCK-END
