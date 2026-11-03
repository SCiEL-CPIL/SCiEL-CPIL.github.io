"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in radians for a set of unit points."""
    # For each point on the sphere, the distance to the nearest point is what matters.
    # The covering radius is the maximum over the sphere, which occurs at the circumcenter
    # of some Voronoi cell on the sphere (or at a vertex if the cell is large).
    # A practical lower bound / estimate: for each point, compute the distance to its nearest neighbor,
    # but the true covering radius is larger.
    # Better approach: The covering radius is the maximum circumradius of the spherical Voronoi cells.
    # For a practical estimate, we can use the fact that for each point, the farthest point in its
    # Voronoi cell is at most half the distance to the nearest neighbor in some sense, but this is an underestimate.
    # 
    # Actually, the covering radius is the smallest r such that caps of angular radius r around each point cover the sphere.
    # This is equivalent to: max over all directions y of min over i of angle(y, p_i).
    # 
    # For optimization purposes, we can use a smooth approximation or a direct computation.
    # A common approach: for each point p_i, the boundary of its Voronoi cell is determined by equidistant points to p_i and its neighbors.
    # The farthest point in the Voronoi cell from p_i is the vertex of the spherical polygon formed by the intersection of the bisecting great circles.
    # 
    # For a practical optimization, let's compute the covering radius by checking the circumradii of the facets of the convex hull.
    # The covering radius is the maximum circumradius of the facets (triangles) of the convex hull of the points, where the circumradius
    # is the angular distance from the facet center to the vertices, but only if the cap is empty.
    #
    # Simpler approach for optimization: use the maximum of the angular distances from each point to its farthest neighbor in its Voronoi cell.
    # For a quick estimate, we can use the fact that the covering radius is at least half the minimum pairwise angular distance,
    # and at most the maximum of (distance to nearest neighbor) / 2 * some factor.
    #
    # Let's use a direct computation: sample many points on the sphere and find the max min-distance.
    # This is expensive but accurate enough for optimization if we use a moderate number of samples.
    
    # For efficiency during optimization, use a coarser grid
    # Generate a grid of test points on the sphere
    # Use a Fibonacci sphere for test points
    n_test = 2000
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_points = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # Normalize test points
    norms = np.linalg.norm(test_points, axis=1, keepdims=True)
    test_points = test_points / norms
    
    # Compute min distance from each test point to any configuration point
    # points: (N, 3), test_points: (n_test, 3)
    # Use matrix multiplication for dot products
    dots = test_points @ points.T  # (n_test, N)
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (n_test, N)
    min_angles = np.min(angles, axis=1)  # (n_test,)
    
    return np.max(min_angles)


def _objective(params):
    """Objective function for optimization: covering radius."""
    # params is a flat array of 3N values
    points = params.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        return 100.0  # Penalty for zero vectors
    points = points / norms
    
    # Check if all points are in a closed hemisphere
    # If there exists a vector v such that v . p_i >= 0 for all i, then they're in a hemisphere
    # This is equivalent to the origin being in the convex hull of the points
    # Quick check: if the sum of points is close to zero, they might be spread out
    # More robust: check if the convex hull contains the origin
    # For simplicity, add a penalty if the points seem to be in a hemisphere
    # We'll rely on the covering radius being large in that case
    
    return _covering_radius(points)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # Start with a Fibonacci spiral configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0_flat = x0.flatten()
    
    # Use Nelder-Mead optimization
    # This is a derivative-free method suitable for non-smooth objectives
    result = minimize(_objective, x0_flat, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    best_points = result.x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        # Fallback to initial
        best_points = x0
    else:
        best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
