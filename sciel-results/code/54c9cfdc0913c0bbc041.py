"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _fibonacci_sphere(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _lloyd_iteration(points, test_points, max_iter=100):
    """
    Lloyd's algorithm on the sphere.
    Moves each point to the centroid of its Voronoi cell defined by test_points.
    """
    points = _normalize(points).copy()
    
    for _ in range(max_iter):
        # Compute distance from each test point to each configuration point
        # dots[i, j] = dot(test_points[i], points[j])
        dots = test_points @ points.T
        # The nearest point index for each test point
        nearest_idx = np.argmax(dots, axis=1)
        
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        
        for i in range(len(test_points)):
            idx = nearest_idx[i]
            new_points[idx] += test_points[i]
            counts[idx] += 1
        
        # Update points to centroids
        for j in range(N):
            if counts[j] > 0:
                new_points[j] /= counts[j]
                new_points[j] /= np.linalg.norm(new_points[j])
            else:
                # If a point has no Voronoi cell (unlikely with good init), keep it
                new_points[j] = points[j]
                
        points = new_points
        
    return points


def _covering_radius(points, test_points):
    """
    Calculate the covering radius (in radians) as the max angular distance
    from any test point to its nearest configuration point.
    """
    points = _normalize(points)
    dots = test_points @ points.T
    max_dots = np.max(dots, axis=1)
    # Clip to avoid numerical issues in arccos
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    return np.max(angles)


def _optimize_covering(points, test_points):
    """
    Optimize the configuration to minimize the covering radius using Nelder-Mead.
    """
    def objective(params):
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts, test_points)
        
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # 1. Initialize with Fibonacci Spiral
    points = _fibonacci_sphere(N)
    
    # 2. Generate a set of random test points on the sphere
    # Use a fixed seed for reproducibility in the test environment? 
    # The task doesn't specify, but consistent results are better.
    np.random.seed(42)
    # 5000 points is a good balance between accuracy and speed
    test_points = np.random.randn(5000, 3)
    test_points = _normalize(test_points)
    
    # 3. Lloyd's Algorithm to get a good initial configuration
    # This spreads the points out based on the density of test points
    points = _lloyd_iteration(points, test_points, max_iter=50)
    
    # 4. Local Optimization on Covering Radius
    # Use a denser set of test points for the final optimization to be more accurate
    np.random.seed(123)
    test_points_fine = np.random.randn(10000, 3)
    test_points_fine = _normalize(test_points_fine)
    
    points = _optimize_covering(points, test_points_fine)
    
    # 5. Final Normalization
    points = _normalize(points)
    
    return points
# EVOLVE-BLOCK-END
