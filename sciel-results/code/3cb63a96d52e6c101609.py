# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """Calculate the minimum area among all triangles formed by the points."""
    n = len(points)
    # Generate all unique triplets
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    
    # Get points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas
    areas = 0.5 * np.abs(p1[:, 0]*(p2[:, 1] - p3[:, 1]) + 
                         p2[:, 0]*(p3[:, 1] - p1[:, 1]) + 
                         p3[:, 0]*(p1[:, 1] - p2[:, 1]))
    
    return np.min(areas)

def _hull_area(points):
    """Calculate the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _objective(params):
    """
    Objective function for optimization.
    params: array of shape (22,) for points 3 to 13.
    Returns negative ratio to minimize.
    """
    # Reconstruct full points array
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    rest = params.reshape(-1, 2)
    points = np.vstack((p0, p1, rest))
    
    # Check for degeneracy
    a_min = _min_triangle_area(points)
    if a_min < 1e-10:
        return 1e10
        
    a_hull = _hull_area(points)
    if a_hull < 1e-10:
        return 1e10
        
    ratio = a_min / a_hull
    return -ratio

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area using numerical optimization.
    """
    n = 13
    
    # Initialize with a regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    reg_points = np.column_stack((x, y))
    
    # Transform regular polygon so that p0=(0,0) and p1=(1,0)
    v0 = np.array([1.0, 0.0])
    v1 = np.array([np.cos(2*np.pi/13), np.sin(2*np.pi/13)])
    
    # Translate
    v0_t = v0 - v0
    v1_t = v1 - v0
    
    # Scale
    d = np.linalg.norm(v1_t)
    s = 1.0 / d
    v0_s = v0_t * s
    v1_s = v1_t * s
    
    # Rotate
    angle_v1_s = np.arctan2(v1_s[1], v1_s[0])
    rot_matrix = np.array([
        [np.cos(-angle_v1_s), -np.sin(-angle_v1_s)],
        [np.sin(-angle_v1_s),  np.cos(-angle_v1_s)]
    ])
    
    transformed_points = (reg_points - v0) * s
    transformed_points = transformed_points @ rot_matrix.T
    
    # Force p0 and p1
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    initial_rest = transformed_points[2:].flatten()
    
    # Optimize
    result = minimize(_objective, initial_rest, method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 2000})
    
    if result.success or result.fun < -0.1:
        optimal_rest = result.x.reshape(-1, 2)
        points = np.vstack((p0, p1, optimal_rest))
    else:
        points = reg_points
        
    return points

# EVOLVE-BLOCK-END
