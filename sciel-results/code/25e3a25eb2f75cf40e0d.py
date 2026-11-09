"""
Spherical Tammes/Covering optimization for N=130 points.
Strategy:
1. High-quality initialization (Fibonacci sphere).
2. Robust Lloyd's Algorithm (Voronoi centroid relaxation) using a fine grid.
   This is a structural change from generic gradient descent, specifically
   targeting the geometry of the covering problem by moving points to the
   centers of their "worst-case" empty regions (Voronoi cells).
3. Final polishing with L-BFGS-B to handle the non-smoothness of the max-circumradius
   objective near the optimum.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum circumradius of the facets
    of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            tri = points[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            # Circumcenter x satisfies x·a = x·b = x·c and |x|=1
            # x is in the intersection of planes x·(a-b)=0 and x·(b-c)=0
            v1 = a - b
            v2 = b - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-10:
                continue
            x = x_dir / x_norm
            
            # Ensure x is on the same side as the triangle centroid (outward normal)
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                if np.dot(x, centroid) < 0:
                    x = -x
            
            cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            
            max_circumradius = max(max_circumradius, R)
        
        return max_circumradius
    except Exception:
        # Fallback: grid-based approximation
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        dots = grid_points @ points.T
        cos_angles = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        min_angles = np.min(angles, axis=1)
        return np.max(min_angles)

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_iteration(points, grid):
    """
    One iteration of Lloyd's algorithm.
    Move each point to the centroid of its Voronoi cell (approximated by grid).
    """
    n = len(points)
    new_points = np.zeros_like(points)
    
    # Assign each grid point to its nearest configuration point
    # dot product is maximized when angle is minimized
    dots = grid @ points.T  # (num_grid, N)
    nearest = np.argmax(dots, axis=1)
    
    # Compute centroid of Voronoi cell for each point
    # Using bincount for efficiency
    # We need to sum the grid points for each label
    # grid is (M, 3), nearest is (M,)
    # We can use np.bincount with weights, but we need 3D weights.
    # Alternatively, loop is fine for N=130 if M is not huge, but let's try vectorized.
    
    # Efficient way:
    # For each dimension, sum grid[:, d] where nearest == i
    # We can use np.add.at or bincount if we flatten, but bincount only does 1D.
    # Let's use a simple loop over N, which is 130. M is ~40000.
    # 130 * (boolean mask + sum) is fast enough in numpy.
    
    for i in range(n):
        mask = (nearest == i)
        count = np.sum(mask)
        if count > 0:
            cell_points = grid[mask]
            centroid = np.mean(cell_points, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                new_points[i] = centroid / centroid_norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
            
    return new_points

def _lloyd_optimize(points, time_limit):
    """
    Run Lloyd's algorithm for a specified time limit.
    """
    start_time = time.time()
    
    # Create a fine grid for Voronoi approximation
    # 200x200 grid is ~40k points, very fast for N=130
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    current = points.copy()
    best = current.copy()
    best_val = _covering_radius(best)
    
    for it in range(1000): # Max iterations, will break on time
        if time.time() - start_time > time_limit:
            break
        
        new_pts = _lloyd_iteration(current, grid)
        
        # Check convergence
        if np.max(np.linalg.norm(new_pts - current, axis=1)) < 1e-7:
            break
            
        current = new_pts
        
        # Evaluate
        val = _covering_radius(current)
        if val < best_val:
            best_val = val
            best = current.copy()
            
    return best, best_val

def _polish(points, time_limit):
    """
    Final polish using L-BFGS-B.
    """
    start_time = time.time()
    
    def flat_objective(flat):
        pts = _normalize(flat)
        return _covering_radius(pts)
    
    best = points.copy()
    best_val = _covering_radius(best)
    
    try:
        result = minimize(
            flat_objective,
            points.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 400, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.fun < best_val:
            best = _normalize(result.x)
            best_val = result.fun
    except Exception:
        pass
    
    return best, best_val

def construct():
    """Construct 130 points on the unit sphere to minimize covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    # 1. Initialize
    points = _fibonacci_sphere(N)
    
    # 2. Lloyd's Algorithm (Structural improvement: geometry-aware relaxation)
    # Allocate most time here
    lloyd_time = 100.0
    points, val = _lloyd_optimize(points, lloyd_time)
    
    # 3. Polish with Gradient Descent
    if time.time() < deadline - 10:
        points, val = _polish(points, 30.0)
        
    # 4. Final sanity check and normalization
    points = _normalize(points)
    
    # Ensure no coincident points (should be rare with Lloyd)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                points[j] = _normalize(points[j] + np.random.normal(0, 0.01, 3))
                
    return points
