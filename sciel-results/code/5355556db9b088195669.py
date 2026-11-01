import numpy as np
import scipy.spatial
import time

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of the given points on the unit sphere.
    Returns the radius in radians and the index of the worst point (circumcenter).
    """
    # Convex hull of the points
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., points are coplanar or degenerate)
        # Calculate max distance to nearest point on a grid
        grid = _generate_grid(100)
        dists = np.array([np.min(np.arccos(np.clip(np.dot(grid, p), -1, 1))) for p in points])
        return np.max(dists), -1

    worst_radius = 0
    worst_facet_idx = -1
    worst_circumcenter = None

    # Iterate over facets (triangles) of the convex hull
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # Normal to the plane
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # The circumcenter is either n or -n.
        # We choose the one that is "close" to the triangle (dot product > 0 with a)
        if np.dot(n, a) < 0:
            n = -n
            
        # Distance from circumcenter to vertices
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        
        if r > worst_radius:
            worst_radius = r
            worst_facet_idx = simplex[0] # Just an index, we need the simplex
            # Store simplex indices
            worst_simplex = simplex
            worst_circumcenter = n

    return worst_radius, worst_simplex, worst_circumcenter

def _generate_grid(n_theta=50, n_phi=100):
    """Generate a grid of points on the sphere for fallback evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _local_search(points, max_steps=500, step_size=0.1):
    """
    Local search to minimize the covering radius.
    Moves the vertices of the worst facet towards its circumcenter.
    """
    current_points = points.copy()
    best_radius, _, _ = _covering_radius(current_points)
    
    for step in range(max_steps):
        radius, simplex, circumcenter = _covering_radius(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        # If radius is very small, we might be stuck or converged
        if radius < 1e-6:
            break
            
        # Get the 3 points of the worst facet
        idx1, idx2, idx3 = simplex
        
        # Move these points towards the circumcenter
        # We use a small step size to avoid overshooting
        alpha = step_size
        
        p1 = current_points[idx1]
        p2 = current_points[idx2]
        p3 = current_points[idx3]
        
        # Move towards circumcenter
        # new_p = normalize(p + alpha * (circumcenter - p))
        # This is a gradient step on the distance to the circumcenter
        
        # To be more robust, we can move all 3 points
        new_p1 = p1 + alpha * (circumcenter - p1)
        new_p2 = p2 + alpha * (circumcenter - p2)
        new_p3 = p3 + alpha * (circumcenter - p3)
        
        # Normalize
        new_p1 = new_p1 / np.linalg.norm(new_p1)
        new_p2 = new_p2 / np.linalg.norm(new_p2)
        new_p3 = new_p3 / np.linalg.norm(new_p3)
        
        current_points[idx1] = new_p1
        current_points[idx2] = new_p2
        current_points[idx3] = new_p3
        
        # Occasionally, we might want to perturb other points to escape local minima
        # But for now, this greedy approach is fast.
        
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_points = None
    best_radius = np.inf
    
    # Number of restarts
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize points
        # Use Fibonacci sphere with random perturbation
        rng = np.random.default_rng(seed=42 + i)
        
        # Base Fibonacci
        j = np.arange(N) + 0.5
        z = 1.0 - 2.0 * j / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * j
        r = np.sqrt(1.0 - z * z)
        base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add noise
        noise = rng.normal(0, 0.1, size=(N, 3))
        points = base_points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Run local search
        pts, rad = _local_search(points, max_steps=200, step_size=0.05)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts
            
        # If we found a very good solution, we can stop early?
        # The best known is ~18.3 deg = 0.319 rad
        if best_radius < 0.32:
            # Continue to see if we can improve further, but we have a good candidate
            pass

    # Final polish on the best point found
    if best_points is not None:
        # Run a finer local search
        pts, rad = _local_search(best_points, max_steps=500, step_size=0.01)
        if rad < best_radius:
            best_radius = rad
            best_points = pts

    if best_points is None:
        # Fallback to seed
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_points
