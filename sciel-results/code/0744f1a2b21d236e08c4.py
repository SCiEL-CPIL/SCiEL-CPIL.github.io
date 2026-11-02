"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull
import math

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n

def _covering_radius(points):
    """Compute covering radius in degrees using convex hull facets."""
    points = np.asarray(points)
    points = np.array([_normalize(p) for p in points])
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use max distance to nearest point on a grid
        # This is a rough estimate
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2*np.pi, 200)
        T, P = np.meshgrid(theta, phi)
        grid_x = np.sin(T) * np.cos(P)
        grid_y = np.sin(T) * np.sin(P)
        grid_z = np.cos(T)
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        # Compute max distance to nearest point
        max_dist = 0.0
        for gp in grid_pts:
            dists = np.arccos(np.clip(np.dot(points, gp), -1, 1))
            max_dist = max(max_dist, dists.min())
        return np.degrees(max_dist)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        # Compute circumcenter of the facet on the sphere
        # The circumcenter of a spherical triangle is the direction of the vector sum of the vertices
        # (for small facets, this approximates the center of the empty cap)
        c = np.sum(facet_pts, axis=0)
        c = _normalize(c)
        
        # Check if the cap is empty: c should be a convex combination direction
        # The covering radius contribution is the angular distance from c to the farthest vertex
        # But we need to ensure c is inside the facet. For a convex hull on the sphere,
        # the circumcenter of each facet gives the center of the largest empty circle in that region.
        
        # Compute angular distance from c to each vertex of the facet
        dists = np.arccos(np.clip(np.dot(facet_pts, c), -1, 1))
        radius = dists.max()
        
        # Verify that c is actually in the interior of the cap defined by the facet
        # The cap is the set of points y such that dot(y, v) >= dot(c, v) for all v in facet
        # Actually, the covering radius for this facet is the max distance from c to vertices
        # only if c is the circumcenter. Let's use a more robust approach:
        # Find the point in the facet that maximizes the minimum distance to all points.
        
        max_radius = max(max_radius, radius)
    
    return np.degrees(max_radius)

def _objective_and_grad(x_flat):
    """Objective function for optimization: covering radius."""
    x = x_flat.reshape(N, 3)
    x = np.array([_normalize(p) for p in x])
    cr = _covering_radius(x)
    return cr, None

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Local optimization using Nelder-Mead on subsets
    # Since full Nelder-Mead on 390 dimensions is too slow, use coordinate descent or 
    # optimize small neighborhoods
    
    # Strategy: iterate, for each point, optimize its position given others fixed
    for iteration in range(20):
        for idx in range(N):
            # Optimize point idx
            p = points[idx]
            
            def sub_obj(delta):
                new_p = _normalize(p + delta)
                new_points = points.copy()
                new_points[idx] = new_p
                return _covering_radius(new_points)
            
            # Use Nelder-Mead in 3D for this point
            res = minimize(sub_obj, np.zeros(3), method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success:
                points[idx] = _normalize(p + res.x)
    
    # Final verification
    points = np.array([_normalize(p) for p in points])
    return points
# EVOLVE-BLOCK-END
