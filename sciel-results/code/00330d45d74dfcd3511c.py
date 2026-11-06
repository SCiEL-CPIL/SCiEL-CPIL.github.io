"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _initial_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _covering_radius(pts):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest point in the set. This is equivalent to the maximum circumradius
    of the spherical Voronoi cells (facets of the convex hull).
    """
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        for simplex in hull.simplices:
            # Get the 3 points forming the facet
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # Compute circumcenter of the triangle p1, p2, p3 in 3D
            # Formula: circumcenter = p1 + cross(p2-p1, p3-p1) * something...
            # Easier: solve linear system or use vector algebra.
            # Let v1 = p2 - p1, v2 = p3 - p1
            # Circumcenter C = p1 + a*v1 + b*v2
            # |C - p1|^2 = |C - p2|^2 => 2 C . v1 = |v1|^2
            # |C - p1|^2 = |C - p3|^2 => 2 C . v2 = |v2|^2
            
            v1 = p2 - p1
            v2 = p3 - p1
            
            # Solve for coefficients a, b such that:
            # v1 . (a*v1 + b*v2) = 0.5 * |v1|^2
            # v2 . (a*v1 + b*v2) = 0.5 * |v2|^2
            # Matrix A = [[v1.v1, v1.v2], [v2.v1, v2.v2]]
            # RHS = [0.5*|v1|^2, 0.5*|v2|^2]
            
            A = np.array([
                [np.dot(v1, v1), np.dot(v1, v2)],
                [np.dot(v2, v1), np.dot(v2, v2)]
            ])
            b = np.array([
                0.5 * np.dot(v1, v1),
                0.5 * np.dot(v2, v2)
            ])
            
            try:
                coeffs = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Degenerate triangle, skip or handle
                continue
                
            c = p1 + coeffs[0] * v1 + coeffs[1] * v2
            
            # Project circumcenter to unit sphere
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-10:
                continue
            c_unit = c / norm_c
            
            # Calculate angular distances from vertices to c_unit
            # Angular distance = arccos(clamp(dot, -1, 1))
            d1 = np.arccos(np.clip(np.dot(p1, c_unit), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(p2, c_unit), -1.0, 1.0))
            d3 = np.arccos(np.clip(np.dot(p3, c_unit), -1.0, 1.0))
            
            # The local covering radius for this facet is the max distance to the vertices
            # Note: If the circumcenter lies outside the spherical triangle, the farthest point
            # might be a vertex, but for small triangles (acute), the circumcenter is inside.
            # For N=50, triangles are small.
            local_rad = max(d1, d2, d3)
            
            if local_rad > max_rad:
                max_rad = local_rad
                
        return max_rad
    except Exception:
        return np.pi # Return large value if hull fails

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Start with Fibonacci spiral
    pts = _initial_points()
    
    # Flatten for optimizer
    x0 = pts.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to ensure they stay on sphere during optimization steps
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        return _covering_radius(pts)
        
    # Use Nelder-Mead as the objective is non-smooth (max function)
    # maxiter is set high, but we will check time in a wrapper if needed, 
    # or rely on the fact that Nelder-Mead is fast enough for N=50.
    
    best_pts = pts
    best_score = objective(x0)
    
    # Run optimization
    # We can try a few restarts or just one long run.
    # Given 120s limit, one Nelder-Mead run with ~1000-2000 iterations should be safe.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            best_pts = best_pts / norms
    except Exception:
        pass

    # Check time, if we have time, we could do a second pass or refinement
    # But Nelder-Mead is usually the bottleneck.
    
    return best_pts

# EVOLVE-BLOCK-END
