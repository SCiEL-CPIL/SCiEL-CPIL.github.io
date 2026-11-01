"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    points = np.asarray(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, compute brute force
        # Sample sphere points
        n_samples = 20000
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((n_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        # For each sample, find min distance to any point
        # This is expensive but fallback
        dists = np.zeros(n_samples)
        for i in range(n_samples):
            dots = samples[i] @ points.T
            dots = np.clip(dots, -1.0, 1.0)
            dists[i] = np.arccos(dots.min())
        return np.degrees(dists.max())
    
    # Get facets
    facets = hull.simplices
    # For each facet, compute the circumradius (max angular distance from facet center to vertices)
    # Actually, the covering radius is the max over all points on sphere of distance to nearest point.
    # This equals the max circumradius of the Delaunay triangulation on the sphere.
    # For convex hull facets (triangles), the circumradius of each triangle (on the sphere) gives the worst case.
    
    max_radius = 0.0
    for tri in facets:
        v0, v1, v2 = points[tri[0]], points[tri[1]], points[tri[2]]
        # The center of the empty cap is the circumcenter of the triangle on the sphere
        # which is the normalized cross product of (v1-v0) x (v2-v0) ... no wait.
        # The circumcenter of three points on a sphere is the intersection of the sphere with the plane through the three points,
        # but specifically it's the point equidistant from all three.
        # For a triangle on the unit sphere, the circumcenter (in the spherical sense) is:
        # c = (v0 * |v1 x v2| + v1 * |v2 x v0| + v2 * |v0 x v1|) / |...|
        # Actually, the circumcenter of a spherical triangle is the normalized vector:
        # c = (v0 * sin(2*A) + v1 * sin(2*B) + v2 * sin(2*C)) where A,B,C are angles
        # Simpler: the circumcenter is the intersection of the perpendicular bisectors.
        # For three points on the unit sphere, the circumcenter is:
        # c = (v0 * (|v1|^2 * |v2|^2 - (v1·v2)^2) + ...) -- this is getting complex.
        
        # Alternative: the circumradius of the spherical triangle is the angular distance from the circumcenter to any vertex.
        # The circumcenter can be found as the normalized vector:
        # c = (v0 * (v1·v2 - v0·v1 * v0·v2 / (v0·v1)) + ...) -- no.
        
        # Let's use the formula: the circumcenter of three points a, b, c on the unit sphere is:
        # n = (b - a) x (c - a)  -- this is the normal to the plane, not what we want.
        # Actually, the circumcenter is the point p on the sphere such that |p-a|=|p-b|=|p-c|.
        # This means p·a = p·b = p·c.
        # So p is in the null space of [a-b; b-c] (in the sense that p·(a-b)=0 and p·(b-c)=0).
        # So p is parallel to (a-b) x (b-c).
        d1 = v1 - v0
        d2 = v2 - v1
        c = np.cross(d1, d2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-15:
            continue
        c = c / c_norm
        # Check if c is on the correct side (same hemisphere as the triangle)
        # The centroid of the triangle
        centroid = (v0 + v1 + v2) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        # The angular distance from c to v0 is the circumradius
        dot_val = np.clip(np.dot(c, v0), -1.0, 1.0)
        radius = np.arccos(dot_val)
        if radius > max_radius:
            max_radius = radius
    
    return np.degrees(max_radius)


def _objective_and_grad(x, points):
    """Objective function: sum of squared covering radius contributions, or just the covering radius."""
    # Reshape x to points
    pts = x.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    pts = pts / norms
    
    # Compute covering radius
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        max_r = 0.0
        for tri in facets:
            v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            d1 = v1 - v0
            d2 = v2 - v1
            c = np.cross(d1, d2)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-15:
                continue
            c = c / c_norm
            centroid = (v0 + v1 + v2) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            dot_val = np.clip(np.dot(c, v0), -1.0, 1.0)
            r = np.arccos(dot_val)
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        return np.pi  # Bad configuration


def _initial_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Start with Fibonacci spiral
    points = _initial_points()
    best_points = points.copy()
    best_score = _covering_radius(best_points)
    
    # Local optimization using gradient-free methods
    # We'll use a simple iterative refinement: for each point, move it to reduce the covering radius
    
    rng = np.random.default_rng(12345)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Randomly select a point to move
        idx = rng.integers(0, N)
        p = points[idx].copy()
        
        # Try small random perturbations
        best_local = p.copy()
        best_local_score = best_score
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Generate a random perturbation
            delta = rng.standard_normal(3) * 0.05
            new_p = p + delta
            new_p = new_p / np.linalg.norm(new_p)
            
            # Check if new_p is too close to another point
            too_close = False
            for j in range(N):
                if j == idx:
                    continue
                if np.dot(new_p, points[j]) > 0.9999:
                    too_close = True
                    break
            if too_close:
                continue
            
            # Test the new configuration
            test_points = points.copy()
            test_points[idx] = new_p
            test_score = _covering_radius(test_points)
            
            if test_score < best_local_score:
                best_local_score = test_score
                best_local = new_p
        
        if best_local_score < best_score - 1e-10:
            points[idx] = best_local
            best_score = best_local_score
            best_points = points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
