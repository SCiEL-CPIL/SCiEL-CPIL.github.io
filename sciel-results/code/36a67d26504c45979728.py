"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    Uses the convex hull facets: the covering radius is the maximum circumradius
    of the spherical caps corresponding to each facet.
    For a facet with vertices v1, v2, ..., vk on the unit sphere, the cap center
    is the normalized centroid (or better, the point on the sphere farthest from
    all facet vertices within the cap). The angular distance from the cap center
    to the farthest vertex gives the cap radius.
    
    A more accurate approach: for each triangular facet (from Delaunay triangulation
    of the convex hull), compute the circumcircle of the three points on the sphere.
    The covering radius is the max over all such circumcircles of the angular distance
    from the circumcenter to any of the three vertices.
    """
    pts = _to_sphere(pts)
    # Build Delaunay triangulation of the convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute-force max angular distance
        return _covering_radius_brute(pts)
    
    # Get the triangular facets from the hull
    simplices = hull.simplices  # shape (n_facets, 3)
    
    max_rad = 0.0
    for tri in simplices:
        v1, v2, v3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Compute the circumcenter of the spherical triangle
        # The circumcenter on the sphere is in the direction of the cross product
        # approach: find point c on sphere equidistant from v1, v2, v3
        # c is proportional to (v2-v1) x (v3-v1) ... no, that's the normal.
        # Actually, the circumcenter of a spherical triangle can be found by:
        # The point c on the unit sphere such that c·v1 = c·v2 = c·v3 = cos(theta)
        # This means c is in the plane defined by the three points' "circumcircle"
        
        # Method: The circumcenter direction is given by the vector:
        # c_dir = (v1 x v2) x (v2 x v3) ... let me think differently.
        
        # For three unit vectors, the circumcenter is along the direction:
        # c_dir = v1 * |v2-v3|^2 + v2 * |v3-v1|^2 + v3 * |v1-v2|^2  -- no that's not right either
        
        # Better approach: Solve for c = a*v1 + b*v2 + c*v3 such that
        # c·v1 = c·v2 = c·v3 and |c|=1
        # c·v1 = a + b*(v2·v1) + c*(v3·v1)
        # c·v2 = a*(v1·v2) + b + c*(v3·v2)
        # c·v3 = a*(v1·v3) + b*(v2·v3) + c
        
        # Let d12 = v1·v2, d13 = v1·v3, d23 = v2·v3
        # a + b*d12 + c*d13 = k
        # a*d12 + b + c*d23 = k
        # a*d13 + b*d23 + c = k
        
        # Subtract: (a - b)*1 + (b - c)*d12 ... this is getting complex.
        
        # Simpler: The circumcenter of three points on a sphere lies in the direction
        # of the vector: (v1 - v2) x (v2 - v3) is the normal to the plane through v1,v2,v3.
        # The circumcenter is the intersection of the sphere with the line through the
        # origin in the direction of the circumcenter of the planar triangle projected.
        
        # Actually, for unit vectors, the circumcenter direction is:
        # c_dir = (v1 x v2) * |v3 - proj_{v1xv2} ... 
        
        # Let me use a direct formula. For three unit vectors a, b, c:
        # The circumcenter is in the direction of:
        # (b - a) x (c - a) gives the normal to the plane.
        # The circumcenter lies along: n * t where n is the normal? No.
        
        # Correct approach: The spherical circumcenter is the point on the sphere
        # equidistant from all three vertices. It lies in the plane of the three points.
        # The plane through three unit vectors has normal n = (v2-v1) x (v3-v1).
        # The circumcenter is the point on the sphere closest to the origin in the 
        # intersection of the sphere and the plane? No.
        
        # Let me just use: for a spherical triangle, the circumradius can be computed
        # from the side lengths using the spherical law of cosines.
        
        # Side lengths (angular distances):
        d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
        d23 = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
        d31 = np.arccos(np.clip(np.dot(v3, v1), -1, 1))
        
        # Spherical circumradius R satisfies:
        # sin(R) = sin(d12)*sin(d23)*sin(d31) / (2 * sin(d12)*sin(d23)*sin(d31) ... )
        # Actually the formula is:
        # sin(R) = (sin(d12) * sin(d23) * sin(d31)) / (2 * sin(d12) * sin(d23) * sin(d31) * cos(A/2)...)
        # This is getting complicated. Let me use a numerical approach.
        
        # The circumcenter c on the sphere: minimize max(|c-vi|) subject to |c|=1
        # Equivalently, c is in the direction of the weighted sum.
        
        # Direct computation: c = alpha * (v1 x v2) x (v2 x v3) ... 
        # Actually, the circumcenter direction for three points on a unit sphere is:
        # c_dir = v1 * (1 - d23^2/2) + v2 * (1 - d31^2/2) + v3 * (1 - d12^2/2)
        # No, let me just solve the linear system.
        
        d12d = np.dot(v1, v2)
        d13d = np.dot(v1, v3)
        d23d = np.dot(v2, v3)
        
        # We want c = a*v1 + b*v2 + c*v3 with c·v1 = c·v2 = c·v3
        # Matrix: [[1, d12d, d13d], [d12d, 1, d23d], [d13d, d23d, 1]] * [a,b,c]^T = k*[1,1,1]^T
        # The null space gives the direction. Set k=1 and solve.
        M = np.array([[1, d12d, d13d],
                      [d12d, 1, d23d],
                      [d13d, d23d, 1]])
        rhs = np.array([1, 1, 1])
        
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Degenerate triangle, use centroid
            c_dir = (v1 + v2 + v3) / 3
        else:
            c_dir = sol[0] * v1 + sol[1] * v2 + sol[2] * v3
        
        c_norm = np.linalg.norm(c_dir)
        if c_norm < 1e-12:
            c_dir = (v1 + v2 + v3) / 3
            c_norm = np.linalg.norm(c_dir)
        c_dir = c_dir / c_norm
        
        # Check that c_dir is on the "correct" side (inside the cap)
        # The cap center should be on the same side as the facet normal
        # Compute facet normal
        n = np.cross(v2 - v1, v3 - v1)
        if np.dot(c_dir, n) < 0:
            c_dir = -c_dir
        
        # The angular radius is the max angular distance from c_dir to the three vertices
        ang1 = np.arccos(np.clip(np.dot(c_dir, v1), -1, 1))
        ang2 = np.arccos(np.clip(np.dot(c_dir, v2), -1, 1))
        ang3 = np.arccos(np.clip(np.dot(c_dir, v3), -1, 1))
        rad = max(ang1, ang2, ang3)
        
        if rad > max_rad:
            max_rad = rad
    
    return np.degrees(max_rad)


def _covering_radius_brute(pts, n_samples=20000):
    """Brute-force estimate of covering radius."""
    pts = _to_sphere(pts)
    # Sample points on the sphere
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(n_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # For each sample, find min distance to any point
    # Use matrix multiplication for speed
    # dist[i,j] = 1 - samples[i]·pts[j]
    dots = samples @ pts.T
    min_dots = np.max(dots, axis=1)
    max_ang = np.arccos(np.clip(min_dots, -1, 1))
    return np.degrees(np.max(max_ang))


def _objective(pts):
    """The score to minimize: covering radius in degrees."""
    return _covering_radius(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_score = _objective(pts)
    best_pts = pts.copy()
    
    rng = np.random.default_rng(12345)
    
    # Local optimization: gradient-free (Nelder-Mead style or simple perturbation)
    # We'll use a simple approach: repeatedly perturb random points and keep if improved.
    
    step_size = 0.05
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Generate a small random perturbation
        delta = rng.normal(scale=step_size, size=3)
        new_pts = pts.copy()
        new_pts[idx] = pts[idx] + delta
        new_pts = _to_sphere(new_pts)
        
        # Evaluate
        new_score = _objective(new_pts)
        
        if new_score < best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            # Decrease step size slightly
            step_size = max(0.001, step_size * 0.995)
        else:
            # Revert
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
