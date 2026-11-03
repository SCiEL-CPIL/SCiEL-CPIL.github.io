"""Covering the sphere with 130 equal caps.

Strategy:
1. Start from a Fibonacci spiral (deterministic, spread out).
2. Compute the convex hull of the points on the sphere.
3. The covering radius is determined by the largest circumcircle of the hull facets.
   For each triangular facet (a, b, c) on the unit sphere, the circumcenter is the
   point on the sphere farthest from all three vertices within that spherical triangle.
   The covering radius is the maximum over all facets of the distance from the
   facet's circumcenter to its vertices (the circumradius).
4. Use gradient-based optimization to move points to reduce the covering radius.
   We use a soft-max / smooth approximation to handle the max over facets.
5. Polish with local moves: for each point, try small perturbations in the
   tangent plane and accept if they reduce the covering radius.

We implement a direct approach:
- Compute the Delaunay triangulation on the sphere (via convex hull in 3D).
- For each triangle, compute the spherical circumcenter and circumradius.
- The covering radius is the max circumradius.
- Optimize by moving points to reduce the max circumradius.

For efficiency within 120s, we use:
- A few hundred iterations of local optimization.
- Each iteration: compute hull, find worst facets, move points.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _spherical_circumcenter(a, b, c):
    """Compute the spherical circumcenter of triangle (a,b,c) on unit sphere.
    
    The circumcenter is the point on the sphere equidistant (in angular distance)
    from a, b, c. It's the intersection of the sphere with the plane through a,b,c
    ... actually, for a spherical triangle, the circumcenter is the point on the
    sphere such that the great-circle arcs to a, b, c are equal.
    
    Method: The circumcenter lies along the line through the origin perpendicular
    to the plane containing a, b, c (if the triangle doesn't contain the origin
    in its convex hull... actually for points on the unit sphere, the plane through
    a,b,c has normal n = a×b + b×c + c×a (or cross products). The circumcenter
    on the sphere is in the direction of the cross product of the edges... 
    
    Actually, a simpler approach: the circumcenter of a spherical triangle is the
    point p on the unit sphere such that p·a = p·b = p·c (cosine of angular distance
    is the same). This means p is in the null space of [a; b; c] - [1,1,1]... 
    No, p·a = p·b = p·c means (a-b)·p = 0 and (b-c)·p = 0, so p is parallel to
    (a-b) × (b-c).
    """
    # p is parallel to (a-b) × (b-c)
    ab = a - b
    bc = b - c
    cross = np.cross(ab, bc)
    cn = np.linalg.norm(cross)
    if cn < 1e-10:
        # Degenerate: points are collinear on the sphere
        # Use the midpoint of the longest edge as a fallback
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        if d_ab >= d_bc and d_ab >= d_ca:
            p = _normalize(a + b)
        elif d_bc >= d_ab and d_bc >= d_ca:
            p = _normalize(b + c)
        else:
            p = _normalize(c + a)
        return p
    p = cross / cn
    # Choose the sign that gives the correct hemisphere (the one "inside" the triangle)
    # The circumcenter should be on the same side of each edge as the opposite vertex
    # Check: for edge ab, the circumcenter should be on the same side as c
    # The side is determined by the sign of (a×b)·p vs (a×b)·c
    ab_cross = np.cross(a, b)
    if np.dot(ab_cross, p) * np.dot(ab_cross, c) < 0:
        p = -p
    # Verify and adjust
    # Actually, let's just check which sign gives smaller max distance
    d1 = np.arccos(np.clip(np.dot(p, a), -1, 1))
    d2 = np.arccos(np.clip(np.dot(-p, a), -1, 1))
    # The circumcenter should be the one that minimizes the max distance
    # But actually both p and -p are valid circumcenters; we want the one
    # that is "inside" the triangle. For a triangle that doesn't contain
    # the antipode, the circumcenter is unique.
    # Let's use the one that has the same orientation as the triangle
    # Check winding: if a,b,c are counterclockwise as seen from outside,
    # the circumcenter should be on the outside
    # Simpler: just use the one with smaller circumradius (closer to vertices)
    d_p = np.max([
        np.arccos(np.clip(np.dot(p, a), -1, 1)),
        np.arccos(np.clip(np.dot(p, b), -1, 1)),
        np.arccos(np.clip(np.dot(p, c), -1, 1)),
    ])
    d_negp = np.max([
        np.arccos(np.clip(np.dot(-p, a), -1, 1)),
        np.arccos(np.clip(np.dot(-p, b), -1, 1)),
        np.arccos(np.clip(np.dot(-p, c), -1, 1)),
    ])
    if d_negp < d_p:
        p = -p
    return p


def _covering_radius(points):
    """Compute the covering radius in radians.
    
    Uses the convex hull to find facets, then computes the circumradius
    of each facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use nearest-neighbor based estimate
        # For each point, find its 3 nearest neighbors and compute the
        # circumradius of that triangle
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(points, k=4)
        max_r = 0.0
        for i in range(len(points)):
            j, k, l = idx[i, 1], idx[i, 2], idx[i, 3]
            a, b, c = points[i], points[j], points[k]
            p = _spherical_circumcenter(a, b, c)
            r = np.max([
                np.arccos(np.clip(np.dot(p, a), -1, 1)),
                np.arccos(np.clip(np.dot(p, b), -1, 1)),
                np.arccos(np.clip(np.dot(p, c), -1, 1)),
            ])
            max_r = max(max_r, r)
        return max_r
    
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        p = _spherical_circumcenter(a, b, c)
        r = np.max([
            np.arccos(np.clip(np.dot(p, a), -1, 1)),
            np.arccos(np.clip(np.dot(p, b), -1, 1)),
            np.arccos(np.clip(np.dot(p, c), -1, 1)),
        ])
        max_r = max(max_r, r)
    return max_r


def _refine_local(points, max_iters=500, tol=1e-10, deadline=None):
    """Local refinement: iteratively move the point that contributes most to
    the covering radius toward reducing the worst facet's circumradius."""
    points = points.copy()
    n = len(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() - deadline > 5:
            break
        
        # Compute covering radius and identify worst facets
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        worst_r = 0.0
        worst_facet = None
        for simplex in hull.simplices:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            p = _spherical_circumcenter(a, b, c)
            r = np.max([
                np.arccos(np.clip(np.dot(p, a), -1, 1)),
                np.arccos(np.clip(np.dot(p, b), -1, 1)),
                np.arccos(np.clip(np.dot(p, c), -1, 1)),
            ])
            if r > worst_r:
                worst_r = r
                worst_facet = simplex
        
        if worst_facet is None:
            break
        
        # For the worst facet, try moving each of its 3 vertices
        # to reduce the circumradius
        improved = False
        for idx in worst_facet:
            pt = points[idx]
            # Generate candidate moves in the tangent plane
            # Try moving toward the circumcenter of the facet
            a, b, c = points[worst_facet[0]], points[worst_facet[1]], points[worst_facet[2]]
            p_center = _spherical_circumcenter(a, b, c)
            
            # Try several step sizes moving pt toward p_center
            for step in [0.01, 0.005, 0.001, 0.05]:
                # Move pt toward p_center by step radians
                # Use spherical linear interpolation
                cos_theta = np.clip(np.dot(pt, p_center), -1, 1)
                theta = np.arccos(cos_theta)
                if theta < 1e-10:
                    continue
                # Move by 'step' radians toward p_center
                ratio = min(step / theta, 1.0)
                new_pt = _normalize(pt * np.cos(ratio * theta) + p_center * np.sin(ratio * theta))
                
                old_r = _covering_radius(points)
                points[idx] = new_pt
                new_r = _covering_radius(points)
                
                if new_r < old_r - 1e-12:
                    improved = True
                    break
                else:
                    points[idx] = pt
            
            if improved:
                break
        
        if not improved:
            # Try random perturbations
            best_r = _covering_radius(points)
            for _ in range(20):
                idx = np.random.randint(n)
                pt = points[idx]
                # Random tangent direction
                v = np.random.randn(3)
                v -= np.dot(v, pt) * pt
                vn = np.linalg.norm(v)
                if vn < 1e-10:
                    continue
                v = v / vn
                for step in [0.01, 0.005]:
                    new_pt = _normalize(pt * np.cos(step) + v * np.sin(step))
                    points[idx] = new_pt
                    r = _covering_radius(points)
                    if r < best_r - 1e-12:
                        best_r = r
                        improved = True
                        break
                    else:
                        points[idx] = pt
                if improved:
                    break
            
            if not improved:
                break
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Local refinement
    points = _refine_local(points, max_iters=300, deadline=start_time)
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return points
    
    # Try a few more rounds with different random seeds for perturbation
    np.random.seed(42)
    for trial in range(3):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Perturb from the current best
        perturbed = points.copy()
        for idx in range(N):
            v = np.random.randn(3)
            v -= np.dot(v, perturbed[idx]) * perturbed[idx]
            vn = np.linalg.norm(v)
            if vn > 1e-10:
                v = v / vn
                perturbed[idx] = _normalize(perturbed[idx] * np.cos(0.05) + v * np.sin(0.05))
        
        refined = _refine_local(perturbed, max_iters=200, deadline=start_time)
        if _covering_radius(refined) < _covering_radius(points):
            points = refined
    
    return points
