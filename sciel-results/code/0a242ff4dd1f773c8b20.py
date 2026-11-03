"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This equals the maximum circumradius of the facets
    of the convex hull (the spherical Voronoi cells).
    
    We compute this by finding, for each point, the maximum angular distance to
    its neighbors that form a face. A more robust approach: the covering radius
    is the maximum over all spherical Voronoi vertices (circumcenters of spherical
    triangles) of the distance to the nearest site.
    
    For efficiency, we use the fact that the covering radius equals the maximum
    over all pairs/triples of points of the circumradius, but only for those that
    are faces of the convex hull. A simpler but valid upper bound: the maximum
    over all triples of the circumradius of the spherical triangle formed by them.
    But this is expensive. Instead, we use a practical approach:
    
    The covering radius is the smallest r such that every point on the sphere is
    within angular distance r of some site. This is equivalent to: the maximum
    over all points on the sphere of the minimum distance to a site.
    
    A key insight: the covering radius equals the maximum circumradius of the
    spherical Voronoi cells. Each Voronoi cell is a spherical polygon. The
    circumcenter of a spherical triangle (3 points) that is a face of the convex
    hull gives a Voronoi vertex.
    
    For a practical computation: we compute the convex hull of the points, find
    all triangular faces, and for each face compute the circumcenter (the point
    equidistant from all three vertices on the sphere). The covering radius is
    the maximum distance from any such circumcenter to the nearest site.
    """
    from scipy.spatial import ConvexHull
    
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices  # shape (n_faces, 3) for 3D
    except Exception:
        # Fallback: use all triples
        faces = np.array(list(itertools.combinations(range(N), 3)))
    
    # For each face (triangle of 3 points on sphere), compute the circumcenter
    # The circumcenter of a spherical triangle is the point equidistant from all three
    # vertices. It's the normalized cross product of (a-b) x (a-c) projected... 
    # Actually, the circumcenter on the sphere is the point c such that |c-a| = |c-b| = |c-c|
    # in Euclidean distance (which corresponds to equal angular distances).
    # This is equivalent to: c is perpendicular to (a-b) and (a-c), so c is parallel to (a-b) x (a-c).
    
    n_faces = len(faces)
    if n_faces == 0:
        return np.pi
    
    circumcenters = np.zeros((n_faces, 3))
    for i, face in enumerate(faces):
        a = pts[face[0]]
        b = pts[face[1]]
        c = pts[face[2]]
        # Circumcenter direction: (a-b) x (a-c)
        cc = np.cross(a - b, a - c)
        norm_cc = np.linalg.norm(cc)
        if norm_cc < 1e-12:
            circumcenters[i] = a  # degenerate
        else:
            circumcenters[i] = cc / norm_cc
    
    # The covering radius is the max over all circumcenters of the min distance to any site
    # Distance from circumcenter to nearest site
    # For each circumcenter, find min angular distance to any of the N points
    # Angular distance = arccos(dot product)
    
    # Vectorized: for each circumcenter, compute dot products with all N points
    dots = circumcenters @ pts.T  # (n_faces, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # max dot = min angular distance
    min_angles = np.arccos(min_dots)
    
    return np.max(min_angles)


def _covering_radius_grad(pts):
    """
    Compute gradient of the covering radius with respect to point positions.
    This is complex, so we use numerical gradient or a simplified version.
    For L-BFGS-B, we can use finite differences or provide an analytical gradient.
    
    Actually, for optimization, it's better to minimize a smooth surrogate.
    Let's use a different approach: minimize the sum of squared distances to
    the covering radius, or use a penalty-based approach.
    
    Alternative: use the fact that the covering radius is the max over Voronoi
    vertices. We can use a soft-max approximation for smoothness.
    """
    return None  # We'll use numerical gradient


def _soft_max_covering_radius(pts, temperature=0.01):
    """
    Smooth approximation of the covering radius using soft-max.
    """
    from scipy.spatial import ConvexHull
    
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
    except Exception:
        faces = np.array(list(itertools.combinations(range(N), 3)))
    
    n_faces = len(faces)
    if n_faces == 0:
        return np.pi, np.zeros_like(pts)
    
    circumcenters = np.zeros((n_faces, 3))
    for i, face in enumerate(faces):
        a = pts[face[0]]
        b = pts[face[1]]
        c = pts[face[2]]
        cc = np.cross(a - b, a - c)
        norm_cc = np.linalg.norm(cc)
        if norm_cc < 1e-12:
            circumcenters[i] = a
        else:
            circumcenters[i] = cc / norm_cc
    
    dots = circumcenters @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    
    # Soft max
    scaled = min_angles / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    soft_max = temperature * (max_scaled + np.log(np.sum(exp_scaled)))
    
    return soft_max, None


def _polish(pts, time_limit=100.0):
    """
    Polish the configuration using local optimization.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _normalize(p)
        # Use a smooth objective: sum of exp of angles
        # Or use the soft-max covering radius
        val, _ = _soft_max_covering_radius(p, temperature=0.05)
        return val
    
    # Use L-BFGS-B with numerical gradient
    result = minimize(
        objective_and_grad,
        pts.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if time.time() - start_time > time_limit:
        return pts
    
    pts = _normalize(result.x.reshape(N, 3))
    return pts


def _random_start(n_restarts=10, seed=42):
    """Generate random starting configurations."""
    rng = np.random.RandomState(seed)
    best_pts = None
    best_radius = np.inf
    
    start_time = time.time()
    time_limit = 120.0
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Random points on sphere
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        
        # Polish
        polished = _polish(pts, time_limit=min(20.0, time_limit - (time.time() - start_time)))
        radius = _covering_radius(polished)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = polished
    
    return best_pts, best_radius


def _fibonacci_start():
    """Golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    hard_deadline = 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    pts = _fibonacci_start()
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Try random restarts with polishing
    rng = np.random.RandomState(123)
    n_restarts = 50
    restart_count = 0
    
    for r in range(n_restarts):
        if time.time() - start_time > hard_deadline:
            break
        
        # Random perturbation of Fibonacci or fully random
        if r < n_restarts // 2:
            # Perturb Fibonacci
            pts = _fibonacci_start()
            noise = rng.randn(N, 3) * 0.1
            pts = _normalize(pts + noise)
        else:
            # Fully random
            pts = rng.randn(N, 3)
            pts = _normalize(pts)
        
        # Polish with time limit
        remaining = hard_deadline - (time.time() - start_time)
        if remaining < 5:
            break
        
        polished = _polish(pts, time_limit=min(15.0, remaining - 5))
        radius = _covering_radius(polished)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = polished.copy()
            # Continue polishing the best
            remaining2 = hard_deadline - (time.time() - start_time)
            if remaining2 > 10:
                further_polished = _polish(best_pts, time_limit=min(10.0, remaining2 - 5))
                further_radius = _covering_radius(further_polished)
                if further_radius < best_radius:
                    best_radius = further_radius
                    best_pts = further_polished
    
    # Final polish if time permits
    if time.time() - start_time < hard_deadline - 5:
        final_polished = _polish(best_pts, time_limit=min(5.0, hard_deadline - (time.time() - start_time) - 1))
        final_radius = _covering_radius(final_polished)
        if final_radius < best_radius:
            best_pts = final_polished
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
