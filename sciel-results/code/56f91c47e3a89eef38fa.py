"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130
DEADLINE = 140.0  # wall-clock seconds; keep safety margin before 150s kill

# EVOLVE-BLOCK-START


def _normalize(pts):
    """Normalize rows of pts to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius of points on the unit sphere.
    Uses the convex hull: the covering radius is the maximum circumradius
    of the facets (2D faces) of the convex hull.
    For a triangle with vertices a, b, c on the unit sphere, the circumradius
    (angular) is computed from the side lengths.
    """
    from scipy.spatial import ConvexHull
    import math

    # Compute convex hull
    hull = ConvexHull(pts)
    # Hull.simplices gives the 2D faces (triangles) for a 3D hull
    simplices = hull.simplices

    # For each triangular face, compute the circumradius
    # The circumradius R of a triangle with side lengths a, b, c is:
    # R = abc / (4 * area)
    # where area can be computed via Heron's formula or cross product.
    # Since points are on the unit sphere, the side lengths are chord lengths.
    # The angular circumradius is arcsin(chord_circumradius) but we need the
    # angular radius which is the angle from the sphere center to the circumcenter.
    # Actually, for points on the unit sphere, the circumradius of the spherical
    # triangle (angular) can be found differently.
    # 
    # Better approach: The covering radius is the maximum over all points on the
    # sphere of the distance to the nearest point. This equals the maximum
    # circumradius of the Delaunay triangulation on the sphere, which corresponds
    # to the convex hull facets.
    #
    # For a planar triangle with vertices on the unit sphere, the circumcenter
    # of the triangle (in 3D) projected back... actually, the angular circumradius
    # is the angle between the origin and the circumcenter of the triangle plane
    # intersected with the sphere... 
    #
    # Simpler: for a triangle with chord lengths a, b, c, the circumradius in
    # 3D (Euclidean) is R_euc = abc / (4 * area). The angular radius is
    # arcsin(R_euc) if R_euc <= 1, but that's not quite right either.
    #
    # Actually, the correct formula: for a spherical triangle, the circumradius
    # (angular distance from circumcenter to vertices) can be computed. But since
    # the vertices are on the unit sphere and the face is a planar triangle,
    # the circumcenter of the planar triangle lies along the normal from the
    # origin. The angular circumradius theta satisfies:
    #   sin(theta) = R_euc * cos(theta) ... no.
    #
    # Let me think again. The three points a, b, c are on the unit sphere.
    # The plane through them has a circumcenter O_c (in 3D space) and circumradius
    # R_euc (Euclidean distance from O_c to a, b, c). The angular circumradius
    # is the angle at the origin between the origin and any of a, b, c measured
    # from the direction of O_c... 
    #
    # Actually, the covering radius for a facet is the angular distance from the
    # origin to the farthest point in the spherical triangle, which is the
    # circumcenter of the spherical triangle. For a planar triangle inscribed
    # in the unit sphere, the angular circumradius is:
    #   theta = arcsin(R_euc)
    # where R_euc is the Euclidean circumradius of the planar triangle.
    # Wait, that's only if the circumcenter is on the sphere, which it isn't.
    #
    # Correct approach: The angular circumradius of a spherical triangle with
    # vertices at angular distances from origin = 1 is computed as follows:
    # If the planar triangle has circumradius R_euc and the distance from origin
    # to the plane is d, then R_euc^2 + d^2 = 1 (since vertices are on unit sphere
    # and circumcenter is in the plane). The angular circumradius theta satisfies
    # cos(theta) = d, so theta = arccos(d) = arccos(sqrt(1 - R_euc^2)).
    #
    # So: theta = arccos(sqrt(1 - R_euc^2))

    max_theta = 0.0
    for idx in simplices:
        a = pts[idx[0]]
        b = pts[idx[1]]
        c = pts[idx[2]]
        # Chord lengths
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)
        # Area of triangle via cross product
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area < 1e-15:
            continue
        # Euclidean circumradius
        R_euc = (ab * bc * ca) / (4.0 * area)
        if R_euc > 1.0:
            R_euc = 1.0
        d = np.sqrt(max(0.0, 1.0 - R_euc * R_euc))
        theta = np.arccos(d)
        if theta > max_theta:
            max_theta = theta

    return np.degrees(max_theta)


def _initial_fibonacci():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _repulsive_energy(pts):
    """
    Compute a repulsive energy that encourages spreading.
    Use sum of inverse distances or similar.
    """
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)  # avoid division by zero
    # Use 1/d^2 as repulsive potential
    energy = np.sum(1.0 / (dists ** 2))
    return energy


def _local_search(pts, max_iters=5000, step_scale=0.1):
    """
    Local search: iteratively move each point to reduce the covering radius.
    Use a greedy approach: for each point, try small perturbations and keep
    the best one.
    """
    pts = pts.copy()
    current_cr = _covering_radius(pts)
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Pick a random point
        idx = rng.integers(0, N)
        orig = pts[idx].copy()
        
        # Try several perturbations
        best_pt = orig
        best_cr = current_cr
        
        for trial in range(20):
            # Random direction
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            # Small step
            step = step_scale * (0.5 + 0.5 * rng.random())
            candidate = orig + step * direction
            candidate = _normalize(candidate.reshape(1, 3))[0]
            
            pts[idx] = candidate
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pt = candidate.copy()
        
        pts[idx] = best_pt
        current_cr = best_cr
    
    return pts, current_cr


def _simulated_annealing(pts, max_iters=10000, t_start=0.5, t_end=0.01):
    """
    Simulated annealing to escape local minima.
    """
    pts = pts.copy()
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Annealing schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Pick a random point and perturb it
        idx = rng.integers(0, N)
        orig = pts[idx].copy()
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        step = 0.1 * (0.5 + 0.5 * rng.random())
        candidate = orig + step * direction
        candidate = _normalize(candidate.reshape(1, 3))[0]
        
        pts[idx] = candidate
        new_cr = _covering_radius(pts)
        delta = new_cr - current_cr
        
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = pts.copy()
        else:
            pts[idx] = orig
    
    return best_pts, best_cr


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    # Phase 1: Local search
    pts, cr = _local_search(pts, max_iters=3000, step_scale=0.05)
    if cr < best_cr:
        best_cr = cr
        best_pts = pts.copy()
    
    if time.time() > DEADLINE - 30:
        return _normalize(best_pts)
    
    # Phase 2: Simulated annealing
    pts, cr = _simulated_annealing(pts, max_iters=5000, t_start=0.3, t_end=0.005)
    if cr < best_cr:
        best_cr = cr
        best_pts = pts.copy()
    
    if time.time() > DEADLINE - 10:
        return _normalize(best_pts)
    
    # Phase 3: Final local search polish
    pts, cr = _local_search(pts, max_iters=2000, step_scale=0.02)
    if cr < best_cr:
        best_cr = cr
        best_pts = pts.copy()
    
    return _normalize(best_pts)


# EVOLVE-BLOCK-END
