"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees using convex hull facet analysis.
    The covering radius equals the maximum circumradius of the Delaunay facets
    (triangular faces of the convex hull whose circumcenter cap is empty).
    """
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        # The spherical circumcenter is the point on the sphere equidistant
        # (angularly) from p0, p1, p2. It is ±normalize((p0-p1) x (p0-p2)).
        cross = np.cross(p0 - p1, p0 - p2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            continue
        
        y = cross / norm_cross
        
        # Compute min angular distance from y and -y to all points
        dots_y = np.clip(np.dot(pts, y), -1.0, 1.0)
        dots_neg_y = np.clip(np.dot(pts, -y), -1.0, 1.0)
        
        min_angle_y = np.arccos(dots_y).min()
        min_angle_neg_y = np.arccos(dots_neg_y).min()
        
        # The relevant circumradius is the one with the larger min distance
        # (the empty cap direction)
        max_radius = max(max_radius, min_angle_y, min_angle_neg_y)
    
    return np.degrees(max_radius)


def _covering_radius_fallback(pts):
    """Fallback: approximate covering radius by sampling the sphere."""
    n_samples = 20000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = dots.max(axis=1)
    min_angles = np.arccos(min_dots)
    return np.degrees(min_angles.max())


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _repulsive_force_step(pts, lr, n_samples=5000):
    """
    Move points to reduce covering radius using a force-based approach.
    For each point, compute the "worst-case" direction (the direction on the
    tangent plane that maximizes the minimum distance to other points) and
    move the point in that direction.
    
    This is equivalent to: for each point, find the center of the largest
    empty cap around it, and move toward that center.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute all pairwise dot products
    dots = pts @ pts.T  # (n, n)
    dots = np.clip(dots, -1.0, 1.0)
    
    new_pts = np.zeros_like(pts)
    
    for i in range(n):
        # For point i, find the direction on the tangent plane that is
        # farthest from all other points.
        # The tangent plane at pts[i] is {v : v · pts[i] = 0}.
        # We want to find v on the unit circle in this plane that maximizes
        # min_{j != i} angular_dist(pts[i] + eps*v, pts[j]).
        # For small eps, this is approximately maximizing min_{j != i} (v · d_j)
        # where d_j is the direction from pts[i] to pts[j] projected on the tangent plane.
        
        # Compute tangent directions to all other points
        # d_j = (pts[j] - (pts[j]·pts[i]) pts[i]) / ||...||
        proj = dots[i, :]  # pts[j] · pts[i] for all j
        # Tangent component: pts[j] - proj[j] * pts[i]
        # For j = i, this is zero.
        
        # The angular distance from pts[i] to pts[j] is arccos(proj[j]).
        # Moving pts[i] in direction v (unit tangent) changes the dot product
        # with pts[j] to approximately proj[j] + eps * (v · (pts[j] - proj[j]*pts[i]))
        # = proj[j] + eps * ||tangent_j|| * cos(angle between v and tangent_j_dir)
        
        # To maximize the minimum distance, we want to move in the direction
        # that is most "away" from all neighbors. This is the direction that
        # maximizes the minimum of (v · t_j) where t_j is the unit tangent direction
        # toward neighbor j.
        
        # Compute unit tangent directions to neighbors
        tangent_vecs = np.zeros_like(pts)
        for j in range(n):
            if i == j:
                continue
            tv = pts[j] - dots[i, j] * pts[i]
            tv_norm = np.linalg.norm(tv)
            if tv_norm > 1e-12:
                tangent_vecs[j] = tv / tv_norm
        
        # Now find the direction v on the unit circle in the tangent plane
        # that maximizes min_{j != i} (v · tangent_vecs[j]).
        # This is a 2D problem. Find an orthonormal basis for the tangent plane.
        
        p = pts[i]
        # Find two orthonormal vectors perpendicular to p
        # Pick an arbitrary vector not parallel to p
        if abs(p[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, p) * p
        u /= np.linalg.norm(u)
        v_basis = np.cross(p, u)
        v_basis /= np.linalg.norm(v_basis)
        
        # Project tangent_vecs onto this basis
        # tangent_vecs[j] = a_j * u + b_j * v_basis
        a = np.array([np.dot(tangent_vecs[j], u) for j in range(n)])
        b = np.array([np.dot(tangent_vecs[j], v_basis) for j in range(n)])
        
        # We want to find (cos θ, sin θ) that maximizes min_j (a_j cos θ + b_j sin θ)
        # This is the direction that is farthest from all neighbor directions.
        
        # A good heuristic: the direction opposite to the "center of mass" of neighbors
        # weighted by their proximity.
        # Or: use the direction that is the negative of the average of the
        # nearest few neighbor directions.
        
        # Compute distances to neighbors
        dists = np.arccos(dots[i, :])
        dists[i] = np.inf
        
        # Take the 5 nearest neighbors
        k = min(5, n - 1)
        nearest_idx = np.argsort(dists)[:k]
        
        # Average their tangent directions
        avg_tangent = np.mean(tangent_vecs[nearest_idx], axis=0)
        avg_norm = np.linalg.norm(avg_tangent)
        
        if avg_norm > 1e-12:
            # Move opposite to the average neighbor direction
            move_dir = -avg_tangent / avg_norm
        else:
            # No clear direction, use a random tangent direction
            move_dir = u * np.cos(np.random.rand() * 2 * np.pi) + v_basis * np.sin(np.random.rand() * 2 * np.pi)
        
        # Project move_dir onto tangent plane (should already be there)
        move_dir = move_dir - np.dot(move_dir, p) * p
        move_norm = np.linalg.norm(move_dir)
        if move_norm > 1e-12:
            move_dir /= move_norm
        
        # Move point i
        new_pts[i] = pts[i] + lr * move_dir
        new_pts[i] /= np.linalg.norm(new_pts[i])
    
    return new_pts


def _gradient_step(pts, learning_rate, n_samples=5000):
    """
    Gradient-based step to minimize covering radius.
    For each point, move it toward the center of the largest empty cap
    in its vicinity.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Sample points on the sphere
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find the nearest configuration point
    dots = samples @ pts.T  # (n_samples, n)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Use soft-min for smoothness
    beta = 30.0
    exp_neg = np.exp(-beta * dots)
    exp_neg_sum = exp_neg.sum(axis=1, keepdims=True)
    weights = exp_neg / exp_neg_sum  # (n_samples, n)
    
    # For each point j, compute the weighted centroid of samples assigned to it
    weight_sums = weights.sum(axis=0)  # (n,)
    weight_sums[weight_sums < 1e-12] = 1.0
    centroids = (weights.T @ samples) / weight_sums[:, np.newaxis]  # (n, 3)
    
    # Move each point toward its centroid, projected onto the tangent space
    new_pts = np.zeros_like(pts)
    for j in range(n):
        p = pts[j]
        c = centroids[j]
        # Tangent direction
        tangent = c - np.dot(c, p) * p
        norm_tangent = np.linalg.norm(tangent)
        if norm_tangent > 1e-12:
            tangent /= norm_tangent
        new_pts[j] = p + learning_rate * tangent
        new_pts[j] /= np.linalg.norm(new_pts[j])
    
    return new_pts


def _local_search(pts, max_iters=100, tol=1e-6):
    """
    Local search: iteratively adjust each point to minimize its local covering radius.
    For each point, find the optimal position given all other points fixed.
    """
    pts = _normalize(pts)
    n = len(pts)
    best_radius = _covering_radius(pts)
    
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            # Fix all points except i, find best position for point i
            # The optimal position is the center of the largest empty cap
            # in the region where point i is the nearest.
            
            # Sample the tangent plane around pts[i]
            p = pts[i]
            if abs(p[0]) < 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            else:
                ref = np.array([0.0, 1.0, 0.0])
            u = ref - np.dot(ref, p) * p
            u /= np.linalg.norm(u)
            v_basis = np.cross(p, u)
            v_basis /= np.linalg.norm(v_basis)
            
            # Try several candidate positions
            best_local = None
            best_local_val = np.inf
            
            for theta in np.linspace(0, 2 * np.pi, 12):
                for radius in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
                    if radius == 0.0:
                        candidate = p.copy()
                    else:
                        move = radius * (np.cos(theta) * u + np.sin(theta) * v_basis)
                        candidate = p + move
                        candidate /= np.linalg.norm(candidate)
                    
                    # Compute local covering radius contribution
                    # Distance from candidate to all other points
                    dots = np.clip(np.dot(pts, candidate), -1.0, 1.0)
                    dots[i] = 1.0  # distance to itself is 0
                    min_dist = np.arccos(dots).min()
                    
                    if min_dist < best_local_val:
                        best_local_val = min_dist
                        best_local = candidate
            
            if best_local is not None:
                old_pts = pts.copy()
                pts[i] = best_local
                new_radius = _covering_radius(pts)
                if new_radius < best_radius - tol:
                    best_radius = new_radius
                    improved = True
                else:
                    pts = old_pts
        
        if not improved:
            break
    
    return _normalize(pts), best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Multiple random initializations
    best_pts = None
    best_radius = np.inf
    
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            # Fibonacci spiral initialization
            pts = _fibonacci_sphere(N)
        else:
            # Random initialization with some structure
            rng = np.random.default_rng(restart * 12345)
            pts = rng.standard_normal((N, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        pts = _normalize(pts)
        current_radius = _covering_radius(pts)
        
        # Phase 1: Gradient-based refinement
        lr = 0.03
        n_samples = 5000
        stagnant = 0
        
        while time.time() < deadline - 50.0:
            new_pts = _gradient_step(pts, lr, n_samples)
            new_radius = _covering_radius(new_pts)
            
            if new_radius < current_radius - 1e-7:
                current_radius = new_radius
                pts = new_pts
                stagnant = 0
                lr *= 0.995
            else:
                stagnant += 1
                lr *= 0.9
                if stagnant > 20 or lr < 0.001:
                    break
        
        # Phase 2: Repulsive force refinement
        lr = 0.02
        stagnant = 0
        while time.time() < deadline - 30.0:
            new_pts = _repulsive_force_step(pts, lr)
            new_radius = _covering_radius(new_pts)
            
            if new_radius < current_radius - 1e-7:
                current_radius = new_radius
                pts = new_pts
                stagnant = 0
                lr *= 0.99
            else:
                stagnant += 1
                lr *= 0.9
                if stagnant > 15 or lr < 0.0005:
                    break
        
        # Phase 3: Local search
        if time.time() < deadline - 15.0:
            pts, current_radius = _local_search(pts, max_iters=20, tol=1e-7)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
    
    # Final verification
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
