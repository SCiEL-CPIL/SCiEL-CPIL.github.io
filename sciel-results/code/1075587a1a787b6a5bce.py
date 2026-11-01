"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin before 120s kill


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull.
    
    The covering radius equals the maximum circumradius of the facets of the
    convex hull (for points on the unit sphere).
    """
    try:
        hull = ConvexHull(pts, qhull_options="QJ")
    except Exception:
        # Fallback: grid-based approximation
        return _covering_radius_grid(pts)
    
    max_r = 0.0
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        # Compute circumcenter of the spherical triangle
        # For a triangle on the unit sphere, the circumcenter is the normalized
        # sum of the vertices if the triangle is small, but more precisely
        # we solve for the point equidistant from all three vertices.
        
        # The circumcenter c satisfies |c - v_i| = R for i=0,1,2
        # On the sphere, this means c.v_i = cos(R) for all i
        # So c is in the span of the face points and c.v_i is constant.
        
        # Solve: find c = a*v0 + b*v1 + c*v2 such that c.v0 = c.v1 = c.v2
        # This is equivalent to finding the point in the plane of the triangle
        # (extended) that is equidistant from all three vertices.
        
        # Actually, for a spherical triangle, the circumcenter is the intersection
        # of the perpendicular bisectors. In 3D, we can find it by:
        # The circumcenter lies along the normal to the face plane through the
        # circumcenter of the planar triangle formed by the three points.
        
        # Simpler approach: the circumcenter of the spherical triangle is the
        # normalized vector in the direction of the solution to the linear system.
        
        # Let's use the fact that for three points a, b, c on the sphere,
        # the circumcenter direction is given by:
        # (a x b) x (b x c) ... no, let's use the formula:
        # circumcenter = normalize(a * sin(2A) + b * sin(2B) + c * sin(2C))
        # where A, B, C are the angles opposite to a, b, c.
        
        # Even simpler: solve the linear system
        # c . a = c . b = c . c2 = cos(R)
        # c = alpha*a + beta*b + gamma*c2
        # This gives a 3x3 system.
        
        a, b, c2 = face_pts[0], face_pts[1], face_pts[2]
        
        # The circumcenter c satisfies:
        # c . a = c . b  => c . (a - b) = 0
        # c . b = c . c2 => c . (b - c2) = 0
        # c is in the span of a, b, c2
        
        # So c = x*a + y*b + z*c2
        # And c . (a-b) = 0, c . (b-c2) = 0
        # Plus we want to minimize |c| or just find the direction.
        
        # Actually, the circumcenter of a spherical triangle is the point on the
        # sphere equidistant from all three vertices. It lies along the line
        # through the origin and the circumcenter of the planar triangle.
        
        # The planar circumcenter of triangle (a, b, c) in 3D:
        # It's the point p minimizing max(|p-a|^2, |p-b|^2, |p-c|^2)
        # subject to p being in the plane of a, b, c.
        
        # The plane has normal n = (b-a) x (c-a)
        # The circumcenter in the plane: solve (p-a).(b-a) = |b-a|^2/2, etc.
        
        # Let's just compute it directly:
        # p = a + s*(b-a) + t*(c-a)
        # (p-a).(b-a) = |b-a|^2/2  =>  s*|b-a|^2 + t*(c-a).(b-a) = |b-a|^2/2
        # (p-a).(c-a) = |c-a|^2/2  =>  s*(b-a).(c-a) + t*|c-a|^2 = |c-a|^2/2
        
        ab = b - a
        ac = c2 - a
        ab_sq = np.dot(ab, ab)
        ac_sq = np.dot(ac, ac)
        ab_ac = np.dot(ab, ac)
        
        # Solve:
        # s * ab_sq + t * ab_ac = ab_sq / 2
        # s * ab_ac + t * ac_sq = ac_sq / 2
        
        det = ab_sq * ac_sq - ab_ac * ab_ac
        if abs(det) < 1e-14:
            # Degenerate triangle, skip
            continue
        
        s = (ab_sq / 2.0 * ac_sq - ab_ac * ac_sq / 2.0) / det
        t = (ab_sq / 2.0 * ab_ac - ac_sq / 2.0 * ab_ac) / det
        
        p = a + s * ab + t * ac
        
        # The spherical circumcenter is p normalized (if p is not zero)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-14:
            continue
        
        circumcenter = p / p_norm
        
        # The angular distance from the circumcenter to any vertex is the circumradius
        dot_val = np.dot(circumcenter, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        r = np.arccos(dot_val)
        
        if r > max_r:
            max_r = r
    
    if max_r == 0.0:
        # Fallback if no facets found
        return _covering_radius_grid(pts)
    
    return max_r


def _covering_radius_grid(pts, n_grid=40):
    """Grid-based approximation of covering radius."""
    # Create a grid of points on the sphere
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, n_grid * 2)
    THETA, PHI = np.meshgrid(theta, phi)
    grid_pts = np.column_stack([
        np.sin(THETA) * np.cos(PHI),
        np.sin(THETA) * np.sin(PHI),
        np.cos(THETA)
    ]).reshape(-1, 3)
    
    # For each grid point, find the minimum angular distance to any pts
    # Use vectorized computation in chunks to save memory
    max_min_dist = 0.0
    chunk_size = 1000
    for start in range(0, len(grid_pts), chunk_size):
        chunk = grid_pts[start:start + chunk_size]
        # Compute dot products: chunk (m,3) @ pts.T (3,n) -> (m,n)
        dots = chunk @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)  # max dot = min angle
        min_angles = np.arccos(min_dots)
        max_min_dist = max(max_min_dist, np.max(min_angles))
    
    return max_min_dist


def _gradient_step(pts, step_size=0.01):
    """Perform a gradient-like step to reduce the covering radius.
    
    Find the worst facet, compute its circumcenter, and move the three vertices
    of that facet toward the circumcenter (which reduces the circumradius).
    """
    try:
        hull = ConvexHull(pts, qhull_options="QJ")
    except Exception:
        return pts.copy()
    
    # Find the facet with the largest circumradius
    max_r = 0.0
    worst_simplex = None
    worst_circumcenter = None
    
    for simplex in hull.simplices:
        face_pts = pts[simplex]
        a, b, c2 = face_pts[0], face_pts[1], face_pts[2]
        
        ab = b - a
        ac = c2 - a
        ab_sq = np.dot(ab, ab)
        ac_sq = np.dot(ac, ac)
        ab_ac = np.dot(ab, ac)
        
        det = ab_sq * ac_sq - ab_ac * ab_ac
        if abs(det) < 1e-14:
            continue
        
        s = (ab_sq / 2.0 * ac_sq - ab_ac * ac_sq / 2.0) / det
        t = (ab_sq / 2.0 * ab_ac - ac_sq / 2.0 * ab_ac) / det
        
        p = a + s * ab + t * ac
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-14:
            continue
        
        circumcenter = p / p_norm
        dot_val = np.clip(np.dot(circumcenter, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        
        if r > max_r:
            max_r = r
            worst_simplex = simplex
            worst_circumcenter = circumcenter
    
    if worst_simplex is None:
        return pts.copy()
    
    new_pts = pts.copy()
    for idx in worst_simplex:
        v = new_pts[idx]
        # Move toward the circumcenter
        new_pts[idx] = v + step_size * (worst_circumcenter - v)
        # Renormalize
        norm = np.linalg.norm(new_pts[idx])
        if norm > 1e-14:
            new_pts[idx] /= norm
    
    return new_pts


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed_offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on the sphere."""
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    global _START_TIME
    start_time = _START_TIME
    deadline = start_time + _TIME_LIMIT
    
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple starting configurations
    starts = []
    # Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    # Rotated Fibonacci spirals
    for angle in [0.5, 1.0, 1.5, 2.0]:
        pts = _fibonacci_sphere(N, seed_offset=angle)
        starts.append(pts)
    # Random starts
    for _ in range(5):
        starts.append(_random_sphere(N, rng))
    
    for start_pts in starts:
        if time.time() > deadline - 5.0:
            break
        
        current_pts = start_pts.copy()
        current_radius = _covering_radius(current_pts)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
        
        # Iterative refinement
        step_size = 0.05
        no_improve_count = 0
        max_iter = 500
        
        for iteration in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            new_pts = _gradient_step(current_pts, step_size)
            new_radius = _covering_radius(new_pts)
            
            if new_radius < current_radius - 1e-10:
                current_pts = new_pts
                current_radius = new_radius
                no_improve_count = 0
                # Adaptively decrease step size
                if iteration % 50 == 0 and step_size > 0.001:
                    step_size *= 0.8
            else:
                no_improve_count += 1
                if no_improve_count > 20:
                    # Try smaller step
                    step_size *= 0.5
                    if step_size < 0.0005:
                        break
                    no_improve_count = 0
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
    
    # Final polish: use scipy.optimize with a smooth objective
    if best_pts is not None and time.time() < deadline - 10.0:
        # Use a smooth approximation: maximize the minimum of (1 - v_i . y) over y on sphere
        # Equivalently, minimize the maximum circumradius.
        # We can use the support function approach or just do a fine local search.
        
        # Fine local search with smaller steps
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb each point slightly and keep if it improves
            improved = False
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                
                # Random small perturbation
                direction = rng.normal(size=3)
                direction = np.cross(best_pts[i], direction)  # Tangential direction
                d_norm = np.linalg.norm(direction)
                if d_norm < 1e-12:
                    continue
                direction /= d_norm
                
                for step in [0.001, 0.005, 0.01]:
                    new_pts = best_pts.copy()
                    new_pts[i] = best_pts[i] + step * direction
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    
                    new_r = _covering_radius(new_pts)
                    if new_r < best_radius - 1e-12:
                        best_pts = new_pts
                        best_radius = new_r
                        improved = True
                        break
            
            if not improved:
                break
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                # Separate them
                best_pts[j] = best_pts[j] + 0.01 * rng.normal(size=3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
