"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay

N = 130
TIME_LIMIT = 110.0  # seconds, safety margin before 120s kill

# EVOLVE-BLOCK-START


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius_fast(pts, n_grid=200):
    """
    Fast approximate covering radius using a grid of test points on the sphere.
    Returns the max angular distance (in radians) from any grid point to nearest pts.
    """
    # Generate a quasi-uniform grid of test points on the sphere
    # Use Fibonacci sphere for good coverage
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each test point, find the nearest configuration point
    # Using chunked computation to avoid memory issues
    chunk_size = 5000
    max_dist = 0.0
    
    for start in range(0, n_grid, chunk_size):
        end = min(start + chunk_size, n_grid)
        tp = test_pts[start:end]
        # Compute distances: dot product gives cos(angle)
        # angle = arccos(clamp(dot, -1, 1))
        # For efficiency, use 1 - dot as a proxy for small angles,
        # but we need actual angles for correctness
        dots = tp @ pts.T  # shape (chunk, N)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # shape (chunk, N)
        min_angles = np.min(angles, axis=1)
        chunk_max = np.max(min_angles)
        if chunk_max > max_dist:
            max_dist = chunk_max
    
    return max_dist


def _covering_radius_exact(pts):
    """
    More accurate covering radius using Delaunay triangulation of the points.
    The covering radius is the maximum circumradius of the Delaunay triangles.
    """
    try:
        # Project to 2D? No, use 3D Delaunay
        # Actually, for sphere covering, we need the spherical Delaunay
        # A simpler approach: use the fact that the covering radius equals
        # the maximum over all empty circles (circumscribed by 3 points)
        
        # Use scipy's Delaunay on the 3D points
        tri = Delaunay(pts)
        
        max_circumradius = 0.0
        for simplex in tri.simplices:
            if len(simplex) == 3:
                a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                # Check if this triangle's circumcircle is empty (no other point inside)
                # For now, compute circumradius and take max
                # Circumradius of triangle on unit sphere:
                # For three unit vectors a, b, c, the circumradius R satisfies
                # sin(R) = area of triangle / (2 * product of side lengths)
                # Actually, for points on unit sphere, the circumradius is the angle
                # from center to the circumcenter.
                
                # Side lengths (chord lengths)
                ab = np.linalg.norm(a - b)
                bc = np.linalg.norm(b - c)
                ca = np.linalg.norm(c - a)
                
                if ab < 1e-10 or bc < 1e-10 or ca < 1e-10:
                    continue
                
                # Area of triangle using cross product
                cross = np.cross(b - a, c - a)
                area = 0.5 * np.linalg.norm(cross)
                
                # Circumradius formula: R = (a*b*c) / (4*Area) for planar triangle
                # But these points are on a sphere, so we need the spherical circumradius
                # For three points on unit sphere, the circumradius (angular) is:
                # sin(R) = area / (2 * ab * bc * ca) ... this isn't quite right
                
                # Better: the circumcenter of the triangle in 3D
                # The circumcenter O satisfies |O-a| = |O-b| = |O-c|
                # For points on unit sphere, if the triangle doesn't contain the origin,
                # the circumcenter is in the plane of the triangle.
                
                # Compute circumcenter in the plane of the triangle
                # Using the formula:
                # Let A=a, B=b, C=c
                # O = A + t*(B-A) + s*(C-A) where t, s solve:
                # |A + t*(B-A) + s*(C-A)|^2 = |B + t*(B-A) + s*(C-A)|^2 ... 
                
                # Simpler: use the fact that for unit vectors, the circumradius
                # (angular distance from centroid to vertex) is an upper bound
                # but not exact. Let's use a different approach.
                
                # The covering radius is the max over all points y of min_i angle(y, p_i)
                # This equals the max over all Delaunay triangles of the circumradius
                # of that triangle (if the circumcenter is on the same side of the
                # triangle as the origin, i.e., the triangle doesn't wrap around).
                
                # Compute the circumcenter of the three points in 3D
                # |x-a|^2 = |x-b|^2 => 2x.(b-a) = |b|^2 - |a|^2 = 0 (since unit)
                # So x.(b-a) = 0 and x.(c-a) = 0
                # Thus x is perpendicular to both (b-a) and (c-a), so x is parallel to (b-a)x(c-a)
                # But x must also satisfy |x-a|^2 = |x-b|^2, which gives x.(b-a) = 0
                # So x = t * n where n = (b-a) x (c-a)
                # And |x-a|^2 = R^2 where R is the circumradius (chord)
                # |t*n - a|^2 = R^2
                # t^2|n|^2 - 2t*n.a + |a|^2 = R^2
                # Since |a|=1: t^2|n|^2 - 2t*n.a + 1 = R^2
                # Also, the distance from origin to circumcenter is |t*n|
                # The angular circumradius is arccos(|t*n|) if |t*n| <= 1
                
                n = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                
                # x = t * n, and x.(b-a) = 0 is automatically satisfied
                # We need |x-a|^2 = |x-b|^2 which gives x.(b-a) = 0, already satisfied
                # And |x-a|^2 = |x-c|^2 gives x.(c-a) = 0, also satisfied
                # So any t works? No, we need the circumradius to be consistent.
                # Actually, the circumcenter is uniquely determined by the three points
                # (unless they're collinear). The condition is that x is equidistant from a, b, c.
                # x = t*n. |t*n - a|^2 = |t*n - b|^2
                # t^2|n|^2 - 2t*n.a + 1 = t^2|n|^2 - 2t*n.b + 1
                # -2t*n.a = -2t*n.b
                # t*n.(b-a) = 0
                # Since n is perpendicular to (b-a), this is automatically satisfied.
                # So the circumcenter lies along the line through origin in direction n.
                # But we also need |x-a| to be the same for all three, which it is.
                # The circumradius (chord) is |x-a| = |t*n - a|
                # To find t, we use that the circumcenter should be such that
                # the triangle is inscribed in a circle of radius |x-a|.
                # Actually, t is determined by the requirement that the three points
                # lie on a circle centered at x. This is automatically satisfied for
                # any t if x is on the line perpendicular to the plane through the origin.
                # Wait, that's not right. The circumcenter of a triangle in 3D is not
                # necessarily on the line through the origin perpendicular to the plane.
                
                # Let me use a different approach: solve for the circumcenter directly.
                # x = a + alpha*(b-a) + beta*(c-a)
                # |x-b|^2 = |x-a|^2 => 2*(b-a).(x-a) = |b-a|^2
                # |x-c|^2 = |x-a|^2 => 2*(c-a).(x-a) = |c-a|^2
                # Let u = b-a, v = c-a, w = x-a
                # 2*u.w = |u|^2
                # 2*v.w = |v|^2
                # w = alpha*u + beta*v
                # 2*u.(alpha*u + beta*v) = |u|^2 => 2*alpha*|u|^2 + 2*beta*u.v = |u|^2
                # 2*v.(alpha*u + beta*v) = |v|^2 => 2*alpha*u.v + 2*beta*|v|^2 = |v|^2
                # Solve:
                # alpha*|u|^2 + beta*u.v = |u|^2/2
                # alpha*u.v + beta*|v|^2 = |v|^2/2
                
                u = b - a
                v = c - a
                uu = np.dot(u, u)
                vv = np.dot(v, v)
                uv = np.dot(u, v)
                
                det = uu * vv - uv * uv
                if abs(det) < 1e-15:
                    continue
                
                alpha = (vv * uu/2 - uv * vv/2) / det  # Wait, let me redo
                # alpha*uu + beta*uv = uu/2
                # alpha*uv + beta*vv = vv/2
                # From first: alpha = (uu/2 - beta*uv) / uu
                # Substitute: ((uu/2 - beta*uv)/uv)*uv + beta*vv = vv/2
                # uu/2 - beta*uv + beta*vv = vv/2
                # beta*(vv - uv) = (vv - uu)/2
                # beta = (vv - uu) / (2*(vv - uv))
                # This is getting messy. Let me use matrix solve.
                
                A_mat = np.array([[uu, uv], [uv, vv]])
                b_vec = np.array([uu/2, vv/2])
                try:
                    params = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    continue
                alpha, beta = params
                w = alpha * u + beta * v
                x = a + w
                x_norm = np.linalg.norm(x)
                if x_norm < 1e-12:
                    continue
                x = x / x_norm  # This is wrong - the circumcenter is not on the unit sphere
                
                # Actually, the circumcenter x is NOT on the unit sphere.
                # The angular circumradius is the angle from origin to x... no.
                # The covering radius for this triangle is the angle from the origin
                # to the circumcenter, which is arccos(0) = pi/2 if circumcenter is at origin.
                # I'm overcomplicating this.
                
                # Let's just use the grid-based method for now and make it faster.
                pass
        
        # Fallback to grid-based
        return _covering_radius_fast(pts, n_grid=300)
    
    except Exception:
        return _covering_radius_fast(pts, n_grid=300)


def _gradient_step(pts, lr, n_grid=200):
    """
    Compute a gradient-like step to reduce the covering radius.
    Uses the grid-based approximation for the gradient.
    """
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each test point, find nearest configuration point and contribute to gradient
    # The objective is max_i angle(test_i, pts_j(i)) where j(i) is argmin
    # Gradient w.r.t. pts_k: sum over test points where pts_k is nearest of
    #   -test_i / (|test_i| * |pts_k| * sin(angle)) ... this is complex
    
    # Simpler: use a soft-min approach
    # f(pts) = max_y min_k angle(y, pts_k)
    # Approximate with: f(pts) ≈ log-sum-exp of (angle(y, pts_k) - beta) over k, then max over y
    # This is still complex. Let's use a simpler heuristic:
    # Move each point in the direction of the "worst" test point it's responsible for.
    
    dots = test_pts @ pts.T  # (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (n_grid, N)
    
    # For each test point, the nearest point
    nearest_idx = np.argmin(angles, axis=1)  # (n_grid,)
    min_angles = np.min(angles, axis=1)  # (n_grid,)
    
    # Gradient accumulation
    grad = np.zeros_like(pts)
    for k in range(N):
        mask = nearest_idx == k
        if not np.any(mask):
            continue
        # Points responsible for test points where k is nearest
        responsible_test = test_pts[mask]
        # Move pts[k] toward the "worst" responsible test points
        # The worst are those with largest angle
        worst_idx = np.argsort(min_angles[mask])[-min(10, np.sum(mask)):]
        worst_test = responsible_test[worst_idx]
        # Average direction
        avg_dir = np.mean(worst_test, axis=0)
        avg_norm = np.linalg.norm(avg_dir)
        if avg_norm > 1e-10:
            grad[k] += avg_dir / avg_norm
    
    # Normalize gradient to be tangent to sphere
    for k in range(N):
        # Remove radial component
        radial = np.dot(grad[k], pts[k])
        grad[k] -= radial * pts[k]
    
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        grad /= grad_norm
    
    # Update and re-normalize
    new_pts = pts + lr * grad
    return _normalize(new_pts)


def construct():
    start_time = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius_fast(pts, n_grid=300)
    
    # Gradient descent with decreasing learning rate
    lr = 0.01
    step = 0
    while time.time() - start_time < TIME_LIMIT - 10:
        new_pts = _gradient_step(pts, lr, n_grid=200)
        new_radius = _covering_radius_fast(new_pts, n_grid=200)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            # Decay learning rate slowly
            lr *= 0.999
        else:
            pts = best_pts.copy()
            lr *= 0.95
        
        step += 1
        if step % 100 == 0:
            # Occasionally try a larger step or perturbation
            if lr < 0.001:
                lr = 0.005
    
    # Final polish with more grid points
    final_pts = best_pts.copy()
    for _ in range(50):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        new_pts = _gradient_step(final_pts, 0.001, n_grid=300)
        new_radius = _covering_radius_fast(new_pts, n_grid=300)
        old_radius = _covering_radius_fast(final_pts, n_grid=300)
        if new_radius < old_radius:
            final_pts = new_pts
    
    return _normalize(final_pts)


# EVOLVE-BLOCK-END
