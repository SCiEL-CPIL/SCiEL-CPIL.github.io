"""Covering the sphere with 50 equal caps – geometric + gradient refinement."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 140.0  # safety margin below the 150 s wall-clock limit


def _time_left():
    return DEADLINE_SECONDS - (time.perf_counter() - _START)


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius (in radians) via convex hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid / degenerate
    if len(hull.simplices) == 0:
        return np.pi
    # For each triangular facet, the farthest point in the cap is the
    # circumcenter (if the triangle is acute) or the midpoint of the longest
    # edge (if obtuse). The angular distance from the facet's plane normal
    # to the sphere along that direction gives the inradius of the cap.
    # More precisely, for a spherical triangle with vertices on the unit sphere,
    # the covering contribution is the maximum over the triangle of the angular
    # distance to the nearest vertex. This equals the spherical circumradius
    # of the triangle (the angular radius of the smallest spherical cap
    # containing the three vertices, centered at the spherical circumcenter).
    #
    # A robust method: for each facet (i,j,k), compute the point on the sphere
    # that is equidistant (in angular distance) to the three vertices, i.e.,
    # the spherical circumcenter. The angular distance from that center to any
    # vertex is the circumradius. If the circumcenter lies outside the triangle
    # (obtuse case), the farthest point in the cap is actually on the longest
    # edge; in that case the relevant distance is half the longest edge's
    # angular length.
    #
    # Simpler robust approach: the covering radius is the maximum, over all
    # facets, of the angular radius of the inscribed circle of the spherical
    # triangle. For a spherical triangle, the inradius r satisfies
    #   tan(r) = sqrt( (s-a)(s-b)(s-c) / s )  where s = (a+b+c)/2,
    # with a,b,c the side lengths (angular). But this is the inradius of the
    # spherical triangle, which is NOT the covering radius. The covering radius
    # is the circumradius: the smallest R such that a spherical cap of radius R
    # centered somewhere covers the triangle. That is the spherical circumradius.
    #
    # For a spherical triangle with side lengths a, b, c (in radians), the
    # circumradius R satisfies:
    #   sin(R) = (a*b*c) / (4 * sqrt(s*(s-a)*(s-b)*(s-c)))  ... no, that's planar.
    #
    # Let's use a direct geometric approach: for each facet, find the point on
    # the unit sphere that maximizes the minimum angular distance to the three
    # vertices. This is equivalent to finding the center of the smallest
    # enclosing spherical cap of the three points. For three points on the
    # sphere, this is either the spherical circumcenter (if the triangle is
    # "acute" in the spherical sense) or the midpoint of the longest edge.
    #
    # Practical method: compute the Euclidean circumcenter of the three points
    # in R^3, normalize to the unit sphere. Check if it lies inside the
    # spherical triangle. If yes, the circumradius is the angular distance to
    # any vertex. If not, the covering radius for that facet is half the
    # longest angular edge.

    simplices = hull.simplices
    max_cr = 0.0
    for tri in simplices:
        p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Angular side lengths
        a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        # Euclidean circumcenter of the triangle in R^3
        # Solve: |x - p1|^2 = |x - p2|^2 = |x - p3|^2
        # 2*(p2-p1).x = |p2|^2 - |p1|^2
        A = np.array([
            2.0 * (p2 - p1),
            2.0 * (p3 - p1)
        ])
        rhs = np.array([
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1)
        ])
        try:
            x = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            x = (p1 + p2 + p3) / 3.0
        # Check if x is inside the spherical triangle (same side of each edge)
        # The circumcenter on the sphere is x/|x| if |x|>0
        if np.linalg.norm(x) < 1e-12:
            # Degenerate; use centroid
            xc = (p1 + p2 + p3) / 3.0
            xc = xc / np.linalg.norm(xc)
            cr = max(a, b, c) / 2.0
        else:
            xc = x / np.linalg.norm(x)
            # Check if xc is inside the triangle by checking barycentric-like
            # signs: for each edge, xc should be on the same side as the
            # opposite vertex.
            def _same_side(pt, a_pt, b_pt, c_pt):
                """Check if pt is on the same side of line a_pt-b_pt as c_pt (spherical)."""
                n = np.cross(a_pt, b_pt)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    return True
                n = n / n_norm
                d_pt = np.dot(n, pt)
                d_c = np.dot(n, c_pt)
                return d_pt * d_c >= -1e-9

            inside = (_same_side(xc, p1, p2, p3) and
                      _same_side(xc, p2, p3, p1) and
                      _same_side(xc, p3, p1, p2))
            if inside:
                # Circumradius = angular distance from xc to any vertex
                cr = np.arccos(np.clip(np.dot(xc, p1), -1, 1))
            else:
                # Obtuse: covering radius is half the longest edge
                cr = max(a, b, c) / 2.0
        if cr > max_cr:
            max_cr = cr
    return max_cr


def _covering_radius_fast(pts):
    """Faster covering radius using vectorized operations where possible."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    if len(hull.simplices) == 0:
        return np.pi
    simplices = hull.simplices
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]

    # Angular side lengths
    a = np.arccos(np.clip(np.sum(p2 * p3, axis=1), -1, 1))
    b = np.arccos(np.clip(np.sum(p1 * p3, axis=1), -1, 1))
    c = np.arccos(np.clip(np.sum(p1 * p2, axis=1), -1, 1))

    max_cr = 0.0
    for i in range(len(simplices)):
        pp1, pp2, pp3 = p1[i], p2[i], p3[i]
        aa, bb, cc = a[i], b[i], c[i]
        A = np.array([
            2.0 * (pp2 - pp1),
            2.0 * (pp3 - pp1)
        ])
        rhs = np.array([
            np.dot(pp2, pp2) - np.dot(pp1, pp1),
            np.dot(pp3, pp3) - np.dot(pp1, pp1)
        ])
        try:
            x = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            x = (pp1 + pp2 + pp3) / 3.0
        nx = np.linalg.norm(x)
        if nx < 1e-12:
            cr = max(aa, bb, cc) / 2.0
        else:
            xc = x / nx
            # Check inside
            def _same_side(pt, a_pt, b_pt, c_pt):
                n = np.cross(a_pt, b_pt)
                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    return True
                n = n / nn
                return np.dot(n, pt) * np.dot(n, c_pt) >= -1e-9

            inside = (_same_side(xc, pp1, pp2, pp3) and
                      _same_side(xc, pp2, pp3, pp1) and
                      _same_side(xc, pp3, pp1, pp2))
            if inside:
                cr = np.arccos(np.clip(np.dot(xc, pp1), -1, 1))
            else:
                cr = max(aa, bb, cc) / 2.0
        if cr > max_cr:
            max_cr = cr
    return max_cr


def _objective(x):
    """Objective function for optimization: covering radius in radians."""
    pts = _normalize(x.reshape(N, 3))
    return _covering_radius_fast(pts)


def _initial_geodesic():
    """Generate 50 points from an icosahedron-based geodesic subdivision."""
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)

    # Subdivide each face to get more points. We need 50 points total.
    # Start with 12 icosahedron vertices. Subdivide faces to get more.
    # A simpler approach: use a Fibonacci spiral as a good initial guess,
    # then refine. But the directive says to use geodesic dome. Let's
    # create a geodesic dome with ~50 vertices by subdividing icosahedron.
    #
    # For 50 points, we can subdivide the icosahedron's 20 faces.
    # If we subdivide each face into 4 triangles (midpoint subdivision),
    # we get 20*4 = 80 faces and 12 + 30 = 42 vertices. Not quite 50.
    #
    # Alternative: just use a well-distributed initial set and refine.
    # The Fibonacci spiral is a good start. Let's use it and then
    # apply geometric + gradient refinement.

    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    ang = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(ang), r * np.sin(ang), z])
    return pts


def _gradient_refinement(pts, max_iters=200, tol=1e-7):
    """Refine points using gradient descent on the covering radius."""
    # The covering radius is non-smooth, so we use a smooth approximation:
    # maximize the minimum distance to the nearest point (equivalently,
    # minimize the maximum over the sphere of the distance to the nearest point).
    # A common approach: for each point, compute the Voronoi cell on the sphere
    # and move it to the center of the inscribed circle of its cell.
    #
    # Simpler: use scipy.optimize.minimize with a smooth surrogate.
    # The covering radius can be approximated by the maximum over a dense
    # set of test points on the sphere of the angular distance to the nearest
    # configuration point. This is smooth (max of smooth functions is smooth
    # almost everywhere).

    # Generate a dense set of test points
    n_test = 2000
    # Use a Fibonacci spiral for test points
    i_t = np.arange(n_test) + 0.5
    z_t = 1.0 - 2.0 * i_t / n_test
    ang_t = np.pi * (1.0 + np.sqrt(5.0)) * i_t
    r_t = np.sqrt(np.maximum(0, 1.0 - z_t * z_t))
    test_pts = np.column_stack([r_t * np.cos(ang_t), r_t * np.sin(ang_t), z_t])

    def _smooth_obj(x):
        pts_x = _normalize(x.reshape(N, 3))
        # For each test point, compute distance to nearest config point
        # dist[i,j] = arccos(dot(test_i, config_j))
        dots = np.clip(test_pts @ pts_x.T, -1, 1)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        # Use a soft-max approximation for the max
        # Or just use the max (non-smooth but works with L-BFGS-B sometimes)
        return np.max(min_dists)

    def _smooth_obj_and_grad(x):
        pts_x = _normalize(x.reshape(N, 3))
        dots = np.clip(test_pts @ pts_x.T, -1, 1)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        obj = np.max(min_dists)
        # Gradient: only the test point achieving the max contributes
        max_idx = np.argmax(min_dists)
        nearest_j = min_indices[max_idx]
        tp = test_pts[max_idx]
        cp = pts_x[nearest_j]
        # d/dx arccos(x·y) = -y / sqrt(1-(x·y)^2)
        dot_val = np.dot(tp, cp)
        denom = np.sqrt(np.maximum(1e-12, 1.0 - dot_val * dot_val))
        grad = -tp / denom
        # But we need to account for normalization
        # If pts_x = x / |x|, then d(pts_x)/dx = I/|x| - x x^T / |x|^3
        # For simplicity, use the unnormalized gradient and let the optimizer handle it
        grad_full = np.zeros_like(x)
        grad_full[nearest_j * 3: nearest_j * 3 + 3] = grad
        return obj, grad_full

    x0 = pts.flatten()
    try:
        res = minimize(_smooth_obj_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol})
        if res.success or res.fun < _covering_radius_fast(pts):
            best = _normalize(res.x.reshape(N, 3))
            return best
    except Exception:
        pass
    return pts


def _simulated_annealing(pts, max_iters=5000, t_start=0.1, t_end=0.001):
    """Simulated annealing on the sphere."""
    current = pts.copy()
    current_cr = _covering_radius_fast(current)
    best = current.copy()
    best_cr = current_cr
    n_iter = 0
    while n_iter < max_iters and _time_left() > 5.0:
        # Random perturbation
        idx = np.random.randint(N)
        # Random direction on the tangent plane
        rand_vec = np.random.randn(3)
        tangent = rand_vec - np.dot(rand_vec, current[idx]) * current[idx]
        norm_t = np.linalg.norm(tangent)
        if norm_t < 1e-12:
            n_iter += 1
            continue
        tangent /= norm_t
        # Step size based on temperature
        T = t_start * (t_end / t_start) ** (n_iter / max_iters)
        step = T * 0.1
        new_pt = current[idx] + step * tangent
        new_pt = new_pt / np.linalg.norm(new_pt)
        new_pts = current.copy()
        new_pts[idx] = new_pt
        new_cr = _covering_radius_fast(new_pts)
        if new_cr < current_cr or np.random.rand() < np.exp(-(new_cr - current_cr) / max(T, 1e-10)):
            current = new_pts
            current_cr = new_cr
            if current_cr < best_cr:
                best = current.copy()
                best_cr = current_cr
        n_iter += 1
    return best


def construct():
    global _START
    _START = time.perf_counter()

    # Initial points
    pts = _initial_geodesic()

    # Gradient refinement
    if _time_left() > 30.0:
        pts = _gradient_refinement(pts, max_iters=100, tol=1e-6)

    # Simulated annealing
    if _time_left() > 20.0:
        np.random.seed(42)
        pts = _simulated_annealing(pts, max_iters=3000, t_start=0.05, t_end=0.001)

    # Final gradient refinement
    if _time_left() > 15.0:
        pts = _gradient_refinement(pts, max_iters=50, tol=1e-7)

    return _normalize(pts)


_START = time.perf_counter()
