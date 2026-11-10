<<<<<<< SEARCH
"""Rewritten program: force-based repulsion + local search for 50-point sphere covering."""
N = 50
import time
import numpy as np
from scipy.spatial import ConvexHull


def _cover_radius(pts):
    """Compute covering radius (in radians) as max circumradius over hull facets."""
    if len(pts) < 3:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_angle = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

        # Spherical circumcenter: normalize the sum of vertices weighted by sin of opposite angles
        # sin(angle at a) = |b-c| / (2*|b||c|) but since unit, = |b-c|/2
        # Actually for unit vectors, the circumcenter direction is:
        # (a * sin(A) + b * sin(B) + c * sin(C)) where A,B,C are angles opposite a,b,c
        # sin(A) = |b x c| / (|b||c|) = |b x c| for unit vectors
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)

        w_a = np.linalg.norm(cross_bc)
        w_b = np.linalg.norm(cross_ca)
        w_c = np.linalg.norm(cross_ab)

        total = w_a + w_b + w_c
        if total < 1e-15:
            continue

        circum = a * w_a + b * w_b + c * w_c
        cnorm = np.linalg.norm(circum)
        if cnorm < 1e-15:
            continue
        circum /= cnorm

        # The circumcenter should be on the same side as the facet normal (outward)
        # Check: dot with facet normal should be positive
        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm > 1e-15:
            facet_normal /= fnorm
            if np.dot(circum, facet_normal) < 0:
                circum = -circum

        # Angular distance from circumcenter to each vertex
        dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
        min_dot = np.clip(min(dots), -1.0, 1.0)
        angle = np.arccos(min_dot)
        if angle > max_angle:
            max_angle = angle

    return max_angle
=======
"""Optimized 50-point sphere covering using L-BFGS-B with Tammes proxy + covering radius refinement."""
N = 50
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _cover_radius(pts):
    """Compute covering radius (in radians) as max circumradius over hull facets."""
    if len(pts) < 3:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_angle = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]

        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)

        w_a = np.linalg.norm(cross_bc)
        w_b = np.linalg.norm(cross_ca)
        w_c = np.linalg.norm(cross_ab)

        total = w_a + w_b + w_c
        if total < 1e-15:
            continue

        circum = a * w_a + b * w_b + c * w_c
        cnorm = np.linalg.norm(circum)
        if cnorm < 1e-15:
            continue
        circum /= cnorm

        facet_normal = np.cross(b - a, c - a)
        fnorm = np.linalg.norm(facet_normal)
        if fnorm > 1e-15:
            facet_normal /= fnorm
            if np.dot(circum, facet_normal) < 0:
                circum = -circum

        dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
        min_dot = np.clip(min(dots), -1.0, 1.0)
        angle = np.arccos(min_dot)
        if angle > max_angle:
            max_angle = angle

    return max_angle


def _tammes_objective(flat_pts):
    """
    Objective for L-BFGS-B: minimize negative of minimum squared Euclidean distance.
    This is a Tammes problem proxy that correlates with good covering.
    """
    pts = flat_pts.reshape(N, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms

    # Compute all pairwise squared distances
    # d2_ij = 2 - 2*dot(p_i, p_j)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 2.0)  # Set diagonal to 2 so d2 = 0
    d2 = 2.0 - dots
    np.fill_diagonal(d2, 1e10)  # Exclude self-pairs

    min_d2 = np.min(d2)
    # We want to maximize min_d2, so minimize -min_d2
    return -min_d2


def _tammes_gradient(flat_pts):
    """
    Gradient of the Tammes objective with respect to flat coordinates.
    """
    pts = flat_pts.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms

    dots = pts @ pts.T
    d2 = 2.0 - dots
    np.fill_diagonal(d2, 1e10)

    # Find the minimum distance pair
    min_idx = np.unravel_index(np.argmin(d2), d2.shape)
    i, j = min_idx

    # Gradient: d(-d2_ij)/d(p_k) for the pair achieving min
    # d2_ij = 2 - 2*p_i·p_j
    # d(d2_ij)/dp_i = -2*p_j, d(d2_ij)/dp_j = -2*p_i
    # So d(-d2_ij)/dp_i = 2*p_j, d(-d2_ij)/dp_j = 2*p_i
    
    grad = np.zeros((N, 3))
    grad[i] = 2.0 * pts[j]
    grad[j] = 2.0 * pts[i]

    # Project gradient onto tangent space (remove radial component)
    for k in range(N):
        grad[k] -= np.dot(grad[k], pts[k]) * pts[k]

    return grad.flatten()


def _lbfgs_tammes_optimize(init_pts, max_iter=200, deadline=None):
    """
    Optimize using L-BFGS-B on the Tammes proxy objective.
    """
    flat_init = init_pts.flatten()

    def obj_and_grad(flat):
        if deadline is not None and time.time() > deadline:
            return np.inf, np.zeros_like(flat)
        f = _tammes_objective(flat)
        g = _tammes_gradient(flat)
        return f, g

    try:
        result = minimize(
            obj_and_grad,
            flat_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        opt_pts = result.x.reshape(N, 3)
        norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        opt_pts = opt_pts / norms
    except Exception:
        opt_pts = init_pts

    return opt_pts


def _refine_covering_radius(pts, max_iters=500, deadline=None):
    """
    Refine the configuration to minimize the actual covering radius using
    local moves targeting the worst facet.
    """
    pts = pts.copy()
    best_radius = _cover_radius(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(42)
    step = 0.01

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        worst_radius = 0.0
        worst_circum = None
        worst_facet_indices = None

        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            cross_bc = np.cross(b, c)
            cross_ca = np.cross(c, a)
            cross_ab = np.cross(a, b)
            w_a = np.linalg.norm(cross_bc)
            w_b = np.linalg.norm(cross_ca)
            w_c = np.linalg.norm(cross_ab)
            total = w_a + w_b + w_c
            if total < 1e-15:
                continue
            circum = a * w_a + b * w_b + c * w_c
            cnorm = np.linalg.norm(circum)
            if cnorm < 1e-15:
                continue
            circum /= cnorm
            facet_normal = np.cross(b - a, c - a)
            fnorm = np.linalg.norm(facet_normal)
            if fnorm > 1e-15:
                facet_normal /= fnorm
                if np.dot(circum, facet_normal) < 0:
                    circum = -circum
            dots = np.array([np.dot(circum, a), np.dot(circum, b), np.dot(circum, c)])
            min_dot = np.clip(min(dots), -1.0, 1.0)
            angle = np.arccos(min_dot)
            if angle > worst_radius:
                worst_radius = angle
                worst_circum = circum
                worst_facet_indices = simplex

        if worst_circum is None:
            break

        # Try moving points that are nearest to the worst circumcenter
        dots_to_circum = pts @ worst_circum
        # Sort by proximity to circumcenter (largest dot = closest)
        sorted_indices = np.argsort(-dots_to_circum)

        improved = False
        for idx in sorted_indices[:5]:  # Try top 5 closest points
            if deadline is not None and time.time() > deadline:
                break

            # Generate random tangent direction
            random_dir = rng.randn(3)
            tangent = random_dir - np.dot(random_dir, pts[idx]) * pts[idx]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm

            # Try moving toward circumcenter
            to_circum = worst_circum - pts[idx]
            to_circum_tangent = to_circum - np.dot(to_circum, pts[idx]) * pts[idx]
            tc_norm = np.linalg.norm(to_circum_tangent)
            if tc_norm > 1e-10:
                to_circum_tangent /= tc_norm

                new_pts = pts.copy()
                new_pts[idx] = pts[idx] + step * to_circum_tangent
                new_pts[idx] /= np.linalg.norm(new_pts[idx])

                new_radius = _cover_radius(new_pts)
                if new_radius < best_radius - 1e-12:
                    pts = new_pts
                    best_radius = new_radius
                    best_pts = pts.copy()
                    improved = True
                    continue

            # Try random tangent direction
            new_pts = pts.copy()
            new_pts[idx] = pts[idx] + step * tangent
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            new_radius = _cover_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True
                continue

            # Try opposite
            new_pts[idx] = pts[idx] - step * tangent
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            new_radius = _cover_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True

        if not improved:
            step *= 0.9
            if step < 1e-7:
                break

    return best_pts, best_radius
>>>>>>> REPLACE

