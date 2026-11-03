"""Heilbronn convex regions n=14 – support-function parametrisation with local refinement."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14
TIME_LIMIT = 115.0  # safety margin below 120 s


def _hull_area(pts):
    """Convex hull area of a (m,2) array; 0.0 on failure."""
    if len(pts) < 3:
        return 0.0
    try:
        h = ConvexHull(pts)
        return float(h.volume)
    except Exception:
        return 0.0


def _min_triangle_area(pts):
    """Minimum triangle area over all C(n,3) triples; inf if n<3."""
    n = len(pts)
    if n < 3:
        return float('inf')
    # Vectorised pairwise cross products
    # For speed with n=14 (364 triples) a simple loop is fine
    min_a = float('inf')
    for i in range(n - 2):
        xi, yi = pts[i]
        for j in range(i + 1, n - 1):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                a = 0.5 * abs((xj - xi) * (yk - yi) - (yk - yi) * (xj - xi))
                # cross product: (xj-xi)*(yk-yi) - (xk-xi)*(yj-yi)
                a = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                if a < min_a:
                    min_a = a
    return min_a


def _score(pts):
    """min triangle area / hull area; 0.0 on degeneracy."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ha = _hull_area(pts)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(pts)
    if not np.isfinite(ma):
        return 0.0
    return ma / ha


def _support_to_points(coeffs):
    """
    Convert support-function coefficients to N points on the boundary.
    
    coeffs layout: [a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    h(theta) = a0 + sum_{k=1}^{5} [a_k cos(k theta) + b_k sin(k theta)]
    
    We sample N equally-spaced angles and use the outward normal to get
    the boundary point: p(theta) = h(theta) * (cos theta, sin theta)
    + h'(theta) * (-sin theta, cos theta)
    
    For strict convexity we need h(theta) > 0 and h(theta) + h''(theta) > 0.
    We enforce this by construction: a0 is large enough and harmonics small.
    """
    a0 = coeffs[0]
    a1, b1 = coeffs[1], coeffs[2]
    a2, b2 = coeffs[3], coeffs[4]
    a3, b3 = coeffs[5], coeffs[6]
    a4, b4 = coeffs[7], coeffs[8]
    a5, b5 = coeffs[9], coeffs[10]

    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ct = np.cos(thetas)
    st = np.sin(thetas)

    # h(theta)
    h = a0
    h += a1 * ct + b1 * st
    h += a2 * np.cos(2 * thetas) + b2 * np.sin(2 * thetas)
    h += a3 * np.cos(3 * thetas) + b3 * np.sin(3 * thetas)
    h += a4 * np.cos(4 * thetas) + b4 * np.sin(4 * thetas)
    h += a5 * np.cos(5 * thetas) + b5 * np.sin(5 * thetas)

    # h'(theta)
    hp = 0.0
    hp += -a1 * st + b1 * ct
    hp += -2 * a2 * np.sin(2 * thetas) + 2 * b2 * np.cos(2 * thetas)
    hp += -3 * a3 * np.sin(3 * thetas) + 3 * b3 * np.cos(3 * thetas)
    hp += -4 * a4 * np.sin(4 * thetas) + 4 * b4 * np.cos(4 * thetas)
    hp += -5 * a5 * np.sin(5 * thetas) + 5 * b5 * np.cos(5 * thetas)

    # p = h * (cos, sin) + h' * (-sin, cos)
    px = h * ct - hp * st
    py = h * st + hp * ct

    pts = np.column_stack([px, py])
    return pts


def _neg_score_from_coeffs(coeffs):
    pts = _support_to_points(coeffs)
    s = _score(pts)
    return -s


def _neg_score_from_pts_flat(flat):
    pts = flat.reshape(N, 2)
    s = _score(pts)
    return -s


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_score = 0.0
    best_pts = None

    # --- Phase 1: Differential Evolution on support-function coefficients ---
    # Bounds: a0 in [0.5, 2.0] (must be positive and dominant)
    # harmonics in [-0.3, 0.3] to keep convexity
    bounds = [(0.5, 2.0)] + [(-0.3, 0.3)] * 10
    x0 = [1.0] + [0.0] * 10  # circle of radius 1

    def de_obj(coeffs):
        if time.time() > deadline - 5:
            return 1e10
        return _neg_score_from_coeffs(coeffs)

    try:
        de_result = differential_evolution(
            de_obj, bounds,
            seed=42, maxiter=30, tol=1e-6,
            popsize=15, init='sobol',
            polish=False,
            callback=lambda xk, conv: None
        )
        if time.time() < deadline - 10:
            s = -de_result.fun
            if s > best_score:
                best_score = s
                best_pts = _support_to_points(de_result.x)
    except Exception:
        pass

    # --- Phase 2: Multi-start local refinement on raw point coordinates ---
    # Use several starting points: DE result, regular polygon, random perturbations
    starts = []
    if best_pts is not None:
        starts.append(best_pts.copy())
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    starts.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    # A few random starts
    rng = np.random.RandomState(0)
    for _ in range(4):
        ang = np.sort(rng.uniform(0, 2 * np.pi, N))
        r = 1.0 + 0.2 * rng.randn(N)
        r = np.clip(r, 0.5, 1.5)
        starts.append(np.column_stack([r * np.cos(ang), r * np.sin(ang)]))

    for start in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                _neg_score_from_pts_flat,
                start.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass

    # --- Phase 3: Powell refinement on best point set ---
    if best_pts is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                _neg_score_from_pts_flat,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = np.array([[0.0, 0.0]] * N)

    return best_pts


# EVOLVE-BLOCK-START
# (construct is defined above; kept here for interface clarity)
# EVOLVE-BLOCK-END
