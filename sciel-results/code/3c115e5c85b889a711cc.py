"""
Rewritten approach:
  - Start from a deterministic icosahedral-seed jitter.
  - Use a smooth epigraph surrogate for the covering radius (max facet circumradius),
    computed exactly from the convex hull each evaluation.
  - Minimize with L-BFGS-B on the flattened 390-dim coordinates; renormalize inside the
    objective.
  - Follow with a few direct "worst-facet vertex push" steps that are guaranteed to
    reduce the maximum facet circumradius in small local moves.
"""

N = 130

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def construct():
    t0 = time.time()
    T_LIMIT = 150.0          # kill
    SAFETY = 12.0            # safety margin
    DEADLINE = t0 + T_LIMIT - SAFETY

    def elapsed_ok():
        return time.time() < DEADLINE

    # ---------- deterministic jittered icosahedral seed ----------
    rng = np.random.default_rng(12345)
    pts = _icosahedral_seed(N, rng)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    def compute_covering_radius(P):
        """Exact max facet circumradius in radians via convex hull."""
        try:
            hull = ConvexHull(P)
        except Exception:
            return np.pi
        worst = 0.0
        for tri in hull.simplices:
            a = P[tri[0]]; b = P[tri[1]]; c = P[tri[2]]
            r = _facet_circumradius(a, b, c)
            if r > worst:
                worst = r
        return worst

    def smooth_objective(x):
        if not elapsed_ok():
            return 1e6
        P = x.reshape(N, 3)
        nrm = np.linalg.norm(P, axis=1, keepdims=True)
        nrm = np.where(nrm < 1e-12, 1.0, nrm)
        P = P / nrm

        try:
            hull = ConvexHull(P)
        except Exception:
            return 1e6

        radii = np.empty(len(hull.simplices))
        for i, tri in enumerate(hull.simplices):
            radii[i] = _facet_circumradius(P[tri[0]], P[tri[1]], P[tri[2]])

        # log-sum-exp smoothing of the max, plus a small penalty for any facet near degenerate
        m = np.max(radii)
        if m < 1e-15:
            return 0.0
        beta = 40.0
        softmax = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta
        # mild penalty to keep points distinct
        dots = P @ P.T
        off = np.sum(np.maximum(0.0, 1.0 - np.abs(dots)))  # not ideal but cheap; skip
        return softmax

    # ---------- Phase 1: L-BFGS-B on the smooth objective ----------
    if elapsed_ok():
        x0 = pts.flatten()
        res = minimize(
            smooth_objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
        )
        P1 = res.x.reshape(N, 3)
        nrm = np.linalg.norm(P1, axis=1, keepdims=True)
        nrm = np.where(nrm < 1e-12, 1.0, nrm)
        P1 = P1 / nrm
        r1 = compute_covering_radius(P1)
        if r1 < compute_covering_radius(pts):
            pts = P1

    # ---------- Phase 2: worst-facet vertex push (exact local improvement) ----------
    lr = 0.005
    for it in range(3000):
        if not elapsed_ok():
            break
        r_now = compute_covering_radius(pts)
        if r_now < 0.19705:  # ~11.275 deg, well below best known 11.316
            break
        tri, cc, r_f = _worst_facet(pts)
        if tri is None:
            break

        # move each vertex of the worst triangle slightly toward the circumcenter
        # in the tangent plane (which, on the unit sphere, is just projecting the vector
        # cc - vi onto the plane perpendicular to vi)
        new_pts = pts.copy()
        for idx in tri:
            v = pts[idx]
            diff = cc - v
            # remove component along v to stay on the sphere
            diff_t = diff - np.dot(diff, v) * v
            nrm = np.linalg.norm(diff_t)
            if nrm < 1e-12:
                continue
            step = lr * diff_t / nrm
            new_pts[idx] = v + step
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)

        r_new = compute_covering_radius(new_pts)
        if r_new < r_now:
            pts = new_pts
            lr *= 1.001  # very gentle increase
        else:
            lr *= 0.7
            if lr < 1e-6:
                break

    # ---------- Phase 3: a few more L-BFGS steps with a higher beta ----------
    if elapsed_ok():
        global_beta = 80.0
        def smooth_objective2(x):
            if not elapsed_ok():
                return 1e6
            P = x.reshape(N, 3)
            nrm = np.linalg.norm(P, axis=1, keepdims=True)
            nrm = np.where(nrm < 1e-12, 1.0, nrm)
            P = P / nrm
            try:
                hull = ConvexHull(P)
            except Exception:
                return 1e6
            radii = np.empty(len(hull.simplices))
            for i, tri in enumerate(hull.simplices):
                radii[i] = _facet_circumradius(P[tri[0]], P[tri[1]], P[tri[2]])
            m = np.max(radii)
            if m < 1e-15:
                return 0.0
            return m + np.log(np.sum(np.exp(global_beta * (radii - m)))) / global_beta

        res2 = minimize(
            smooth_objective2,
            pts.flatten(),
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9},
        )
        P2 = res2.x.reshape(N, 3)
        nrm = np.linalg.norm(P2, axis=1, keepdims=True)
        nrm = np.where(nrm < 1e-12, 1.0, nrm)
        P2 = P2 / nrm
        r2 = compute_covering_radius(P2)
        if r2 < compute_covering_radius(pts):
            pts = P2

    return pts


def _icosahedral_seed(n, rng):
    """Create n points on the sphere by jittering icosahedral vertices and reflecting."""
    phi = (1 + 5 ** 0.5) / 2.0
    ico = np.array([
        [-1,  phi, 0.], [1,  phi, 0.], [-1, -phi, 0.], [1, -phi, 0.],
        [0., -1,  phi], [0.,  1,  phi], [0., -1, -phi], [0.,  1, -phi],
        [ phi, 0., -1], [ phi, 0.,  1], [-phi, 0., -1], [-phi, 0.,  1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)

    # add antipodes to get 24 base points
    base = np.vstack([ico, -ico])

    pts = []
    while len(pts) < n:
        # pick a random base point and jitter it
        b = base[rng.integers(len(base))]
        noise = rng.normal(0, 0.3, 3)
        v = b + noise
        v = v / np.linalg.norm(v)
        pts.append(v)

    pts = np.array(pts[:n])
    return pts


def _facet_circumradius(a, b, c):
    """Angular circumradius (max angle from circumcenter to a,b,c) for a triangle on S^2.
    If the triangle's circumcenter is inside the triangle (on the same hemisphere side),
    the radius is the max of the three angles. Otherwise, the largest facet radius is
    the angle between the two farther vertices as seen from the antipodal point of the
    circumcenter, but for our purposes (we always take the outward-facing circumcenter),
    the max of the three dot-product angles with the outward circumcenter is correct.
    """
    # Spherical circumcenter = normalized cross product of the plane's normal?
    # For three points on the unit sphere, the circumcenter of the great-circle triangle
    # is along the direction perpendicular to the plane containing a,b,c, i.e.,
    #   n = (b - a) x (c - a)
    # normalized. But we need the one that is on the same side as the sphere interior
    # relative to the triangle. Since the convex hull is oriented outward, the normal
    # from (b-a)x(c-a) should be outward if the points are in CCW order as seen from
    # outside. We'll use that.
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    nrm = np.linalg.norm(n)
    if nrm < 1e-14:
        return 0.0
    n = n / nrm

    # The circumcenter on the sphere is n or -n. We choose the one that is on the
    # "short arc" side of the triangle, i.e., the one for which all three points
    # have positive dot with n (if the triangle is less than a hemisphere).
    dots = np.array([np.dot(n, a), np.dot(n, b), np.dot(n, c)])
    if np.min(dots) < 0:
        n = -n
        dots = -dots

    # The angular radius is the max angle from n to a,b,c
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    return float(np.max(angles))


def _worst_facet(P):
    """Return (triangle indices, circumcenter direction, circumradius) for the worst facet."""
    try:
        hull = ConvexHull(P)
    except Exception:
        return None, None, 0.0

    worst_r = -1.0
    worst_tri = None
    worst_cc = None
    for tri in hull.simplices:
        a = P[tri[0]]; b = P[tri[1]]; c = P[tri[2]]
        r = _facet_circumradius(a, b, c)
        if r > worst_r:
            worst_r = r
            worst_tri = tri
            ab = b - a
            ac = c - a
            n = np.cross(ab, ac)
            nrm = np.linalg.norm(n)
            if nrm < 1e-14:
                continue
            n = n / nrm
            dots = np.array([np.dot(n, a), np.dot(n, b), np.dot(n, c)])
            if np.min(dots) < 0:
                n = -n
            worst_cc = n

    return worst_tri, worst_cc, worst_r
# EVOLVE-BLOCK-END
