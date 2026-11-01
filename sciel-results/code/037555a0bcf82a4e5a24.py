"""Optimized 130-point sphere covering: smooth max-distance GD + facet polish."""
import time
import numpy as np

N = 130
_DEADLINE = 140.0  # safety margin under 150s wall-clock
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return pts / n


def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex-hull facets."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    best = 0.0
    for simplex in hull.simplices:
        v = pts[simplex]  # (k,3) k<=4
        # circumcenter of the spherical facet: solve for c maximizing min angle
        # For a facet, the farthest point is the circumcenter of the spherical polygon.
        # Approximate: the point on sphere equidistant to all facet vertices.
        # Solve (v_i . c)^2 = const with ||c||=1 -> v_i . c = t for all i.
        # For k<=4 points, find c in span of v_i.
        k = v.shape[0]
        # Solve V c = t * 1  (k equations, 3 unknowns + t)
        # Use least squares: minimize ||V c - t|| with constraint.
        # Better: the circumcenter is the normalized solution of V^T V c = V^T (1) * t
        # Actually solve: for the point equidistant, c = argmin ||c|| s.t. V c = t*1
        # Use SVD-free: c is in row space of V. Let c = V^T w. Then V V^T w = t 1.
        G = v @ v.T  # k x k
        # Solve G w = 1 (then t = (G w)_i / ... ) -- actually V c = t 1 with c=V^T w gives G w = t 1.
        # We want the c on unit sphere. The direction c = V^T w / ||V^T w||, and the
        # angular radius is arccos(t) where t = (G w)_i (constant).
        try:
            w = np.linalg.solve(G, np.ones(k))
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(G) @ np.ones(k)
        c = v.T @ w
        nc = np.linalg.norm(c)
        if nc < 1e-10:
            continue
        c = c / nc
        t = float(np.mean(v @ c))
        t = np.clip(t, -1.0, 1.0)
        ang = np.degrees(np.arccos(t))
        if ang > best:
            best = ang
    return best


def _smooth_max_radius_grad(pts, beta=40.0):
    """
    Smooth surrogate: R_beta = (1/beta) log sum_i exp(beta * d_i) where d_i = max_j angle(i,j).
    Returns (value, grad). d_i = arccos(min_j v_i . v_j)  (nearest neighbor angle).
    """
    G = pts @ pts.T  # N x N
    G = np.clip(G, -1.0, 1.0)
    # nearest neighbor angular distance for each i (excluding self)
    np.fill_diagonal(G, 2.0)  # so self is not min
    min_dot = G.min(axis=1)  # N
    min_dot = np.clip(min_dot, -1.0, 1.0)
    d = np.arccos(min_dot)  # N, nearest-neighbor angular distance (radians)
    # smooth max
    dm = d - d.max()
    logsum = d.max() + np.log(np.sum(np.exp(beta * dm)))
    R = logsum / beta
    # gradient: d R / d d_i = (beta / (beta * sum exp)) * beta * exp(beta d_i)?? 
    # R = (1/beta) log sum exp(beta d_i)
    # dR/dd_i = exp(beta d_i) / sum exp(beta d_j)
    w = np.exp(beta * dm)
    w = w / w.sum()  # softmax weights
    # d d_i / d v_i: d_i = arccos(min_j v_i . v_j). Let j_i = argmin.
    # d(arccos x)/dx = -1/sqrt(1-x^2). And d(v_i . v_j)/d v_i = v_j.
    j = G.argmin(axis=1)
    vj = pts[j]
    # derivative of d_i w.r.t. v_i: -1/sqrt(1 - min_dot^2) * vj_i
    denom = np.sqrt(np.maximum(1.0 - min_dot * min_dot, 1e-12))
    # dR/dv_i = w_i * (-1/denom_i) * vj_i
    grad = np.zeros_like(pts)
    scale = -w / denom
    grad = scale[:, None] * vj
    # But v_i also appears in other rows' min (as the neighbor). Add those contributions.
    # For each j, it is the min-neighbor of some set of i. Contribution to dR/dv_j:
    # sum over i where j_i==j of w_i * (-1/denom_i) * v_i
    for i in range(N):
        jj = j[i]
        grad[jj] += scale[i] * pts[i]
    return R, grad


def _repulsive_grad(pts, beta=40.0):
    """Gradient of smooth-max nearest-neighbor distance (pushes points apart)."""
    return _smooth_max_radius_grad(pts, beta)[1]


def _gd_step(pts, beta=40.0, lr=0.02):
    R, grad = _smooth_max_radius_grad(pts, beta)
    pts = pts - lr * grad
    return _normalize(pts), R


def _facet_polish(pts, max_iters=2000, tol=1e-9):
    """
    Move each vertex toward its facet's circumcenter (shrinks circumradius).
    Iterate until no improvement.
    """
    from scipy.spatial import ConvexHull
    pts = _normalize(pts)
    for it in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        # For each facet, compute circumcenter c_facet and the angular radius.
        # Accumulate a target for each vertex: move toward the circumcenter of
        # each facet it belongs to, weighted by how "tight" that facet is.
        disp = np.zeros_like(pts)
        improved = 0.0
        for simplex in hull.simplices:
            v = pts[simplex]
            k = v.shape[0]
            G = v @ v.T
            try:
                w = np.linalg.solve(G, np.ones(k))
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(G) @ np.ones(k)
            c = v.T @ w
            nc = np.linalg.norm(c)
            if nc < 1e-10:
                continue
            c = c / nc
            t = float(np.mean(v @ c))
            t = np.clip(t, -1.0, 1.0)
            # circumradius angle
            # Move each vertex of this facet toward c to shrink the cap.
            # Step proportional to how far the facet is from being "tight".
            for idx, vi in zip(simplex, v):
                # direction from vi toward c on the sphere (tangent)
                tang = c - (vi @ c) * vi
                nt = np.linalg.norm(tang)
                if nt > 1e-10:
                    tang = tang / nt
                    # step size: small, scaled by (1 - t) so tight facets move more
                    step = 0.01 * (1.0 - t)
                    disp[idx] += step * tang
        if np.linalg.norm(disp) < tol:
            break
        pts = _normalize(pts + disp)
        if _elapsed() > _DEADLINE:
            break
    return _normalize(pts)


def _random_start(seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    return _normalize(pts)


def construct():
    best_pts = None
    best_rad = np.inf

    # Multi-restart smooth-max GD from different seeds.
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        if _elapsed() > _DEADLINE - 25.0:
            break
        pts = _random_start(seed)
        # Anneal beta from low to high, lr decay.
        betas = [8.0, 16.0, 32.0, 64.0, 128.0]
        lrs = [0.05, 0.03, 0.015, 0.008, 0.004]
        for b, lr in zip(betas, lrs):
            steps = 400
            for s in range(steps):
                if _elapsed() > _DEADLINE - 20.0:
                    break
                pts, R = _gd_step(pts, beta=b, lr=lr)
            # occasional facet polish mid-way
        # final polish
        pts = _facet_polish(pts, max_iters=800)
        try:
            rad = _covering_radius_deg(pts)
        except Exception:
            rad = np.inf
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()

    # Final high-precision polish on the best.
    if best_pts is not None:
        best_pts = _facet_polish(best_pts, max_iters=2000, tol=1e-10)

    if best_pts is None:
        # fallback: Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return _normalize(best_pts)
