# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 600.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _vertex_score(dists):
    """Exact per-vertex score given the 19 distances from one vertex to the others."""
    d = np.sort(dists)
    # d[0] is the nearest (k=0 case), d[18] is the farthest (k=17 case)
    if d[0] < 0.05 or d[18] < 0.1:
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = 0.5 * (d[k] + d[k + 1])
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k - 1:k + 3]
    raw = np.mean(np.maximum(sel / D, D / sel))
    return float(min(1.0, 1.0 / raw))


def _min_score(pts):
    """Minimum vertex score over all vertices (exact)."""
    n = pts.shape[0]
    best = 0.0
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        np.fill_diagonal(d, 0.0)
        s = _vertex_score(d)
        if s > best:
            best = s
    return best


def _smooth_min_score(pts):
    """Smooth surrogate for the minimum vertex score (for optimization)."""
    n = pts.shape[0]
    t = 0.02
    scores = np.empty(n)
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        np.fill_diagonal(d, 0.0)
        d = np.sort(d)
        # hard floor: if nearest or farthest too small, score 0
        if d[0] < 0.05 or d[18] < 0.1:
            scores[i] = 0.0
            continue
        gaps = np.diff(d)
        k = int(np.argmin(gaps))
        D = 0.5 * (d[k] + d[k + 1])
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            sel = d[0:4]
        elif k == 17:
            sel = d[15:19]
        else:
            sel = d[k - 1:k + 3]
        raw = np.mean(np.maximum(sel / D, D / sel))
        scores[i] = min(1.0, 1.0 / raw)
    # log-sum-exp for the minimum: min_i s_i = -logsumexp(-s_i)
    m = np.max(scores)
    if m <= 0.0:
        return 0.0
    lse = -m + np.log(np.sum(np.exp(-(scores - m) / t)) * t)
    return float(-lse)


def _is_valid(pts):
    """Check strict convexity, no repeats, total turn +-2*pi."""
    n = pts.shape[0]
    # no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                return False
    # cross products all same sign and nonzero
    turns = np.empty(n)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        turns[i] = v1[0] * v2[1] - v1[1] * v2[0]
    if np.any(np.abs(turns) < 1e-9):
        return False
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return False
    # total turning +-2*pi: check via exterior angles
    total = 0.0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        a1 = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        a2 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        diff = a2 - a1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        total += diff
    if not (abs(total - 2 * np.pi) < 1e-3 or abs(total + 2 * np.pi) < 1e-3):
        return False
    return True


def _normalize(pts):
    c = pts.mean(axis=0)
    p = pts - c
    dists = np.linalg.norm(p, axis=1)
    mean_d = dists.mean()
    if mean_d <= 0.01:
        return p
    return p / mean_d


def _regular20():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _angular_to_pts(theta):
    """Convert 20 angular perturbations (relative to regular 20-gon) to points."""
    n = 20
    base = 2.0 * np.pi / n
    angles = np.arange(n) * base + theta
    # ensure sorted / cyclic: wrap and sort to preserve order
    angles = angles - angles.min()
    angles = np.sort(angles)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _optimize_from(theta0, iters=4000, lr0=0.02):
    """Nelder-Mead style local optimization on angular perturbations."""
    n = 20
    x = theta0.copy()
    # initial simplex: small perturbations
    simplex = [x.copy()]
    for i in range(n):
        dx = x.copy()
        dx[i] += 0.05
        simplex.append(dx)
    fvals = np.array([_smooth_min_score(_angular_to_pts(v)) for v in simplex])
    for it in range(iters):
        if _time_left() < 30.0:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        # reflect
        xr = centroid + 1.0 * (centroid - worst)
        fr = _smooth_min_score(_angular_to_pts(xr))
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe = _smooth_min_score(_angular_to_pts(xe))
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + 0.5 * (worst - centroid)
            fc = _smooth_min_score(_angular_to_pts(xc))
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = 0.5 * (simplex[i] + simplex[0])
                    fvals[i] = _smooth_min_score(_angular_to_pts(simplex[i]))
        # keep best
        bi = int(np.argmin(fvals))
        x = simplex[bi]
    return x


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    best_pts = _regular20()
    best_score = _min_score(best_pts)

    # Start from regular, try a few random angular perturbations and optimize
    rng = np.random.default_rng(12345)
    n_attempts = 8
    for a in range(n_attempts):
        if _time_left() < 60.0:
            break
        if a == 0:
            theta0 = np.zeros(n)
        else:
            theta0 = rng.normal(0.0, 0.1, size=n)
        theta_opt = _optimize_from(theta0, iters=2500)
        pts = _angular_to_pts(theta_opt)
        pts = _normalize(pts)
        if not _is_valid(pts):
            continue
        s = _min_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    # Final polish: a few more iterations from the best found
    if _time_left() > 60.0 and best_score < 1.0:
        # re-optimize from best
        base = 2.0 * np.pi / n
        # recover angles of best_pts
        ang = np.arctan2(best_pts[:, 1], best_pts[:, 0])
        ang = np.unwrap(ang)
        ang0 = (ang - np.arange(n) * base)
        theta_opt = _optimize_from(ang0, iters=1500)
        pts = _normalize(_angular_to_pts(theta_opt))
        if _is_valid(pts):
            s = _min_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    return best_pts


# EVOLVE-BLOCK-END
