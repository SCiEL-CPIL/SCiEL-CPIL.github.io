"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13
TIME_LIMIT = 145.0  # safety margin from 150s kill


def _all_triangle_areas(pts):
    """Vectorized: compute all C(N,3) triangle areas. Returns (areas, idx_i, idx_j, idx_k)."""
    n = len(pts)
    idx = np.arange(n)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (ii < jj) & (jj < kk)
    ii, jj, kk = ii[mask], jj[mask], kk[mask]
    p1, p2, p3 = pts[ii], pts[jj], pts[kk]
    # area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas, ii, jj, kk


def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return -1e9
    areas, _, _, _ = _all_triangle_areas(pts)
    min_area = areas.min()
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return -1e9
    return min_area / hull_a


def _soft_score(pts, beta=200.0):
    """Smooth approximation of _score for gradient-based optimization."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    areas, _, _, _ = _all_triangle_areas(pts)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    # soft-min of areas
    neg = -beta * areas
    m = neg.max()
    logsum = m + np.log(np.sum(np.exp(neg - m)))
    soft_min = -logsum / beta
    return soft_min / hull_a


# ---------- Edge-vector convex polygon parameterization ----------
# A convex polygon is defined by N edges. Each edge i has:
#   - length L_i > 0
#   - turning angle theta_i in (0, 2*pi)
# with sum(theta_i) = 2*pi. The polygon closes if we start anywhere.
# We parameterize by log(L_i) and (theta_i - 2*pi/N) with the constraint
# that the sum of deviations is 0 (enforced by dropping the last one).

def _edges_to_points(log_lengths, angle_devs):
    """
    log_lengths: (N,) log of edge lengths
    angle_devs: (N-1,) deviations of turning angles from 2*pi/N (last is determined)
    Returns (N,2) polygon vertices (closed, starting at origin).
    """
    base_angle = 2.0 * np.pi / N
    # reconstruct full angle vector
    full_devs = np.zeros(N)
    full_devs[:-1] = angle_devs
    full_devs[-1] = -np.sum(angle_devs)  # ensure sum = 0
    angles = base_angle + full_devs
    # ensure angles are positive
    angles = np.maximum(angles, 1e-6)
    # re-normalize so sum = 2*pi (if clamping changed the sum)
    angles = angles * (2.0 * np.pi / angles.sum())

    lengths = np.exp(log_lengths)
    # edge directions: cumulative turn
    # Start with edge 0 pointing along +x. After edge i, we turn by angles[i].
    # direction of edge i = sum of angles[0..i-1]
    cum = np.cumsum(np.concatenate([[0.0], angles[:-1]]))  # (N,)
    dx = lengths * np.cos(cum)
    dy = lengths * np.sin(cum)
    # vertices: cumulative sum of edge vectors, starting at 0
    verts = np.zeros((N + 1, 2))
    verts[1:, 0] = np.cumsum(dx)
    verts[1:, 1] = np.cumsum(dy)
    # verts[0] == verts[N] ideally (closure). We take the first N vertices.
    pts = verts[:N].copy()
    # Center at centroid
    pts -= pts.mean(axis=0)
    return pts


def _points_to_edges(pts):
    """
    Convert a convex polygon (given in CCW order) to (log_lengths, angle_devs).
    Returns None if not convex / not closed.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return None
    # edges
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-12):
        return None
    # turning angles: angle between edge i and edge i+1
    # For CCW convex polygon, the exterior turning angle is positive.
    angles = np.zeros(N)
    for i in range(N):
        e1 = edges[i]
        e2 = edges[(i + 1) % N]
        # cross product sign tells us left/right turn
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        dot = e1[0] * e2[0] + e1[1] * e2[1]
        ang = np.arctan2(cross, dot)
        if ang < 0:
            ang += 2.0 * np.pi
        angles[i] = ang
    # For a simple closed CCW polygon, sum of turning angles = 2*pi
    # Check convexity: all turning angles in (0, pi)
    if np.any(angles <= 1e-6) or np.any(angles >= np.pi - 1e-6):
        return None
    if abs(angles.sum() - 2.0 * np.pi) > 1e-3:
        return None
    base_angle = 2.0 * np.pi / N
    devs = angles - base_angle
    # drop last
    log_lengths = np.log(lengths)
    angle_devs = devs[:-1].copy()
    return log_lengths, angle_devs


def _regular_polygon_params():
    """Regular 13-gon parameters."""
    base_angle = 2.0 * np.pi / N
    # For a regular N-gon with edge length L, circumradius R = L / (2*sin(pi/N))
    # Pick L = 1.0
    log_lengths = np.zeros(N)
    angle_devs = np.zeros(N - 1)
    return log_lengths, angle_devs


def _param_to_score(log_lengths, angle_devs, beta=None):
    pts = _edges_to_points(log_lengths, angle_devs)
    if beta is None:
        return _score(pts)
    return _soft_score(pts, beta=beta)


def _optimize_from_params(log_lengths, angle_devs, start_time, time_budget, beta=200.0):
    """
    Run L-BFGS-B on the edge parameters to maximize soft score.
    Returns (best_pts, best_score).
    """
    def neg_obj(x):
        ll = x[:N]
        ad = x[N:2*N - 1]
        return -_param_to_score(ll, ad, beta=beta)

    x0 = np.concatenate([log_lengths, angle_devs])
    # bounds: log_lengths in [-3, 3], angle_devs in [-pi/2, pi/2]
    bounds = [(-3.0, 3.0)] * N + [(-np.pi / 2, np.pi / 2)] * (N - 1)

    best_pts = _edges_to_points(log_lengths, angle_devs)
    best_score = _score(best_pts)

    t0 = time.time()
    while (time.time() - t0) < time_budget:
        try:
            res = minimize(
                neg_obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'maxfun': 100000, 'disp': False}
            )
            x0 = res.x
            ll = x0[:N]
            ad = x0[N:2*N - 1]
            pts = _edges_to_points(ll, ad)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        except Exception:
            break
        # If converged, stop
        if res.success and res.status == 0:
            break

    return best_pts, best_score


def _coordinate_descent_min_triangle(pts, start_time, time_budget, max_rounds=2000):
    """
    Local refinement: repeatedly find the minimum-area triangle and
    nudge one of its vertices (in the direction that increases that triangle's area
    while keeping convexity) using a small line search.
    This directly attacks the bottleneck triangle.
    """
    pts = np.array(pts, dtype=float)
    best_score = _score(pts)
    best_pts = pts.copy()
    t0 = time.time()

    for _ in range(max_rounds):
        if (time.time() - t0) > time_budget:
            break
        areas, ii, jj, kk = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        i, j, k = ii[min_idx], jj[min_idx], kk[min_idx]

        # Try moving each of the three vertices slightly outward (away from the
        # triangle's centroid) to increase this triangle's area.
        p_i, p_j, p_k = pts[i], pts[j], pts[k]
        centroid = (p_i + p_j + p_k) / 3.0

        improved = False
        for v in [i, j, k]:
            p_v = pts[v]
            # direction: from centroid toward p_v (pushes the triangle larger)
            direction = p_v - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm

            # line search over step sizes
            best_step = 0.0
            best_local = min_area
            for step in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
                new_pts = pts.copy()
                new_pts[v] = p_v + step * direction
                # quick check: recompute the specific triangle area
                a_new = 0.5 * abs(
                    (new_pts[j][0] - new_pts[i][0]) * (new_pts[k][1] - new_pts[i][1]) -
                    (new_pts[j][1] - new_pts[i][1]) * (new_pts[k][0] - new_pts[i][0])
                )
                # also ensure no triangle became smaller (sample a few)
                # For speed, just check this triangle and overall min via full recompute occasionally
                if a_new > best_local:
                    best_local = a_new
                    best_step = step

            if best_step > 0:
                new_pts = pts.copy()
                new_pts[v] = p_v + best_step * direction
                sc = _score(new_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break

        if not improved:
            # try a small random perturbation to escape
            rng = np.random.default_rng()
            new_pts = pts + rng.normal(0, 1e-4, size=pts.shape)
            sc = _score(new_pts)
            if sc > best_score:
                best_score = sc
                best_pts = new_pts.copy()
                pts = new_pts.copy()
            else:
                break

    return best_pts, best_score


def _differential_evolution_search(start_time, time_budget, popsize=12, maxiter=40, seed=42):
    """Global search in edge-parameter space."""
    def de_obj(x):
        ll = x[:N]
        ad = x[N:2*N - 1]
        return -_param_to_score(ll, ad, beta=100.0)

    # bounds
    bounds = [(-2.0, 2.0)] * N + [(-0.8, 0.8)] * (N - 1)

    def callback(xk, convergence):
        return (time.time() - start_time) > time_budget

    try:
        result = differential_evolution(
            de_obj, bounds,
            seed=seed, popsize=popsize, maxiter=maxiter,
            tol=1e-7, init='halton', polish=False,
            update_interval=1, callback=callback
        )
        ll = result.x[:N]
        ad = result.x[N:2*N - 1]
        pts = _edges_to_points(ll, ad)
        sc = _score(pts)
        return pts, sc, (ll, ad)
    except Exception:
        return None, -1e9, None


def _generate_starts(rng):
    """Generate diverse starting parameters for multi-start."""
    starts = []
    # 1. Regular polygon
    starts.append(_regular_polygon_params())
    # 2-8: perturb regular polygon angles
    for _ in range(7):
        ll = np.zeros(N) + rng.normal(0, 0.1)
        ad = rng.normal(0, 0.1, size=N - 1)
        starts.append((ll, ad))
    # 9-14: larger perturbations
    for _ in range(6):
        ll = np.zeros(N) + rng.normal(0, 0.3)
        ad = rng.normal(0, 0.3, size=N - 1)
        starts.append((ll, ad))
    # 15-18: structured angle variations (alternating large/small turns)
    for amp in [0.2, 0.4, 0.6]:
        ad = np.zeros(N - 1)
        for i in range(N - 1):
            ad[i] = amp * (1 if i % 2 == 0 else -1)
        ll = np.zeros(N)
        starts.append((ll, ad))
    return starts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_score = -1e9

    def _update(pts, sc):
        nonlocal best_pts, best_score
        if pts is not None and sc > best_score:
            best_score = sc
            best_pts = np.array(pts, dtype=float)

    # Phase 1: Differential evolution (global)
    de_time = min(40.0, TIME_LIMIT - (time.time() - start_time) - 60.0)
    if de_time > 10.0:
        pts, sc, params = _differential_evolution_search(
            start_time, de_time, popsize=12, maxiter=30, seed=42
        )
        _update(pts, sc)

    # Phase 2: Multi-start local refinement
    starts = _generate_starts(rng)
    # Also add the DE result if we have it
    if 'params' in locals() and params is not None:
        starts.append(params)

    for idx, (ll, ad) in enumerate(starts):
        if (time.time() - start_time) > TIME_LIMIT - 25.0:
            break
        remaining = TIME_LIMIT - (time.time() - start_time) - 20.0
        per_start = max(2.0, remaining / max(1, len(starts) - idx))

        # L-BFGS-B refinement
        try:
            pts, sc = _optimize_from_params(
                ll, ad, start_time, min(per_start * 0.6, 10.0), beta=200.0
            )
            _update(pts, sc)
        except Exception:
            pass

        # Coordinate descent on min triangle
        if best_pts is not None:
            try:
                pts2, sc2 = _coordinate_descent_min_triangle(
                    best_pts, start_time, min(per_start * 0.4, 8.0), max_rounds=800
                )
                _update(pts2, sc2)
            except Exception:
                pass

    # Phase 3: Final polish on best
    if best_pts is not None and (time.time() - start_time) < TIME_LIMIT - 10.0:
        params = _points_to_edges(best_pts)
        if params is not None:
            ll, ad = params
            remaining = TIME_LIMIT - (time.time() - start_time) - 8.0
            try:
                pts, sc = _optimize_from_params(ll, ad, start_time, remaining * 0.5, beta=500.0)
                _update(pts, sc)
            except Exception:
                pass
            if best_pts is not None:
                try:
                    pts2, sc2 = _coordinate_descent_min_triangle(
                        best_pts, start_time, remaining * 0.5, max_rounds=2000
                    )
                    _update(pts2, sc2)
                except Exception:
                    pass

    if best_pts is None:
        # Fallback: regular polygon
        ll, ad = _regular_polygon_params()
        best_pts = _edges_to_points(ll, ad)

    return best_pts
