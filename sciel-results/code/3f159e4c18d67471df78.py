import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-12:
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(x_flat):
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    shifted = np.clip(shifted, 0, 100)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(x_flat, temperature):
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(x_flat, temperature):
    s = _soft_score(x_flat, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Support-line representation ----------

def _points_to_support_params(points):
    """
    Convert N points (assumed convex, CCW) to support-line parameters.
    For each edge i->i+1, compute the outward normal and offset.
    Returns array of shape (2N,): [n0x, n0y, d0, n1x, n1y, d1, ...]
    where n_i is the unit outward normal and d_i is the offset such that
    n_i · x = d_i defines the supporting line.
    """
    pts = points.copy()
    n = len(pts)
    params = np.zeros(2 * n)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        edge = p2 - p1
        # Outward normal: rotate edge by -90 degrees (for CCW polygon)
        # For CCW polygon, outward normal is (edge_y, -edge_x) normalized
        ex, ey = edge
        nx, ny = ey, -ex
        norm = np.sqrt(nx * nx + ny * ny)
        if norm < 1e-15:
            nx, ny = 1.0, 0.0
        else:
            nx /= norm
            ny /= norm
        d = nx * p1[0] + ny * p1[1]
        params[2 * i] = nx
        params[2 * i + 1] = ny
        params[2 * i + 2] if 2 * i + 2 < 2 * n else None  # placeholder
    # Actually let me redo this with proper indexing
    params = np.zeros(2 * n)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        edge = p2 - p1
        ex, ey = edge
        nx, ny = ey, -ex
        norm = np.sqrt(nx * nx + ny * ny)
        if norm < 1e-15:
            nx, ny = 1.0, 0.0
        else:
            nx /= norm
            ny /= norm
        d = nx * p1[0] + ny * p1[1]
        params[2 * i] = nx
        params[2 * i + 1] = d
    return params


def _support_params_to_points(params):
    """
    Convert support-line parameters to polygon vertices.
    params: shape (2N,) with [n0x, d0, n1x, d1, ..., n(N-1)x, d(N-1)]
    where n_i is unit normal and d_i is offset.
    Vertex i is intersection of lines i and i+1 (mod N).
    """
    n = len(params) // 2
    normals = params[0::2]  # n_x components
    offsets = params[1::2]  # d values
    
    # We need full normals. Let's store them differently.
    # Actually, let me use a different parameterization:
    # params = [angle_0, offset_0, angle_1, offset_1, ..., angle_(N-1), offset_(N-1)]
    # where angle_i is the direction of the outward normal, offset_i is the distance from origin.
    
    angles = params[0::2]
    offsets = params[1::2]
    
    normals_full = np.column_stack([np.cos(angles), np.sin(angles)])
    
    vertices = np.zeros((n, 2))
    valid = True
    for i in range(n):
        i1 = i
        i2 = (i + 1) % n
        n1 = normals_full[i1]
        n2 = normals_full[i2]
        d1 = offsets[i1]
        d2 = offsets[i2]
        
        # Solve: n1 · x = d1, n2 · x = d2
        A = np.array([n1, n2])
        b = np.array([d1, d2])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            valid = False
            break
        vertices[i, 0] = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
        vertices[i, 1] = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
    
    if not valid:
        return None
    
    # Verify convexity and that points are on the correct side
    # Check that all points satisfy n_j · x_i >= d_j - epsilon for all j, i
    # Actually for a convex polygon defined by inward-facing normals, we need n_j · x_i <= d_j
    # But our normals are outward, so n_j · x_i >= d_j is wrong.
    # For outward normals: the polygon is {x : n_j · x <= d_j for all j}
    # Wait, no. If n is outward normal and d is offset, the line is n·x = d.
    # The interior is n·x <= d (since normal points away from interior).
    # So all vertices should satisfy n_j · v_i <= d_j + eps for all j, i.
    
    # Check validity
    for j in range(n):
        n_j = normals_full[j]
        d_j = offsets[j]
        for i in range(n):
            val = n_j[0] * vertices[i, 0] + n_j[1] * vertices[i, 1]
            if val > d_j + 1e-6:
                # Point is outside the half-plane, polygon is not valid
                return None
    
    return vertices


def _neg_score_support(params):
    """Objective for support-line parameter optimization."""
    points = _support_params_to_points(params)
    if points is None:
        return 1e10
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _neg_soft_score_support(params, temperature):
    """Soft objective for support-line parameter optimization."""
    points = _support_params_to_points(params)
    if points is None:
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 1e10
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 1e10
    s = soft_min / hull_area
    return -s


def _points_to_support_angles(points):
    """
    Convert convex polygon points to support-line angle/offset parameters.
    Returns (2N,) array: [angle_0, offset_0, angle_1, offset_1, ...]
    """
    pts = points.copy()
    n = len(pts)
    params = np.zeros(2 * n)
    
    # Compute centroid for reference
    centroid = pts.mean(axis=0)
    
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        edge = p2 - p1
        ex, ey = edge
        # Outward normal for CCW polygon
        nx, ny = ey, -ex
        norm = np.sqrt(nx * nx + ny * ny)
        if norm < 1e-15:
            nx, ny = 1.0, 0.0
        else:
            nx /= norm
            ny /= norm
        angle = np.arctan2(ny, nx)
        # Offset: distance from origin to line along normal
        d = nx * p1[0] + ny * p1[1]
        params[2 * i] = angle
        params[2 * i + 1] = d
    
    return params


# ---------- Seed generators ----------

def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular():
    rng = np.random.RandomState(42)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.05 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_random_ring(rng):
    n_outer = int(rng.integers(5, 10))
    n_inner = N - n_outer
    if n_inner > 0:
        o_angles = np.sort(rng.uniform(0, 2 * np.pi, n_outer))
        i_angles = np.sort(rng.uniform(0, 2 * np.pi, n_inner))
        r_in = rng.uniform(0.3, 0.7)
        o = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
        i = np.column_stack([r_in * np.cos(i_angles), r_in * np.sin(i_angles)])
        return np.vstack([o, i])
    else:
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        return np.column_stack([np.cos(angles), np.sin(angles)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(x0, deadline, max_temp=0.05, min_temp=1e-4, n_steps=12):
    """
    Anneal the soft-min temperature and optimize at each step.
    Uses L-BFGS-B then Powell at each temperature level.
    """
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))

    for temp in temps:
        if time.time() >= deadline - 8:
            break

        if time.time() < deadline - 6:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        if time.time() < deadline - 4:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

    return best, best_s


def _polish_true_objective(x0, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    if time.time() < deadline - 6:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Nelder-Mead',
                options={'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    return best, best_s


def _local_search(points, deadline, n_iter=300, sigma_start=0.02, sigma_end=0.001):
    """
    Coordinate-wise local search with adaptive step size.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)

    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break

        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress

        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)

        for scale in [1.0, 0.5, 0.2, 2.0]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction

            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break

    return best, best_s


def _optimize_support_params(params0, deadline, max_temp=0.05, min_temp=1e-4, n_steps=8):
    """
    Optimize support-line parameters using annealed soft-min.
    params0: (2N,) array of [angle_0, offset_0, angle_1, offset_1, ...]
    """
    best = params0.copy()
    best_points = _support_params_to_points(best)
    best_s = _score(best_points) if best_points is not None else 0.0

    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))

    for temp in temps:
        if time.time() >= deadline - 6:
            break

        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _neg_soft_score_support,
                    best,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 80, 'xatol': 1e-8, 'fatol': 1e-8},
                )
                cand = res.x
                pts = _support_params_to_points(cand)
                if pts is not None:
                    s = _score(pts)
                    if s > best_s:
                        best_s = s
                        best = cand.copy()
                        best_points = pts
            except Exception:
                pass

        if time.time() < deadline - 3:
            try:
                res = minimize(
                    _neg_score_support,
                    best,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True},
                )
                cand = res.x
                pts = _support_params_to_points(cand)
                if pts is not None:
                    s = _score(pts)
                    if s > best_s:
                        best_s = s
                        best = cand.copy()
                        best_points = pts
            except Exception:
                pass

    if best_points is None:
        best_points = _support_params_to_points(best)
    return best_points, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = [
        _seed_regular(),
        _seed_two_ring(7, 0.55, np.pi / 6),
        _seed_two_ring(6, 0.5, np.pi / 5),
        _seed_two_ring(8, 0.4, np.pi / 4),
        _seed_three_ring(5, 4, 4, 0.65, 0.35, np.pi / 5, np.pi / 4),
        _seed_three_ring(6, 4, 3, 0.6, 0.3, np.pi / 3, np.pi / 6),
        _seed_quasi_regular(),
    ]

    for _ in range(8):
        seeds.append(_seed_random_ring(rng))

    for seed in seeds:
        _update_best(seed)

    # Phase 1: Standard Cartesian optimization from each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 40:
            break
        try:
            x_opt, s_opt = _annealed_soft_minimize(
                seed.flatten(),
                deadline,
                max_temp=0.05,
                min_temp=1e-4,
                n_steps=8,
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass

    # Phase 2: Support-line optimization from best points
    if best_points is not None and time.time() < deadline - 30:
        try:
            support_params = _points_to_support_angles(best_points)
            sl_points, sl_s = _optimize_support_params(
                support_params, deadline, max_temp=0.05, min_temp=1e-4, n_steps=6
            )
            _update_best(sl_points)
        except Exception:
            pass

    # Phase 3: Polish best with Cartesian
    if best_points is not None and time.time() < deadline - 20:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    # Phase 4: Local search
    if best_points is not None and time.time() < deadline - 15:
        try:
            ls_points, s_ls = _local_search(best_points, deadline, n_iter=200)
            _update_best(ls_points)
        except Exception:
            pass

    # Phase 5: More support-line optimization from different seeds
    for i, seed in enumerate(seeds[:5]):
        if time.time() > deadline - 20:
            break
        try:
            support_params = _points_to_support_angles(seed)
            sl_points, sl_s = _optimize_support_params(
                support_params, deadline, max_temp=0.03, min_temp=1e-4, n_steps=4
            )
            _update_best(sl_points)
        except Exception:
            pass

    # Phase 6: Final polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            x_final, s_final = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_final.reshape(N, 2))
        except Exception:
            pass

    # Phase 7: Support-line from final best
    if best_points is not None and time.time() < deadline - 8:
        try:
            support_params = _points_to_support_angles(best_points)
            sl_points, sl_s = _optimize_support_params(
                support_params, deadline, max_temp=0.02, min_temp=1e-5, n_steps=4
            )
            _update_best(sl_points)
        except Exception:
            pass

    # Phase 8: One more polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
