import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
_DEADLINE_MARGIN = 8.0  # seconds before hard kill

# Precompute all triangle indices once
_TRI = np.array(list(combinations(range(N), 3)))
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]


def _all_areas(pts):
    """Return array of all C(13,3) triangle areas (vectorized)."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0


def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_all_areas(pts)) / ha)


def _soft_neg_score(pts_flat, beta):
    """Smooth surrogate: -soft_min_area / hull_area."""
    pts = pts_flat.reshape(N, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e6
    areas = _all_areas(pts)
    # log-sum-exp for -beta * areas
    neg_ba = -beta * areas
    m = np.max(neg_ba)
    if m < -1e5:
        return 1e6
    lse = m + np.log(np.sum(np.exp(neg_ba - m)))
    soft_min = -lse / beta
    return -(soft_min / ha)


def _hard_neg(pts_flat):
    return -_score(pts_flat.reshape(N, 2))


def _worst_triangles(pts, n_worst=3):
    """Indices of the n_worst smallest-area triangles."""
    areas = _all_areas(pts)
    idx = np.argsort(areas)[:n_worst]
    return idx, areas[idx]


def _repair_worst_triangles(pts, n_worst=3, max_iters=40, deadline=None):
    """
    Directly move vertices of the worst triangles to increase their area.
    Uses coordinate descent: for each vertex of a worst triangle, try small
    perturbations in 8 directions and keep the best.
    """
    pts = pts.copy()
    current_score = _score(pts)
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]
    ])
    step_sizes = [0.05, 0.03, 0.01, 0.005]

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - _DEADLINE_MARGIN:
            break

        worst_idx, _ = _worst_triangles(pts, n_worst)
        improved = False

        for tri_i in worst_idx:
            i, j, k = _TRI[tri_i]
            # Collect unique vertex indices in this triangle
            verts = list(set([i, j, k]))
            for v in verts:
                best_pos = pts[v].copy()
                best_local = _score(pts)
                for ss in step_sizes:
                    for d in directions:
                        trial = pts.copy()
                        trial[v] = pts[v] + ss * d
                        ts = _score(trial)
                        if ts > best_local + 1e-12:
                            best_local = ts
                            best_pos = trial[v].copy()
                            improved = True
                    if improved:
                        break
                pts[v] = best_pos

        new_score = _score(pts)
        if not improved or new_score <= current_score - 1e-12:
            break
        current_score = new_score

    return pts, current_score


def _symmetric_points(n_hull, n_inner, r_h, r_i, rot_h, rot_i):
    """Two concentric regular polygons."""
    ang_h = np.linspace(rot_h, rot_h + 2 * np.pi, n_hull, endpoint=False)
    xs = r_h * np.cos(ang_h)
    ys = r_h * np.sin(ang_h)
    if n_inner > 0:
        ang_i = np.linspace(rot_i, rot_i + 2 * np.pi, n_inner, endpoint=False)
        xs = np.concatenate([xs, r_i * np.cos(ang_i)])
        ys = np.concatenate([ys, r_i * np.sin(ang_i)])
    return np.column_stack([xs, ys])


def _symmetric_objective(params, n_h, n_i, beta):
    r_h, r_i, rot_h, rot_i = params
    if r_h < 0.2 or r_i < -0.1 or (n_i > 0 and r_i >= r_h):
        return 1e6
    pts = _symmetric_points(n_h, n_i, r_h, max(0.0, r_i), rot_h, rot_i)
    return _soft_neg_score(pts.flatten(), beta)


def _symmetric_hard(params, n_h, n_i):
    r_h, r_i, rot_h, rot_i = params
    if r_h < 0.2 or r_i < -0.1 or (n_i > 0 and r_i >= r_h):
        return 1e6
    pts = _symmetric_points(n_h, n_i, r_h, max(0.0, r_i), rot_h, rot_i)
    return _hard_neg(pts.flatten())


def _diverse_inits():
    """Generate diverse starting configurations."""
    rng = np.random.RandomState(42)
    inits = []

    # Regular 13-gon
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(ang), np.sin(ang)]))

    # 7+6 two-ring
    for r2 in [0.3, 0.4, 0.5, 0.6]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        inits.append(np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        ]))

    # 12+1 center
    a12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    inits.append(np.vstack([
        np.column_stack([np.cos(a12), np.sin(a12)]),
        np.array([[0.0, 0.0]])
    ]))

    # 9+4
    for r2 in [0.4, 0.5, 0.6]:
        a1 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        inits.append(np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        ]))

    # 6+5+2 three-ring
    for r2, r3 in [(0.4, 0.2), (0.5, 0.25), (0.6, 0.3)]:
        a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 2, endpoint=False) + np.pi / 2
        inits.append(np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
            np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
        ]))

    # Random 2-ring
    for _ in range(12):
        n1 = rng.choice([5, 6, 7, 8])
        n2 = N - n1
        r2 = rng.uniform(0.3, 0.7)
        a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2 * np.pi)
        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2 * np.pi)
        inits.append(np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        ]))

    # Random 3-ring
    for _ in range(10):
        n1 = rng.choice([4, 5, 6])
        n2 = rng.choice([3, 4, 5])
        n3 = N - n1 - n2
        if n3 < 1:
            continue
        r2 = rng.uniform(0.4, 0.7)
        r3 = rng.uniform(0.15, 0.35)
        a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2 * np.pi)
        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2 * np.pi)
        a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + rng.uniform(0, 2 * np.pi)
        inits.append(np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
            np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
        ]))

    # Random disk points
    for _ in range(8):
        r = rng.rand(N) ** 0.5
        th = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    return inits


def _soft_refine(pts, deadline):
    """Multi-beta L-BFGS-B refinement."""
    cur = pts.copy()
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        try:
            res = opt.minimize(
                lambda x, b=beta: _soft_neg_score(x, b),
                cur.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_pts = res.x.reshape(N, 2)
            if _score(new_pts) > _score(cur):
                cur = new_pts
        except Exception:
            pass
    return cur


def _nm_polish(pts, deadline):
    """Nelder-Mead final polish."""
    try:
        res = opt.minimize(
            _hard_neg,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        new_pts = res.x.reshape(N, 2)
        if _score(new_pts) >= _score(pts):
            return new_pts
    except Exception:
        pass
    return pts


def construct():
    start = time.time()
    deadline = start + 142.0  # 8s margin before 150s kill

    best_pts = None
    best_score = -1e9

    def _try_update(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # Phase 1: Symmetric structured optimization
    splits = [(13, 0), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]
    for n_h, n_i in splits:
        if time.time() > deadline - 25.0:
            break
        r_h0 = 1.0
        r_i0 = 0.4 if n_i > 0 else 0.0
        rot_h0 = 0.0
        rot_i0 = np.pi / n_i if n_i > 1 else 0.0
        x0 = np.array([r_h0, r_i0, rot_h0, rot_i0])

        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 25.0:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta, nh=n_h, ni=n_i: _symmetric_objective(x, nh, ni, b),
                    x0, method='L-BFGS-B',
                    bounds=[(0.3, 2.0), (0.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x0 = res.x
                pts = _symmetric_points(n_h, n_i, x0[0], max(0.0, x0[1]), x0[2], x0[3])
                _try_update(pts)
            except Exception:
                pass

        # NM polish on symmetric params
        if time.time() > deadline - 25.0:
            break
        try:
            res = opt.minimize(
                lambda x, nh=n_h, ni=n_i: _symmetric_hard(x, nh, ni),
                x0, method='Nelder-Mead',
                options={'maxiter': 150, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            pts = _symmetric_points(n_h, n_i, res.x[0], max(0.0, res.x[1]), res.x[2], res.x[3])
            _try_update(pts)
        except Exception:
            pass

    # Phase 2: Diverse initializations + repair + soft refine
    inits = _diverse_inits()
    for init_pts in inits:
        if time.time() > deadline - 15.0:
            break

        # Direct worst-triangle repair (the key structural addition)
        repaired, _ = _repair_worst_triangles(init_pts, n_worst=3, max_iters=30, deadline=deadline)
        _try_update(repaired)

        # Soft-min gradient refinement
        if time.time() > deadline - 15.0:
            break
        refined = _soft_refine(repaired, deadline)
        _try_update(refined)

        # Another round of repair after gradient refinement
        if time.time() > deadline - 15.0:
            break
        repaired2, _ = _repair_worst_triangles(refined, n_worst=2, max_iters=20, deadline=deadline)
        _try_update(repaired2)

    # Phase 3: Final polish on best found
    if best_pts is not None and time.time() < deadline - 5.0:
        polished = _nm_polish(best_pts, deadline)
        _try_update(polished)

        # One more repair pass
        if time.time() < deadline - 3.0:
            final_repaired, _ = _repair_worst_triangles(polished, n_worst=2, max_iters=15, deadline=deadline)
            _try_update(final_repaired)

    if best_pts is None:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_pts
