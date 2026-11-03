"""Heilbronn triangles: place 11 points to maximize the minimum triangle area.

Strategy: Direct gradient ascent on the minimum-area triangle with
coordinate-wise line searches. Much faster per iteration than soft-min
+ L-BFGS, enabling many more restarts and diversification.
"""
import numpy as np
import time
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 147.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)


def _bary_to_cart(abc):
    x = abc[:, 1] + 0.5 * abc[:, 2]
    y = abc[:, 2] * S3H
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _project_bary(abc):
    abc = np.maximum(abc, 0)
    s = abc.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    return abc / s


def _all_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.abs(cross) * 0.5


def _min_area(pts):
    return np.min(_all_areas(pts))


def _score(pts):
    return _min_area(pts) / TRI_AREA


def _move_in_bary(abc, idx, db, dc):
    """Move point idx by (db, dc) in barycentric coords, project back."""
    cand = abc.copy()
    cand[idx, 1] += db
    cand[idx, 2] += dc
    cand = _project_bary(cand)
    return cand


def _line_search(abc, idx, db, dc, cur_score, max_iter=20):
    """Golden-section line search along (db, dc) direction for point idx."""
    lo, hi = 0.0, 1.0
    best_t = 0.0
    best_sc = cur_score
    gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate

    for _ in range(max_iter):
        t1 = lo + gr * (hi - lo)
        t2 = hi - gr * (hi - lo)
        cand1 = _move_in_bary(abc, idx, t1 * db, t1 * dc)
        sc1 = _score(_bary_to_cart(cand1))
        cand2 = _move_in_bary(abc, idx, t2 * db, t2 * dc)
        sc2 = _score(_bary_to_cart(cand2))

        if sc1 > sc2:
            hi = t1
            if sc1 > best_sc:
                best_sc = sc1
                best_t = t1
        else:
            lo = t2
            if sc2 > best_sc:
                best_sc = sc2
                best_t = t2

    return best_t, best_sc


def _gradient_ascent(abc, max_iter=300, tol=1e-10):
    """Direct gradient ascent: push vertices of min-area triangle apart."""
    cur = abc.copy()
    cur_score = _score(_bary_to_cart(cur))
    best = cur.copy()
    best_score = cur_score

    for it in range(max_iter):
        pts = _bary_to_cart(cur)
        areas = _all_areas(pts)
        min_idx = np.argmin(areas)
        min_val = areas[min_idx]

        i, j, k = _TRIPLETS[min_idx]
        pi, pj, pk = pts[i], pts[j], pts[k]

        # Area = 0.5 * |cross(pj-pi, pk-pi)|
        # For pushing apart, we want to increase the area.
        # Gradient of area w.r.t. each point (assuming fixed orientation):
        # dA/dpi = -0.5 * cross(pk-pi, pj-pi) direction... 
        # Simpler: push each vertex away from the midpoint of the other two.
        
        # Direction for point i: away from midpoint of j,k
        mid_jk = (pj + pk) * 0.5
        dir_i = pi - mid_jk
        ni = np.linalg.norm(dir_i)
        if ni > 1e-12:
            dir_i /= ni
        else:
            dir_i = np.array([0.0, 0.0])

        mid_ik = (pi + pk) * 0.5
        dir_j = pj - mid_ik
        nj = np.linalg.norm(dir_j)
        if nj > 1e-12:
            dir_j /= nj
        else:
            dir_j = np.array([0.0, 0.0])

        mid_ij = (pi + pj) * 0.5
        dir_k = pk - mid_ij
        nk = np.linalg.norm(dir_k)
        if nk > 1e-12:
            dir_k /= nk
        else:
            dir_k = np.array([0.0, 0.0])

        # Convert Cartesian directions to barycentric (db, dc)
        # x = b + 0.5c, y = S3H * c
        # dx = db + 0.5dc, dy = S3H * dc
        # dc = dy / S3H, db = dx - 0.5dc
        def _cart_dir_to_bary(dx, dy):
            dc = dy / S3H
            db = dx - 0.5 * dc
            return db, dc

        # Try moving each of the three vertices
        improved = False
        for idx, dcart in [(i, dir_i), (j, dir_j), (k, dir_k)]:
            db, dc = _cart_dir_to_bary(dcart[0], dcart[1])
            t, sc = _line_search(cur, idx, db, dc, cur_score)
            if sc > cur_score + 1e-12:
                cur = _move_in_bary(cur, idx, t * db, t * dc)
                cur_score = sc
                improved = True
                if sc > best_score:
                    best_score = sc
                    best = cur.copy()
                break

        if not improved:
            # No improvement from the min triangle; try small random kicks
            # on a random point to escape
            ridx = np.random.randint(N)
            db = np.random.normal(0, 0.01)
            dc = np.random.normal(0, 0.01)
            cand = _move_in_bary(cur, ridx, db, dc)
            sc = _score(_bary_to_cart(cand))
            if sc > cur_score:
                cur = cand
                cur_score = sc
                if sc > best_score:
                    best_score = sc
                    best = cur.copy()
            else:
                # Accept with small probability (simulated annealing flavor)
                if np.random.random() < 0.1:
                    cur = cand
                    cur_score = sc

    return best, best_score


def _random_bary(rng):
    u = rng.random(N)
    v = rng.random(N)
    abc = np.column_stack([u, v, 1.0 - u - v])
    return _project_bary(abc)


def _random_cart(rng):
    return _bary_to_cart(_random_bary(rng))


def _sa_bary(abc, n_iter=500, T0=0.02, T1=0.0005, rng=None):
    """Quick simulated annealing to diversify."""
    if rng is None:
        rng = np.random.RandomState(0)

    cur = abc.copy()
    cur_score = _score(_bary_to_cart(cur))
    best = cur.copy()
    best_score = cur_score

    for i in range(n_iter):
        T = T0 * (T1 / T0) ** (i / n_iter)
        idx = rng.randint(N)
        db = rng.normal(0, 0.02)
        dc = rng.normal(0, 0.02)
        cand = _move_in_bary(cur, idx, db, dc)
        sc = _score(_bary_to_cart(cand))

        if sc >= cur_score:
            cur = cand
            cur_score = sc
            if sc > best_score:
                best_score = sc
                best = cur.copy()
        else:
            delta = cur_score - sc
            if delta > 1e-12:
                if rng.random() < np.exp(-delta / max(T, 1e-12)):
                    cur = cand
                    cur_score = sc

    return best, best_score


def _fps_init(rng, n=200):
    """Farthest point sampling for diverse initial points."""
    pts = [_random_cart(rng)]
    for _ in range(N - 1):
        best_d = -1
        best_p = None
        for _ in range(n):
            cand = _random_cart(rng)
            d = np.min(np.linalg.norm(np.array(pts) - cand, axis=1))
            if d > best_d:
                best_d = d
                best_p = cand
        pts.append(best_p)
    return _cart_to_bary(np.array(pts))


def _grid_init(rng, rows=5):
    """Grid-based initialization."""
    pts = []
    for i in range(rows):
        for j in range(rows):
            t = i / (rows - 1)
            s = j / (rows - 1)
            x = t * (1.0 - s) + s * 0.5
            y = s * S3H
            if x >= -1e-9 and y >= -1e-9 and (1.0 - x - y / S3H) >= -1e-9:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        idx = rng.choice(len(pts), N, replace=False)
        return _cart_to_bary(pts[idx])
    return _random_bary(rng)


def _symmetric_init(rng):
    """Symmetric initialization: 3 vertices + 3 edge midpoints + 3 near-centroid + 2 more."""
    verts = np.array([[0, 0], [1, 0], [0.5, S3H]])
    mids = np.array([[0.25, S3H * 0.25], [0.75, S3H * 0.25], [0.5, S3H * 0.5]])
    # 3 points near centroid, offset symmetrically
    cent = np.array([0.5, S3H / 3])
    r = 0.1
    inner = []
    for ang in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        inner.append([cent[0] + r * np.cos(ang), cent[1] + r * np.sin(ang)])
    # 2 more points
    extra = np.array([[0.3, S3H * 0.3], [0.7, S3H * 0.3]])
    all_pts = np.vstack([verts, mids, np.array(inner), extra])
    return _cart_to_bary(all_pts)


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    best_pts = None
    best_score = -float('inf')

    rng = np.random.RandomState(42)

    # Generate diverse initializations
    inits = []

    # 1. FPS
    inits.append(_fps_init(rng, 150))

    # 2. Grid
    inits.append(_grid_init(rng, 5))

    # 3. Symmetric
    inits.append(_symmetric_init(rng))

    # 4. Random
    for _ in range(4):
        inits.append(_random_bary(rng))

    # 5. FPS with different seed
    rng2 = np.random.RandomState(123)
    inits.append(_fps_init(rng2, 100))

    # 6. Another grid
    inits.append(_grid_init(rng2, 4))

    # 7. Vertices + random interior
    verts = np.array([[0, 0], [1, 0], [0.5, S3H]])
    interior = _random_cart(rng2)
    all_pts = np.vstack([verts, interior])
    inits.append(_cart_to_bary(all_pts))

    # 8. Another symmetric variant
    rng3 = np.random.RandomState(777)
    inits.append(_fps_init(rng3, 80))

    # Process each initialization
    for abc_init in inits:
        if elapsed() > DEADLINE - 15:
            break

        try:
            # Quick SA to diversify
            abc_sa, sa_score = _sa_bary(abc_init, n_iter=300, T0=0.02, T1=0.0005, rng=rng)

            if sa_score > best_score:
                best_score = sa_score
                best_pts = _bary_to_cart(abc_sa).copy()

            if elapsed() > DEADLINE - 12:
                break

            # Main gradient ascent
            abc_ga, ga_score = _gradient_ascent(abc_sa, max_iter=400)

            if ga_score > best_score:
                best_score = ga_score
                best_pts = _bary_to_cart(abc_ga).copy()

            if elapsed() > DEADLINE - 10:
                break

            # Second round of gradient ascent from slightly perturbed state
            abc_pert = abc_ga.copy()
            for _ in range(3):
                idx = rng.randint(N)
                abc_pert = _move_in_bary(abc_pert, idx, rng.normal(0, 0.01), rng.normal(0, 0.01))

            abc_ga2, ga2_score = _gradient_ascent(abc_pert, max_iter=300)

            if ga2_score > best_score:
                best_score = ga2_score
                best_pts = _bary_to_cart(abc_ga2).copy()

        except Exception:
            continue

    # Final polish: a few more gradient ascent rounds on the best
    if best_pts is not None and elapsed() < DEADLINE - 5:
        abc_best = _cart_to_bary(best_pts)
        for _ in range(3):
            if elapsed() > DEADLINE - 3:
                break
            try:
                abc_pol, pol_score = _gradient_ascent(abc_best, max_iter=200)
                if pol_score > best_score:
                    best_score = pol_score
                    best_pts = _bary_to_cart(abc_pol).copy()
                    abc_best = abc_pol
            except Exception:
                break

    if best_pts is None:
        best_pts = _bary_to_cart(_random_bary(rng))

    # Ensure all points are inside the triangle
    bary = _cart_to_bary(best_pts)
    bary = _project_bary(bary)
    best_pts = _bary_to_cart(bary)

    return best_pts
