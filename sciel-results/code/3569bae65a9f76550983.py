# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _hull_area(points: np.ndarray) -> float:
    """Area of the convex hull (0.0 on failure)."""
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _triangle_areas(points: np.ndarray) -> np.ndarray:
    """All C(n,3) triangle areas, vectorized. Returns 1-D array of length 286 for n=13."""
    n = points.shape[0]
    if n < 3:
        return np.zeros(0)
    ii, jj, kk = np.meshgrid(
        np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n), indexing='ij'
    )
    mask = (ii < jj) & (jj < kk)
    ii = ii[mask]; jj = jj[mask]; kk = kk[mask]
    pi = points[ii]; pj = points[jj]; pk = points[kk]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)


def _compute_score_fast(points: np.ndarray) -> float:
    """Exact metric A_min / A_hull."""
    n = points.shape[0]
    if n < 3 or not np.all(np.isfinite(points)):
        return 0.0
    A_hull = _hull_area(points)
    if A_hull <= 0:
        return 0.0
    areas = _triangle_areas(points)
    if areas.size == 0:
        return 0.0
    min_area = float(np.min(areas))
    if min_area <= 0:
        return 0.0
    return min_area / A_hull


def _softmin_score(points: np.ndarray, beta: float = 300.0) -> float:
    """Smooth surrogate: soft-min of triangle areas / hull area.

    softmin = -log(sum(exp(-a/beta)))/beta  ->  approaches min as beta -> inf.
    """
    n = points.shape[0]
    if n < 3 or not np.all(np.isfinite(points)):
        return 0.0
    A_hull = _hull_area(points)
    if A_hull <= 0:
        return 0.0
    areas = _triangle_areas(points)
    if areas.size == 0:
        return 0.0
    a = areas / beta
    m = np.min(a)
    s = np.sum(np.exp(-a + m))
    softmin = (m - np.log(s)) * beta
    return softmin / A_hull


def _worst_triple_and_grad(points: np.ndarray, beta: float = 300.0):
    """Return (worst_triple, grad) where grad[i] = d(softmin)/d(points[i]).

    For each triangle t = (i,j,k) with area a_t, weight w_t = exp(-(a_t-min)/beta).
    d(softmin)/d p_x = -sum_t w_t * (d a_t / d p_x) / beta.
    d a_t / d p_i = -0.5*(cross sign) * (perp of (pk-pj))  etc.
    """
    n = points.shape[0]
    ii, jj, kk = np.meshgrid(
        np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n), indexing='ij'
    )
    mask = (ii < jj) & (jj < kk)
    ii = ii[mask]; jj = jj[mask]; kk = kk[mask]

    pi = points[ii]; pj = points[jj]; pk = points[kk]
    # signed double-area
    sx = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(sx)
    m = np.min(areas)
    w = np.exp(-(areas - m) / beta)

    # d a / d p_i  = -0.5 * sign(sx) * ( (pk-pj)_y , -(pk-pj)_x )  ... careful
    # a = 0.5*|sx|.  d|sx|/dp = sign(sx)*dsx/dp.  a=0.5*|sx| => da/dp = 0.5*sign(sx)*dsx/dp.
    # dsx/dpi: sx = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    #   d/dxi = -(yk-yi) + (yj-yi) = yi - yk + yj - yi = yj - yk
    #   d/dyi = -(xk-xi) + (xj-xi) = xi - xk + xj - xi = xj - xk
    # dsx/dpj = (yk-yi) ; d/dyj = -(xk-xi) = xi - xk
    # dsx/dpk = -(yj-yi) ; d/dyk = (xj-xi)
    sign = np.sign(sx)
    c = 0.5 * sign

    d_ai_x = c * (pj[:, 1] - pk[:, 1])
    d_ai_y = c * (pk[:, 0] - pj[:, 0])
    d_aj_x = c * (pk[:, 1] - pi[:, 1])
    d_aj_y = c * (pi[:, 0] - pk[:, 0])
    d_ak_x = c * (pi[:, 1] - pj[:, 1])
    d_ak_y = c * (pj[:, 0] - pi[:, 0])

    grad = np.zeros((n, 2))
    w_over_beta = w / beta
    np.add.at(grad[:, 0], ii, -w_over_beta * d_ai_x)
    np.add.at(grad[:, 1], ii, -w_over_beta * d_ai_y)
    np.add.at(grad[:, 0], jj, -w_over_beta * d_aj_x)
    np.add.at(grid := grad, 0, 0) if False else None
    np.add.at(grad[:, 1], jj, -w_over_beta * d_aj_y)
    np.add.at(grad[:, 0], kk, -w_over_beta * d_ak_x)
    np.add.at(grad[:, 1], kk, -w_over_beta * d_ak_y)

    # worst triple
    t = int(np.argmin(areas))
    worst = (int(ii[t]), int(jj[t]), int(kk[t]))
    return worst, grad


def _regular_polygon_init(n: int) -> np.ndarray:
    """Regular n-gon centered at (0.5,0.5), radius ~0.4, inside [0,1]."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 0.4
    pts = np.column_stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)])
    return pts


def _hexagonal_initialization(n: int) -> np.ndarray:
    points = []
    spacing = 0.25
    row_height = spacing * np.sqrt(3) / 2
    row = 0
    while len(points) < n:
        y = 0.1 + row * row_height
        if row % 2 == 0:
            cols, x_start = 4, 0.1
        else:
            cols, x_start = 3, 0.1 + spacing / 2
        for col in range(cols):
            if len(points) >= n:
                break
            points.append([x_start + col * spacing, y])
        row += 1
    return np.array(points[:n])


def _concentric_initialization(n: int) -> np.ndarray:
    points = [[0.5, 0.5]]
    for r in [0.15, 0.3, 0.4]:
        num_pts = max(3, int(2 * np.pi * r / 0.22))
        for angle in np.linspace(0, 2 * np.pi, num_pts, endpoint=False):
            if len(points) >= n:
                break
            points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    return np.array(points[:n])


def _random_initialization(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.random((n, 2)) * 0.8 + 0.1


def _anneal(points: np.ndarray, deadline: float, beta: float = 300.0,
            T_init: float = 0.05, cooling: float = 0.9985) -> np.ndarray:
    """Anneal using smooth soft-min gradient + random steps; accept on exact score."""
    best = points.copy()
    cur = points.copy()
    best_score = _compute_score_fast(best)
    cur_score = best_score
    n = len(cur)
    step = 0.03
    lr = 0.01
    min_step = 1e-9
    acc = rej = 0

    for it in range(200000):
        if time.time() > deadline:
            break
        T = max(T_init * (cooling ** it), 1e-9)

        worst, grad = _worst_triple_and_grad(cur, beta)
        # normalize gradient direction
        gn = np.linalg.norm(grad)
        if gn > 1e-12:
            gdir = grad / gn
        else:
            gdir = np.zeros_like(grad)

        # candidate = gradient step + random jitter on a random point
        cand = cur.copy()
        cand += lr * gdir
        idx = np.random.randint(n)
        cand[idx, 0] += np.random.randn() * step
        cand[idx, 1] += np.random.randn() * step
        np.clip(cand, 0, 1, out=cand)

        c_score = _compute_score_fast(cand)
        if c_score >= cur_score:
            cur = cand; cur_score = c_score; acc += 1
            if c_score > best_score:
                best_score = c_score; best = cand.copy()
        else:
            delta = cur_score - c_score
            if np.random.random() < np.exp(-delta / max(T, 1e-12)):
                cur = cand; cur_score = c_score; acc += 1
            else:
                rej += 1

        if (acc + rej) % 200 == 0 and (acc + rej) > 0:
            rate = acc / (acc + rej)
            if rate < 0.25:
                step *= 0.8; lr *= 0.8
            elif rate > 0.75:
                step *= 1.15; lr *= 1.1
            step = max(min_step, min(step, 0.1)); lr = max(min_step, min(lr, 0.05))
            acc = rej = 0
    return best


def _local_search(points: np.ndarray, deadline: float) -> np.ndarray:
    """Deterministic grid-step polish on the worst triangle."""
    best = points.copy()
    best_score = _compute_score_fast(best)
    n = len(best)
    step = 0.02
    min_step = 1e-9
    stall = 0
    for it in range(60000):
        if time.time() > deadline:
            break
        areas = _triangle_areas(best)
        t = int(np.argmin(areas))
        ii, jj, kk = np.meshgrid(np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n), indexing='ij')
        m = (ii < jj) & (jj < kk)
        ii, jj, kk = ii[m], jj[m], kk[m]
        wi, wj, wk = int(ii[t]), int(jj[t]), int(kk[t])
        improved = False
        for idx in (wi, wj, wk):
            for dx in (-step, step):
                cand = best.copy(); cand[idx, 0] = np.clip(best[idx, 0] + dx, 0, 1)
                s = _compute_score_fast(cand)
                if s > best_score:
                    best = cand; best_score = s; improved = True
            for dy in (-step, step):
                cand = best.copy(); cand[idx, 1] = np.clip(best[idx, 1] + dy, 0, 1)
                s = _compute_score_fast(cand)
                if s > best_score:
                    best = cand; best_score = s; improved = True
        if not improved:
            step *= 0.85
            stall += 1
            if step < min_step or stall > 40:
                break
        else:
            stall = 0
    return best


def heilbronn_convex13() -> np.ndarray:
    n = 13
    start = time.time()
    deadline = start + 570  # safety margin before 630s kill

    inits = [
        _regular_polygon_init(n),
        _regular_polygon_init(n) + 0.02 * np.random.randn(n, 2),
        _hexagonal_initialization(n),
        _concentric_initialization(n),
    ]
    for s in range(8):
        inits.append(_random_initialization(n, s))

    # Phase 1: score and rank
    scored = []
    for cfg in inits:
        if time.time() > deadline - 300:
            break
        scored.append((_compute_score_fast(cfg), cfg))
    scored.sort(key=lambda x: -x[0])
    top = [c for _, c in scored[:6]]

    best = top[0].copy()
    best_score = _compute_score_fast(best)

    # Phase 2: anneal each top candidate (shared deadline)
    for cfg in top:
        if time.time() > deadline - 120:
            break
        res = _anneal(cfg, deadline - 120)
        s = _compute_score_fast(res)
        if s > best_score:
            best_score = s; best = res.copy()

    # Phase 3: high-precision polish
    if time.time() < deadline - 30:
        res = _local_search(best, deadline - 30)
        s = _compute_score_fast(res)
        if s > best_score:
            best_score = s; best = res.copy()

    # Phase 4: a couple of perturb-and-polish restarts if time remains
    if time.time() < deadline - 45:
        for _ in range(3):
            if time.time() > deadline - 30:
                break
            pert = best.copy() + 0.05 * np.random.randn(n, 2)
            np.clip(pert, 0, 1, out=pert)
            res = _local_search(pert, deadline - 30)
            s = _compute_score_fast(res)
            if s > best_score:
                best_score = s; best = res.copy()

    if best is None or not np.all(np.isfinite(best)):
        best = _regular_polygon_init(n)
    return best


# EVOLVE-BLOCK-END
