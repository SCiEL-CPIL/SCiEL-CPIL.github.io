# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13
# Precompute all 286 triangle index triples
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """Hard Heilbronn score: A_min / A_hull."""
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas) / A_hull

def soft_min_score(points, temperature=0.005):
    """Smooth surrogate: soft-min of triangle areas / hull area (to maximize)."""
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 1e10
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    if A_hull <= 1e-12:
        return 1e10
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    return -soft_min_area / A_hull

def compute_areas(pts):
    """Vectorized areas of all 286 triangles."""
    p1 = pts[TRI_IDX[:, 0]]
    p2 = pts[TRI_IDX[:, 1]]
    p3 = pts[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5, cross

def soft_min_gradient(pts, temperature):
    """
    Analytic gradient of the soft-minimum surrogate w.r.t. point coords.
    Returns (value, grad) where value = soft_min(areas)/A_hull (to maximize)
    and grad points in the direction of increase.
    """
    try:
        hull = ConvexHull(pts)
        A_hull = hull.volume
    except Exception:
        return None
    if A_hull <= 1e-12:
        return None
    areas, cross = compute_areas(pts)
    if not np.all(np.isfinite(areas)):
        return None
    # Soft-min weights (log-sum-exp stable)
    scaled = -areas / temperature
    m = np.max(scaled)
    w = np.exp(scaled - m)
    s = np.sum(w)
    if s <= 0 or not np.isfinite(s):
        return None
    w = w / s
    # soft_min = -temperature * log(s) + temperature * m
    value = (temperature * m - temperature * np.log(s)) / A_hull
    # d(soft_min)/d(pts) = (1/A_hull) * sum_i w_i * d(areas_i)/d(pts)
    # For areas_i = 0.5*|cross_i|, use sign(cross_i) to keep it smooth.
    sign = np.sign(cross)
    coef = w * sign / A_hull  # per-triangle scalar multiplier for d(cross)
    i = TRI_IDX[:, 0]
    j = TRI_IDX[:, 1]
    k = TRI_IDX[:, 2]
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    dx_pj = pj[:, 0] - pi[:, 0]
    dy_pj = pj[:, 1] - pi[:, 1]
    dx_pk = pk[:, 0] - pi[:, 0]
    dy_pk = pk[:, 1] - pi[:, 1]
    # d(cross)/d(pi) = (dy_pj - dy_pk, dx_pk - dx_pj)
    # d(cross)/d(pj) = (dy_pk, -dx_pk)
    # d(cross)/d(pk) = (-dy_pj, dx_pj)
    g = np.zeros_like(pts)
    np.add.at(g, i, coef[:, None] * np.column_stack([dy_pj - dy_pk, dx_pk - dx_pj]))
    np.add.at(g, j, coef[:, None] * np.column_stack([dy_pk, -dx_pk]))
    np.add.at(g, k, coef[:, None] * np.column_stack([-dy_pj, dx_pj]))
    return value, g

def gradient_ascent(pts, deadline, beta_start=8.0, beta_end=600.0,
                    step_start=0.03, step_end=0.0015, n_steps=500):
    """
    Annealed gradient ascent on the soft-min surrogate.
    Low beta (smooth, many triangles contribute) -> high beta (sharp,
    focuses on the true minimum triangle).
    """
    p = pts.copy()
    best_p = p.copy()
    best_val = soft_min_score(p, temperature=1.0 / beta_end)
    for s in range(n_steps):
        if time.time() > deadline:
            break
        prog = s / max(n_steps - 1, 1)
        beta = beta_start + (beta_end - beta_start) * prog
        temperature = 1.0 / beta
        step = step_start + (step_end - step_start) * prog
        res = soft_min_gradient(p, temperature)
        if res is None:
            break
        val, grad = res
        nrm = np.linalg.norm(grad)
        if nrm < 1e-14:
            break
        p = p + step * (grad / nrm) * nrm  # == p + step*grad, but keep scale
        # simpler: p = p + step * grad
        p = p + step * grad
        if not np.all(np.isfinite(p)):
            break
        # Track best in the (final) sharp metric occasionally
        if s % 20 == 0 or s == n_steps - 1:
            cv = soft_min_score(p, temperature=1.0 / beta_end)
            if cv > best_val:
                best_val = cv
                best_p = p.copy()
    return best_p

def local_search(points, time_limit, rng):
    """Coordinate ascent with decreasing step sizes (hard score)."""
    pts = points.copy()
    current_score = compute_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    step_size = 0.12
    while time.time() < time_limit and step_size > 1e-7:
        improved = False
        for idx in range(N):
            for _ in range(2):
                direction = rng.choice([-1, 1])
                dx = direction * step_size * rng.uniform(0.5, 1.0)
                dy = direction * step_size * rng.uniform(0.5, 1.0)
                new_pts = pts.copy()
                new_pts[idx, 0] += dx
                new_pts[idx, 1] += dy
                new_score = compute_score(new_pts)
                if new_score > current_score:
                    pts = new_pts
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
        if not improved:
            step_size *= 0.8
        else:
            step_size *= 0.98
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    rng = np.random.default_rng(42)

    best_points = None
    best_score = -1.0

    candidates = []
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    # 2. 7 outer, 6 inner
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p7 = np.column_stack([np.cos(ang7), np.sin(ang7)])
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    p6 = np.column_stack([0.55 * np.cos(ang6), 0.55 * np.sin(ang6)])
    candidates.append(np.vstack([p7, p6]))
    # 3. 7 outer, 6 inner (offset)
    ang6b = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 12
    p6b = np.column_stack([0.6 * np.cos(ang6b), 0.6 * np.sin(ang6b)])
    candidates.append(np.vstack([p7, p6b]))
    # 4. 5/5/3
    ang5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5o = np.column_stack([np.cos(ang5), np.sin(ang5)])
    ang5m = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5m = np.column_stack([0.6 * np.cos(ang5m), 0.6 * np.sin(ang5m)])
    ang3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
    p3i = np.column_stack([0.3 * np.cos(ang3), 0.3 * np.sin(ang3)])
    candidates.append(np.vstack([p5o, p5m, p3i]))
    # 5. 4/5/4
    ang4o = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4o = np.column_stack([np.cos(ang4o), np.sin(ang4o)])
    ang5m2 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5m2 = np.column_stack([0.65 * np.cos(ang5m2), 0.65 * np.sin(ang5m2)])
    ang4i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i = np.column_stack([0.35 * np.cos(ang4i), 0.35 * np.sin(ang4i)])
    candidates.append(np.vstack([p4o, p5m2, p4i]))
    # 6. 6 outer, 7 inner
    ang6o = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p6o = np.column_stack([np.cos(ang6o), np.sin(ang6o)])
    ang7i = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    p7i = np.column_stack([0.5 * np.cos(ang7i), 0.5 * np.sin(ang7i)])
    candidates.append(np.vstack([p6o, p7i]))
    # 7. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(N) / (golden ** 2)
    fib_radii = np.sqrt(np.arange(N) / N)
    candidates.append(np.column_stack([fib_radii * np.cos(fib_angles),
                                       fib_radii * np.sin(fib_angles)]))
    # 8. 4/5/4
    ang4a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4a = np.column_stack([np.cos(ang4a), np.sin(ang4a)])
    ang5b = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5b = np.column_stack([0.65 * np.cos(ang5b), 0.65 * np.sin(ang5b)])
    ang4c = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4c = np.column_stack([0.3 * np.cos(ang4c), 0.3 * np.sin(ang4c)])
    candidates.append(np.vstack([p4a, p5b, p4c]))
    # 9. 5/4/4
    ang5o2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 10
    p5o2 = np.column_stack([np.cos(ang5o2), np.sin(ang5o2)])
    ang4m2 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4m2 = np.column_stack([0.6 * np.cos(ang4m2), 0.6 * np.sin(ang4m2)])
    ang4i2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i2 = np.column_stack([0.3 * np.cos(ang4i2), 0.3 * np.sin(ang4i2)])
    candidates.append(np.vstack([p5o2, p4m2, p4i2]))
    # 10. Random
    for _ in range(5):
        candidates.append(rng.random((N, 2)) * 2 - 1)
    # Perturbed structured
    for base_idx in [0, 1, 3, 4]:
        if base_idx < len(candidates):
            candidates.append(candidates[base_idx] + rng.normal(0, 0.03, size=(N, 2)))

    n_cand = len(candidates)
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline - 5:
            break
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()

        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        remaining_cands = n_cand - i
        time_per_cand = min(remaining / max(remaining_cands, 1), 90.0)
        cand_deadline = time.time() + time_per_cand

        cur = init_pts.copy()
        # Phase 1: short coordinate warm-up
        if time.time() < cand_deadline - 5:
            ls_pts, ls_score = local_search(cur, cand_deadline - 5, rng)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_pts.copy()
            cur = ls_pts
        # Phase 2: gradient-ascent anneal (the new refinement engine)
        if time.time() < deadline - 10:
            ga_deadline = min(cand_deadline, deadline - 10)
            ga_pts = gradient_ascent(cur, ga_deadline)
            ga_score = compute_score(ga_pts)
            if ga_score > best_score:
                best_score = ga_score
                best_points = ga_pts.copy()
                cur = ga_pts
        # Phase 3: fine coordinate polish after gradient ascent
        if time.time() < deadline - 8:
            fine_deadline = min(cand_deadline, deadline - 8)
            fs_pts, fs_score = local_search(cur, fine_deadline, rng)
            if fs_score > best_score:
                best_score = fs_score
                best_points = fs_pts.copy()
                cur = fs_pts

    # Final polish if time remains
    if best_points is not None and time.time() < deadline - 3:
        ga_pts = gradient_ascent(best_points, deadline - 3,
                                 beta_start=20.0, beta_end=800.0,
                                 step_start=0.01, step_end=0.0008, n_steps=300)
        ga_score = compute_score(ga_pts)
        if ga_score > best_score:
            best_score = ga_score
            best_points = ga_pts.copy()
        if time.time() < deadline - 1:
            fs_pts, fs_score = local_search(best_points, deadline - 1, rng)
            if fs_score > best_score:
                best_score = fs_score
                best_points = fs_pts.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
# EVOLVE-BLOCK-END
