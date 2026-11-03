"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools


def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles. Returns array of length C(n,3)."""
    n = len(pts)
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            parts.append(0.5 * cross)
    if not parts:
        return np.array([0.0])
    return np.concatenate(parts)


def _min_triangle_area(pts):
    return float(np.min(_all_triangle_areas(pts)))


def _hull_area(pts):
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a


def _soft_min_area(pts, alpha):
    """
    Smooth surrogate for the minimum triangle area.
    soft_min = -log(sum(exp(-alpha * areas))) / alpha
    As alpha -> inf, this approaches the true min. Differentiable.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    # Gather all areas in chunks to keep memory bounded
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas_list.append(0.5 * cross)
    areas = np.concatenate(areas_list)
    if len(areas) == 0:
        return 0.0
    a_min = np.min(areas)
    shifted = -alpha * (areas - a_min)
    # log-sum-exp for stability
    m = np.max(shifted)
    log_sum = m + np.log(np.sum(np.exp(shifted - m)))
    return (a_min - log_sum / alpha)


def _neg_soft_score(params, alpha):
    pts = params.reshape(-1, 2)
    sm = _soft_min_area(pts, alpha)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e6
    return -(sm / ha)


def _concentric_init(radii, angles):
    """Build point set from list of (radius, [angles]) rings."""
    pts = []
    for r, angs in zip(radii, angles):
        for a in angs:
            pts.append([r * np.cos(a), r * np.sin(a)])
    return np.array(pts, dtype=float)


def _generate_initializations(rng):
    """Diverse structured + random initializations."""
    inits = []

    # 1. Pure regular polygon (13-gon)
    angs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angs), np.sin(angs)]))

    # 2. Golden-angle spiral (Fibonacci)
    k = np.arange(N)
    r = np.sqrt((k + 0.5) / N)
    th = k * np.pi * (3 - np.sqrt(5))
    inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    # 3. Concentric ring splits: outer ring + inner ring(s)
    # Try various splits of 13 points into rings
    splits = [(13, 0), (9, 4), (8, 5), (7, 6), (6, 7), (5, 8), (10, 3), (11, 2), (12, 1), (4, 9), (3, 10)]
    for (n_out, n_in) in splits:
        if n_out + n_in != N:
            continue
        if n_out == 0:
            angs = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
            inits.append(np.column_stack([np.cos(angs), np.sin(angs)]))
            continue
        angs_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
        angs_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + (np.pi / max(n_in, 1))
        r_in = 0.5
        outer = np.column_stack([np.cos(angs_out), np.sin(angs_out)])
        inner = np.column_stack([r_in * np.cos(angs_in), r_in * np.sin(angs_in)])
        inits.append(np.vstack([outer, inner]))

    # 4. Three-ring
    for (n1, n2, n3, r2, r3) in [(7, 4, 2, 0.5, 0.8), (6, 4, 3, 0.45, 0.75), (5, 5, 3, 0.5, 0.8), (9, 3, 1, 0.5, 0.8)]:
        if n1 + n2 + n3 != N:
            continue
        a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / max(n2, 1)
        a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / max(n3, 1)
        ring1 = np.column_stack([np.cos(a1), np.sin(a1)])
        ring2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        ring3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
        inits.append(np.vstack([ring1, ring2, ring3]))

    # 5. Jittered regular polygon
    for jitter in [0.02, 0.05, 0.1, 0.2]:
        angs = np.linspace(0, 2 * np.pi, N, endpoint=False) + jitter * rng.randn(N)
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angs), r * np.sin(angs)]))

    # 6. Random in unit disk (several)
    for _ in range(8):
        r = np.sqrt(rng.rand(N))
        th = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    # 7. Random on circle with random radii
    for _ in range(5):
        r = 0.3 + 0.7 * rng.rand(N)
        th = np.sort(rng.rand(N) * 2 * np.pi)
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    return inits


def _targeted_worst_triangle_opt(pts, deadline, rng):
    """
    Iteratively find the minimum-area triangle and push its vertices apart
    to increase that triangle's area, accepting only if the global min-area
    (and score) does not decrease. This handles the combinatorial 'which
    triangle is worst' structure that smooth gradients miss.
    """
    current = pts.copy()
    current_score = _score(current)

    for _ in range(3000):
        if time.time() > deadline:
            break
        areas = _all_triangle_areas(current)
        idx = int(np.argmin(areas))
        # Map linear index back to (i, j, k)
        # Build mapping once
        triples = list(itertools.combinations(range(len(current)), 3))
        if idx >= len(triples):
            break
        i, j, k = triples[idx]

        # Compute the triangle's centroid and push vertices away from centroid
        # to enlarge the triangle. Direction: each vertex moves along (v - centroid).
        ci = current[i]; cj = current[j]; ck = current[k]
        centroid = (ci + cj + ck) / 3.0

        # Try several step sizes; pick the best that improves global score
        best_candidate = None
        best_candidate_score = current_score
        for step in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            cand = current.copy()
            cand[i] = ci + step * (ci - centroid)
            cand[j] = cj + step * (cj - centroid)
            cand[k] = ck + step * (ck - centroid)
            s = _score(cand)
            if s > best_candidate_score + 1e-12:
                best_candidate = cand
                best_candidate_score = s
        if best_candidate is None:
            # No improvement found from this triangle; small random jitter to escape
            cand = current + 1e-4 * rng.randn(N, 2)
            s = _score(cand)
            if s > current_score + 1e-12:
                current = cand
                current_score = s
            else:
                # try a few more jitters
                improved = False
                for _ in range(5):
                    cand = current + 1e-3 * rng.randn(N, 2)
                    s = _score(cand)
                    if s > current_score + 1e-12:
                        current = cand
                        current_score = s
                        improved = True
                        break
                if not improved:
                    break
        else:
            current = best_candidate
            current_score = best_candidate_score

    return current, current_score


def _refine_local(pts, deadline):
    """
    Multi-stage local refinement:
    1. Soft-min surrogate optimization (Powell) with increasing alpha.
    2. Nelder-Mead on the exact objective.
    3. Targeted worst-triangle repair.
    """
    current = pts.copy()
    current_score = _score(current)

    # Stage 1: soft-min surrogate, anneal alpha
    for alpha in [10.0, 30.0, 80.0, 200.0]:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _neg_soft_score,
                current.flatten(),
                args=(alpha,),
                method='Powell',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 20000}
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > current_score:
                current = cand
                current_score = s
        except Exception:
            pass

    # Stage 2: Nelder-Mead on exact objective
    if time.time() < deadline:
        try:
            res = minimize(
                lambda p: -_score(p.reshape(-1, 2)),
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-15, 'maxfev': 100000, 'adaptive': True}
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > current_score:
                current = cand
                current_score = s
        except Exception:
            pass

    # Stage 3: targeted worst-triangle repair
    if time.time() < deadline:
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        current, current_score = _targeted_worst_triangle_opt(current, deadline, rng)

    return current, current_score


def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Use a 135s deadline for safety.
    deadline = start_time + 135.0

    rng = np.random.RandomState(12345)

    best_score = 0.0
    best_pts = None

    inits = _generate_initializations(rng)

    # Score all initializations
    scored = []
    for p in inits:
        if time.time() > deadline - 20.0:
            break
        s = _score(p)
        scored.append((s, p))
        if s > best_score:
            best_score = s
            best_pts = p.copy()

    scored.sort(key=lambda x: -x[0])

    # Refine top candidates
    seen = []
    for s, p in scored:
        if time.time() > deadline - 10.0:
            break
        # skip near-duplicates
        dup = False
        for sp in seen:
            if np.allclose(p, sp, atol=1e-4):
                dup = True
                break
        if dup:
            continue
        seen.append(p)
        refined, rs = _refine_local(p, deadline - 5.0)
        if rs > best_score:
            best_score = rs
            best_pts = refined.copy()

    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 2.0:
        rng2 = np.random.RandomState(777)
        polished, ps = _targeted_worst_triangle_opt(best_pts, deadline - 0.5, rng2)
        if ps > best_score:
            best_score = ps
            best_pts = polished.copy()

    if best_pts is None:
        # Fallback: regular polygon
        angs = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angs), np.sin(angs)])

    return best_pts
# EVOLVE-BLOCK-END
