# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.spatial import ConvexHull


def _triangle_areas(pts):
    """Vectorized area of all C(n,3) triangles. Returns (areas, triples)."""
    n = pts.shape[0]
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing="ij")
    mask = (i < j) & (j < k)
    ii, jj, kk = i[mask], j[mask], k[mask]
    a = pts[ii]
    b = pts[jj]
    c = pts[kk]
    # 2*area = |cross(b-a, c-a)|
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = np.abs(cross) * 0.5
    triples = np.stack([ii, jj, kk], axis=1)
    return areas, triples


def _hull_area(pts):
    try:
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0


def _score(pts):
    areas, _ = _triangle_areas(pts)
    amin = float(areas.min())
    if amin <= 1e-15:
        return 0.0
    ahull = _hull_area(pts)
    if ahull <= 1e-15:
        return 0.0
    return amin / ahull


def _repulsion_step(pts, areas, triples, k_worst, step, rng):
    """Move vertices of the k_worst smallest triangles apart, with jitter."""
    n = pts.shape[0]
    order = np.argsort(areas)[:k_worst]
    # Build per-point net displacement
    disp = np.zeros_like(pts)
    for t in triples[order]:
        i, j, k = int(t[0]), int(t[1]), int(t[2])
        pi, pj, pk = pts[i], pts[j], pts[k]
        # For each pair in the triangle, push them apart
        for (a, b) in [(i, j), (i, k), (j, k)]:
            pa, pb = pts[a], pts[b]
            d = pa - pb
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d = d / norm
            else:
                d = rng.normal(size=2)
                d /= (np.linalg.norm(d) + 1e-12)
            # Push a away from b, and b away from a
            disp[a] += d
            disp[b] -= d
    # Add small jitter to all points to escape plateaus
    disp += rng.normal(scale=0.005, size=pts.shape)
    return pts + step * disp


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the
    area of the smallest triangle formed by these points.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    t0 = time.time()
    deadline = t0 + 580.0  # safety margin before 600s kill

    best_pts = None
    best_score = -1.0

    rng = np.random.default_rng(seed=12345)

    def run_one(start_pts, budget):
        nonlocal best_pts, best_score
        pts = start_pts.copy()
        step = 0.02
        k_worst = 6
        it = 0
        while time.time() < budget:
            areas, triples = _triangle_areas(pts)
            cur = float(areas.min())
            if cur > 1e-15:
                ahull = _hull_area(pts)
                if ahull > 1e-15:
                    s = cur / ahull
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
            # Adaptive step: shrink if we're making progress, grow if stuck
            pts_new = _repulsion_step(pts, areas, triples, k_worst, step, rng)
            # Accept if min area improves, otherwise occasionally accept for exploration
            areas_new, _ = _triangle_areas(pts_new)
            if areas_new.min() >= areas.min() or rng.random() < 0.15:
                pts = pts_new
                step = max(0.002, step * 0.995)
            else:
                # Reject, reduce step
                step = max(0.002, step * 0.85)
            it += 1
            if it % 200 == 0:
                # Occasionally re-jump to escape local minima
                if rng.random() < 0.05:
                    pts = pts + rng.normal(scale=0.05, size=pts.shape)
        return best_pts, best_score

    # Phase 1: several random restarts with repulsion optimization
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 60:
            break
        # Start from a random point in unit square, then quickly normalize
        start = rng.random((n, 2))
        # Quick initial spread: a few repulsion steps with larger step
        for _ in range(30):
            if time.time() > deadline - 60:
                break
            areas, triples = _triangle_areas(start)
            start = _repulsion_step(start, areas, triples, 10, 0.05, rng)
        # Run optimization with a per-restart budget
        per_budget = deadline - 60 - (n_restarts - 1 - r) * 5
        per_budget = max(per_budget, time.time() + 5)
        run_one(start, min(per_budget, deadline - 60))

    # Phase 2: final polish on the best result with very small steps
    if best_pts is not None and time.time() < deadline - 5:
        pts = best_pts.copy()
        step = 0.003
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            areas, triples = _triangle_areas(pts)
            cur = float(areas.min())
            if cur > 1e-15:
                ahull = _hull_area(pts)
                if ahull > 1e-15:
                    s = cur / ahull
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
            pts_new = _repulsion_step(pts, areas, triples, 4, step, rng)
            areas_new, _ = _triangle_areas(pts_new)
            if areas_new.min() >= areas.min():
                pts = pts_new
            else:
                step = max(0.0005, step * 0.9)

    if best_pts is None:
        # Fallback: random points (should never happen)
        best_pts = rng.random((n, 2))

    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.nan_to_num(best_pts, nan=0.5, posinf=1.0, neginf=0.0)

    # Ensure convex hull works (non-collinear)
    if _hull_area(best_pts) <= 1e-15:
        # Perturb slightly
        best_pts = best_pts + rng.normal(scale=0.01, size=best_pts.shape)

    return best_pts


# EVOLVE-BLOCK-END
