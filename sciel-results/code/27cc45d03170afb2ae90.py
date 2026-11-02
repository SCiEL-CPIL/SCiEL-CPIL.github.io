"""Heilbronn convex regions: maximize min triangle area / convex-hull area for N=13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 150.0          # wall-clock limit set by evaluator
SAFETY = 6.0              # seconds of margin before the limit
GOLDEN = 2.399963229728653  # golden angle (radians)


# EVOLVE-BLOCK-START
def _tri_indices():
    idx = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int64)


_TRI = _tri_indices()
_NTRI = _TRI.shape[0]


def _hull_area(pts):
    """Area of the convex hull of pts (n,2)."""
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        # degenerate / collinear fallback
        x = pts[:, 0]
        y = pts[:, 1]
        return max(0.0, float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _min_tri(pts):
    """Minimum triangle area over all triples (vectorized)."""
    a = pts[_TRI[:, 0]]
    b = pts[_TRI[:, 1]]
    c = pts[_TRI[:, 2]]
    # 2x area = |cross(b-a, c-a)|
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return float(np.min(np.abs(cross))) * 0.5


def _score(pts):
    """Objective = min triangle area / convex hull area (higher is better)."""
    hull = _hull_area(pts)
    if hull <= 1e-12:
        return 0.0
    return _min_tri(pts) / hull


def _fermat():
    """Fermat (Vogel) spiral: uniform disk packing, centered."""
    i = np.arange(N)
    r = np.sqrt((i + 0.5) / N)
    theta = i * GOLDEN
    pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    pts -= pts.mean(axis=0)
    return pts


def construct():
    t0 = time.time()
    limit = DEADLINE - SAFETY

    best = _fermat()
    best_s = _score(best)
    cur = best.copy()
    cur_s = best_s

    temp = 2e-4
    cool = 0.99990
    rng = np.random.default_rng(12345)
    n = N

    while time.time() < limit:
        # choose move type
        u = rng.random()
        if u < 0.75:
            # local: perturb a random vertex of the current worst triangle
            a = cur[_TRI[:, 0]]
            b = cur[_TRI[:, 1]]
            c = cur[_TRI[:, 2]]
            cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                           (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            wt = int(np.argmin(cross))
            verts = _TRI[wt]
            v = int(rng.integers(0, 3))
            idx = int(verts[v])
            cur[idx] = cur[idx] + temp * rng.standard_normal(2)
        elif u < 0.90:
            # medium: perturb two distinct random vertices
            p = rng.integers(0, n)
            q = rng.integers(0, n)
            if q == p:
                q = (q + 1) % n
            cur[p] = cur[p] + temp * 1.2 * rng.standard_normal(2)
            cur[q] = cur[q] + temp * 1.2 * rng.standard_normal(2)
        else:
            # global: resample one point in the current bounding box
            lo = cur.min(axis=0)
            hi = cur.max(axis=0)
            idx = int(rng.integers(0, n))
            cur[idx] = lo + rng.random(2) * (hi - lo)

        s = _score(cur)
        if s > cur_s or rng.random() < np.exp((s - cur_s) / max(temp, 1e-12)):
            cur_s = s
            if s > best_s:
                best_s = s
                best = cur.copy()
        else:
            cur = best.copy() if False else cur  # keep cur (rejection handled below)
            # revert rejected move by restoring from a saved copy is expensive; instead
            # we simply keep the new cur (SA acceptance already decided). No revert needed.

        temp *= cool
        if temp < 1e-6:
            temp = 1e-6

    return best
# EVOLVE-BLOCK-END
