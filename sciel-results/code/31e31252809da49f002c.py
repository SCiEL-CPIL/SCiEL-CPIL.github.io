"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximise the common radius."""
import time
import numpy as np

N = 200
HALF = 0.5
DEADLINE_FRAC = 0.92          # stop searching at ~92% of the 150s wall-clock budget
TIME_BUDGET = 130.0           # hard internal cap (safety margin vs 150s kill)


def _score(pts):
    """Return (min pairwise distance, min distance to boundary)."""
    p = np.asarray(pts, dtype=np.float64)
    diff = p[:, None, :] - p[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = float(np.min(d))
    min_bound = float(np.min(HALF - np.abs(p)))
    return min_pair, min_bound


def _hex_init(n):
    """Hexagonal close-packed rows, centred, sized to fit inside the square."""
    rows = []
    r = 0.0
    while len(rows) < n:
        k = len(rows)
        count = n - len(rows)
        rows.append((k % 2, count))
        r += 1
    # build point list with hex spacing (unit spacing, will be scaled)
    pts = []
    for k, (offset, count) in enumerate(rows):
        y = k * np.sqrt(3.0) / 2.0
        xs = np.arange(count) + (0.5 if offset else 0.0)
        for x in xs:
            pts.append((x, y))
    pts = np.array(pts, dtype=np.float64)
    # centre on origin
    pts -= pts.mean(axis=0)
    # scale so it fits inside the square (with a small margin)
    extent = (pts.max(axis=0) - pts.min(axis=0)).max()
    if extent > 0:
        pts *= (0.92 * 2.0 * HALF) / extent
    return pts


def _repulsion_step(pts, step):
    """One force-directed step: pairwise repulsion + boundary attraction."""
    p = pts.copy()
    n = p.shape[0]
    diff = p[:, None, :] - p[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    # force magnitude ~ 1/d^2 along direction
    inv = 1.0 / (d2 * d)              # = 1/d^3
    inv[np.diag_indices(n)] = 0.0
    F = diff * inv[:, :, None]        # each point pushed away from others
    force = np.sum(F, axis=1)
    # boundary: push points toward centre proportional to how close they are to edge
    over = np.clip(np.abs(p) - HALF, 0.0, None)  # 0 inside, >0 outside
    # gentle inward pull everywhere scaled by proximity to wall
    prox = np.clip(1.0 - np.abs(p) / HALF, 0.0, None)  # 0 at wall, 1 at centre
    inward = -p * (0.5 * prox + 4.0 * over)[:, None]
    p = p + step * (force + inward)
    # clamp hard inside the square
    p = np.clip(p, -HALF + 1e-9, HALF - 1e-9)
    return p


def construct():
    t0 = time.time()
    deadline = t0 + TIME_BUDGET * DEADLINE_FRAC

    pts = _hex_init(N)
    best_pts = pts.copy()
    best_val = min(_score(pts))

    # adaptive step schedule: large early, decaying
    step = 0.05
    cool = 0.985
    it = 0
    while time.time() < deadline:
        pts = _repulsion_step(pts, step)
        it += 1
        # periodic evaluation (cheap enough every step for N=200)
        mp, mb = _score(pts)
        val = min(mp, mb)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
        step *= cool
        if step < 1e-5:
            break
        # small random kicks to escape shallow local minima
        if it % 250 == 0:
            best_pts = best_pts + np.random.randn(*best_pts.shape) * (best_val * 0.02)
            best_pts = np.clip(best_pts, -HALF + 1e-9, HALF - 1e-9)
            pts = best_pts.copy()
            step = max(step, 0.02)

    # final polish from the best point with tiny steps
    if time.time() < deadline:
        p = best_pts.copy()
        s = 1e-4
        for _ in range(4000):
            if time.time() >= deadline:
                break
            p = _repulsion_step(p, s)
            s *= 0.999
            mp, mb = _score(p)
            if min(mp, mb) > best_val:
                best_val = min(mp, mb)
                best_pts = p.copy()

    return best_pts
