"""Circle packing in perimeter-4 rectangle: force-directed contact-graph solver."""
import time
import numpy as np

N = 21
EPS = 1e-7
DEADLINE = time.time() + 115.0  # safety margin under 120 s wall-clock

# Fixed, diverse initial configurations (centers in unit square, base radius).
def _make_configs():
    cfgs = []
    # 1) grid 5x5 (drop last 4)
    side = 5
    grid = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:N]
    cfgs.append((np.array(grid), 0.5 / side))
    # 2) jittered grid
    g = np.array(grid) + 0.03 * np.array([[np.sin(3 * i + 1), np.cos(2 * i + 2)] for i in range(N)])
    cfgs.append((g, 0.5 / side))
    # 3) 4x5 grid
    cols, rows = 4, 5
    pts = [((i + 0.5) / cols, (j + 0.5) / rows) for i in range(cols) for j in range(rows)][:N]
    cfgs.append((np.array(pts), min(0.5 / cols, 0.5 / rows)))
    # 4) 5x4 grid
    cols, rows = 5, 4
    pts = [((i + 0.5) / cols, (j + 0.5) / rows) for i in range(cols) for j in range(rows)][:N]
    cfgs.append((np.array(pts), min(0.5 / cols, 0.5 / rows)))
    # 5) 3x7 grid
    cols, rows = 3, 7
    pts = [((i + 0.5) / cols, (j + 0.5) / rows) for i in range(cols) for j in range(rows)][:N]
    cfgs.append((np.array(pts), min(0.5 / cols, 0.5 / rows)))
    # 6) 7x3 grid
    cols, rows = 7, 3
    pts = [((i + 0.5) / cols, (j + 0.5) / rows) for i in range(cols) for j in range(rows)][:N]
    cfgs.append((np.array(pts), min(0.5 / cols, 0.5 / rows)))
    # 7) 6x6 grid (drop 15)
    cols, rows = 6, 6
    pts = [((i + 0.5) / cols, (j + 0.5) / rows) for i in range(cols) for j in range(rows)][:N]
    cfgs.append((np.array(pts), 0.5 / cols))
    return cfgs


CONFIGS = _make_configs()


def _feasible(c, r):
    """Return (is_feasible, max_radius, rect_w, rect_h, perimeter)."""
    n = len(c)
    if n != N:
        return False, 0.0, 0.0, 0.0, 0.0
    minx = c[:, 0].min(); maxx = c[:, 0].max()
    miny = c[:, 1].min(); maxy = c[:, 1].max()
    w = maxx - minx; h = maxy - miny
    if w <= 0 or h <= 0:
        return False, 0.0, w, h, 0.0
    # rectangle bounds: each circle must fit inside [minx,maxx]x[miny,maxy]
    lb = np.maximum(r, r)  # placeholder
    # per-circle rectangle limit
    rect_limit = np.minimum(np.minimum(maxx - c[:, 0], c[:, 0] - minx),
                            np.minimum(maxy - c[:, 1], c[:, 1] - miny))
    # pairwise limits
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    pair_limit = (d / 2.0).min(axis=1)
    rmax = np.minimum(rect_limit, pair_limit).min()
    if not (np.isfinite(rmax) and rmax > 0):
        return False, 0.0, w, h, 0.0
    p = 2.0 * (w + h)
    return True, rmax, w, h, p


def _score(c, r):
    ok, rmax, w, h, p = _feasible(c, r)
    if not ok:
        return -1.0
    return p * r


def _solve_one(c0, r0):
    """Force-directed relaxation returning best feasible (c, r) and its score."""
    c = c0.copy().astype(float)
    r = np.full(N, r0, dtype=float)
    n = N
    # contact graph
    contact = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(c[i] - c[j]) < 1.05 * (r[i] + r[j]):
                contact[i, j] = contact[j, i] = True
    k_contact = 0.6
    k_rect = 1.0
    k_min = 10.0
    lr = 0.02
    best = (c.copy(), r.copy(), _score(c, r))
    max_iter = 4000
    it = 0
    while it < max_iter and time.time() < DEADLINE - 2.0:
        # update contact graph occasionally
        if it % 40 == 0:
            for i in range(n):
                for j in range(i + 1, n):
                    contact[i, j] = contact[j, i] = (
                        np.linalg.norm(c[i] - c[j]) < 1.05 * (r[i] + r[j]))
        force = np.zeros_like(c)
        # pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    d = np.array([1e-6, 1e-6]); dist = np.linalg.norm(d)
                overlap = (r[i] + r[j]) - dist
                if overlap > 0:
                    f = k_contact * overlap
                    force[i] += f * d / dist
                    force[j] -= f * d / dist
        # rectangle bounds
        minx = c[:, 0].min(); maxx = c[:, 0].max()
        miny = c[:, 1].min(); maxy = c[:, 1].max()
        for i in range(n):
            if c[i, 0] - minx < r[i]:
                force[i, 0] += k_rect * (r[i] - (c[i, 0] - minx))
            if maxx - c[i, 0] < r[i]:
                force[i, 0] -= k_rect * (r[i] - (maxx - c[i, 0]))
            if c[i, 1] - miny < r[i]:
                force[i, 1] += k_rect * (r[i] - (c[i, 1] - miny))
            if maxy - c[i, 1] < r[i]:
                force[i, 1] -= k_rect * (r[i] - (maxy - c[i, 1]))
        # update
        c = c + lr * force
        # grow radius slowly
        r = r + 0.0006
        # enforce minimum
        r = np.maximum(r, 0.005)
        s = _score(c, r)
        if s > best[2]:
            best = (c.copy(), r.copy(), s)
        it += 1
    return best


def construct():
    global DEADLINE
    best = (None, None, -1.0)
    for c0, r0 in CONFIGS:
        if time.time() > DEADLINE - 5.0:
            break
        c, r, s = _solve_one(c0, r0)
        if s > best[2]:
            best = (c, r, s)
    # final polish: pick best, then a short refinement on it
    if best[0] is None:
        # fallback to plain grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)][:N]
        return np.array(cells)
    c, r, s = best
    # light polish
    for _ in range(200):
        if time.time() > DEADLINE - 2.0:
            break
        # recompute max feasible radius and scale
        ok, rmax, w, h, p = _feasible(c, r)
        if ok and rmax > 0:
            r = np.full(N, rmax)
    # normalize: recompute r to exact feasible max and center in unit square
    ok, rmax, w, h, p = _feasible(c, r)
    if ok and rmax > 0:
        r = np.full(N, rmax)
    # recenter so bounding box starts at 0
    c = c - c.min(axis=0)
    out = np.zeros((N, 3))
    out[:, 0] = c[:, 0]
    out[:, 1] = c[:, 1]
    out[:, 2] = r
    return out


if __name__ == "__main__":
    arr = construct()
    print(arr.shape, arr[:3])
