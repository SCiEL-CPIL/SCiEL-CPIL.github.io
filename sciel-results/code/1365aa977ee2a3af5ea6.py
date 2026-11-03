"""Circle packing: 300 equal circles in unit disk. Ring-based + optimization."""
import numpy as np
import time

N = 300
T0 = time.time()
DEADLINE = 120.0
SAFETY = 10.0  # seconds margin

def _elapsed():
    return time.time() - T0

def _time_left():
    return DEADLINE - SAFETY - _elapsed()

def _score(pts):
    # returns (r, valid)
    if pts.shape != (N, 2):
        return 0.0, False
    if not np.all(np.isfinite(pts)):
        return 0.0, False
    dist_c = np.linalg.norm(pts, axis=1)
    if np.any(dist_c >= 1.0):
        return 0.0, False
    # pairwise min distance
    d = 1.0 - np.einsum('ij,ij->i', pts * pts)
    d2 = 1.0 - 2.0 * (pts @ pts.T) + np.einsum('ij,ij->i', pts * pts)[:, None] \
         + np.einsum('ij,ij->i', pts * pts)[None, :]
    np.fill_diagonal(d2, 1e18)
    d2 = np.maximum(d2, 0.0)
    min_d2 = d2.min()
    r = 0.5 * np.sqrt(min_d2)
    r = min(r, 1.0 - dist_c.max())
    if r <= 0:
        return 0.0, False
    return r, True

def _build_ring_structure():
    # ring counts summing to 299 (excluding center)
    ring_counts = [6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
    assert sum(ring_counts) == N - 1
    return ring_counts

def _make_points(radii, offsets):
    ring_counts = _build_ring_structure()
    pts = [np.zeros(2)]  # center
    for cnt, rad, off in zip(ring_counts, radii, offsets):
        if cnt == 0:
            continue
        ang = off + 2.0 * np.pi * np.arange(cnt) / cnt
        pts.append(np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]))
    return np.vstack(pts)

def _initial_radii():
    # estimate ring radii so circles roughly touch: ring k has radius ~ k * (2r)
    # use spacing ~ 0.1063 (2 * 0.05315)
    ring_counts = _build_ring_structure()
    spacing = 0.1063
    radii = []
    for k in range(1, len(ring_counts) + 1):
        radii.append(k * spacing)
    return np.array(radii)

def _initial_offsets():
    # stagger each ring by half a step relative to previous for better packing
    ring_counts = _build_ring_structure()
    offsets = []
    for k, cnt in enumerate(ring_counts):
        offsets.append((k % 2) * (np.pi / cnt))
    return np.array(offsets)

def _nelder_mead(f, x0, max_iter=400, tol=1e-7, time_budget=60.0):
    """Simple Nelder-Mead with time guard."""
    n = len(x0)
    simplex = [x0.astype(float)]
    for i in range(n):
        v = x0.copy()
        v[i] = v[i] + 0.1 if v[i] == 0 else v[i] * 1.05
        simplex.append(v)
    vals = [f(v) for v in simplex]
    best_iter = 0
    best_val = min(vals)
    start = time.time()
    it = 0
    while it < max_iter and (time.time() - start) < time_budget and _time_left() > 5.0:
        it += 1
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        # reflection
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            # expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe; vals[-1] = fe
            else:
                simplex[-1] = xr; vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr; vals[-1] = fr
        else:
            # contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc; vals[-1] = fc
            else:
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
        cur_best = min(vals)
        if cur_best > best_val:
            best_val = cur_best
            best_iter = it
            best_x = simplex[np.argmin(vals)].copy()
    return best_x

def _local_polish(pts, max_rounds=200, time_budget=30.0):
    """Coordinate-wise moves that increase the minimum constraint."""
    start = time.time()
    r, valid = _score(pts)
    if not valid:
        return pts
    pts = pts.copy()
    for rnd in range(max_rounds):
        if _time_left() < 5.0 or (time.time() - start) > time_budget:
            break
        improved = False
        for i in range(N):
            # try small random moves
            for _ in range(3):
                if _time_left() < 5.0 or (time.time() - start) > time_budget:
                    break
                step = 0.0005
                d = np.random.randn(2) * step
                new_pts = pts.copy()
                new_pts[i] = pts[i] + d
                nr, nv = _score(new_pts)
                if nv and nr > r + 1e-12:
                    pts = new_pts
                    r = nr
                    improved = True
        if not improved:
            break
    return pts

def construct():
    ring_counts = _build_ring_structure()
    n_rings = len(ring_counts)

    radii0 = _initial_radii()
    offsets0 = _initial_offsets()

    # combined parameter vector: radii + offsets
    x0 = np.concatenate([radii0, offsets0])

    def f(x):
        radii = x[:n_rings]
        offsets = x[n_rings:]
        pts = _make_points(radii, offsets)
        r, valid = _score(pts)
        if not valid:
            return -1.0
        return r

    # Nelder-Mead optimization
    if _time_left() > 20.0:
        x_opt = _nelder_mead(f, x0, max_iter=300, time_budget=min(60.0, max(10.0, _time_left() - 20.0)))
    else:
        x_opt = x0

    radii = x_opt[:n_rings]
    offsets = x_opt[n_rings:]
    pts = _make_points(radii, offsets)

    # local polish
    if _time_left() > 10.0:
        pts = _local_polish(pts, max_rounds=100, time_budget=min(30.0, max(5.0, _time_left() - 10.0)))

    # final validity check; fallback to hex lattice if invalid
    r, valid = _score(pts)
    if not valid or r <= 0:
        pts = _hex_fallback()
    return pts

def _hex_fallback():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

if __name__ == "__main__":
    res = construct()
    print("shape", res.shape, "score", _score(res))
