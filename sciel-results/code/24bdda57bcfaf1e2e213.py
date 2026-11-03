"""Pack 50 equal circles in a unit circle, maximizing the common radius."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
_DEADLINE = 150.0  # evaluator kills at 150s wall-clock; we use a safety margin
_MARGIN = 5.0      # seconds of safety margin


def _score(pts):
    """Return (r, feasible) for a placement."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0, False
    if not np.all(np.isfinite(pts)):
        return 0.0, False
    # boundary clearance
    rad = np.linalg.norm(pts, axis=1)
    if np.any(rad >= 1.0):
        return 0.0, False
    boundary = 1.0 - rad
    bmin = float(boundary.min())
    if bmin <= 0.0:
        return 0.0, False
    # pair distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(2.0, output_type="ndarray")
    if len(pairs) == 0:
        pmin = 1e9
    else:
        d = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
        pmin = float(d.min())
    r = 0.5 * min(pmin, bmin * 2.0)
    return r, True


def _init_hex():
    """Hexagonal lattice clipped to disk, pick N best by radius (center-first)."""
    best = None
    for spacing in np.linspace(0.20, 0.10, 40):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0 - 1e-9
        pts = pts[keep]
        if len(pts) >= N:
            # prefer points closer to center for symmetry
            idx = np.argsort(rad, kind="stable")[:N]
            cand = pts[idx]
            r, ok = _score(cand)
            if ok and (best is None or r > best[0]):
                best = (r, cand.copy())
    if best is None:
        # fallback: random
        ang = np.random.uniform(0, 2 * np.pi, N)
        rad = np.random.uniform(0, 0.8, N)
        pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        return pts
    return best[1]


def _init_rings():
    """Concentric rings with staggered angles."""
    # Try a few ring configurations
    configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 18, 13],
        [1, 5, 10, 15, 19],
        [1, 6, 12, 18, 13],
        [0, 6, 12, 18, 14],
    ]
    best = None
    for cfg in configs:
        pts_list = []
        total = 0
        for ring_i, cnt in enumerate(cfg):
            if cnt == 0:
                continue
            if cnt == 1 and ring_i == 0:
                pts_list.append(np.array([0.0, 0.0]))
                total += 1
                continue
            # radius of ring: scale so outer ring near boundary
            r_ring = 0.85 * (ring_i + 1) / len(cfg)
            ang_off = np.random.uniform(0, 2 * np.pi)
            angs = ang_off + 2 * np.pi * np.arange(cnt) / cnt
            pts_list.append(np.column_stack([r_ring * np.cos(angs), r_ring * np.sin(angs)]))
            total += cnt
        if total != N:
            continue
        pts = np.vstack(pts_list)
        r, ok = _score(pts)
        if ok and (best is None or r > best[0]):
            best = (r, pts.copy())
    if best is None:
        return _init_hex()
    return best[1]


def _init_random():
    ang = np.random.uniform(0, 2 * np.pi, N)
    rad = np.random.uniform(0, 0.85, N)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


def _init_seed():
    """A hand-tuned multi-ring seed with decent symmetry."""
    pts_list = []
    # center
    pts_list.append(np.array([0.0, 0.0]))
    # ring 1: 6
    r1 = 0.22
    a1 = 2 * np.pi * np.arange(6) / 6
    pts_list.append(np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)]))
    # ring 2: 12
    r2 = 0.42
    a2 = 2 * np.pi * np.arange(12) / 12 + np.pi / 12
    pts_list.append(np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]))
    # ring 3: 18
    r3 = 0.62
    a3 = 2 * np.pi * np.arange(18) / 18
    pts_list.append(np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]))
    # ring 4: 13
    r4 = 0.82
    a4 = 2 * np.pi * np.arange(13) / 13 + np.pi / 13
    pts_list.append(np.column_stack([r4 * np.cos(a4), r4 * np.sin(a4)]))
    pts = np.vstack(pts_list)
    return pts


def _local_search(pts, target_r, t_start, max_iters=40000):
    """Annealed repulsion + boundary push. Returns improved pts."""
    pts = pts.copy()
    n = len(pts)
    step = 0.02
    temp = 0.05
    best_r, _ = _score(pts)
    best_pts = pts.copy()

    for it in range(max_iters):
        if time.time() > t_start + _DEADLINE - _MARGIN:
            break
        if it % 2000 == 0 and it > 0:
            temp *= 0.95
            step *= 0.98

        # Compute forces
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0, output_type="ndarray")
        forces = np.zeros_like(pts)
        if len(pairs) > 0:
            for a, b in pairs:
                d = pts[a] - pts[b]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    d = np.random.randn(2)
                    dist = np.linalg.norm(d)
                if dist < 2.0 * target_r:
                    # repel
                    f = (2.0 * target_r - dist) / dist
                    forces[a] += f * d
                    forces[b] -= f * d

        # Boundary push: if too close to boundary, push inward
        rad = np.linalg.norm(pts, axis=1)
        clearance = 1.0 - rad
        for i in range(n):
            if clearance[i] < target_r:
                if rad[i] < 1e-12:
                    continue
                inward = -pts[i] / rad[i]
                forces[i] += (target_r - clearance[i]) * inward

        # Random jitter
        forces += np.random.randn(n, 2) * temp

        # Move
        pts_new = pts + step * forces
        # Clamp inside
        rad_new = np.linalg.norm(pts_new, axis=1)
        for i in range(n):
            if rad_new[i] > 1.0 - target_r * 0.5:
                scale = (1.0 - target_r * 0.5) / rad_new[i]
                pts_new[i] *= scale

        r_new, ok = _score(pts_new)
        if ok and r_new >= best_r - 0.01:
            pts = pts_new
            if r_new > best_r:
                best_r = r_new
                best_pts = pts_new.copy()
    return best_pts


def construct():
    t_start = time.time()
    np.random.seed(42)

    # Generate initial candidates
    inits = []
    inits.append(_init_hex())
    inits.append(_init_rings())
    inits.append(_init_seed())
    inits.append(_init_random())
    inits.append(_init_random())

    # Score and pick best initial
    best_r = 0.0
    best_pts = None
    for p in inits:
        r, ok = _score(p)
        if ok and r > best_r:
            best_r = r
            best_pts = p.copy()

    if best_pts is None:
        # Fallback: simple random
        ang = np.random.uniform(0, 2 * np.pi, N)
        rad = np.random.uniform(0, 0.7, N)
        best_pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        best_r, _ = _score(best_pts)

    # Local search on the best initial
    target_r = best_r * 1.05
    best_pts = _local_search(best_pts, target_r, t_start, max_iters=60000)
    best_r, _ = _score(best_pts)

    # Try a couple more multi-start rounds if time allows
    for _ in range(4):
        if time.time() > t_start + _DEADLINE - _MARGIN - 10.0:
            break
        p = _init_random()
        # Quick local search
        r0, ok = _score(p)
        if not ok:
            continue
        p2 = _local_search(p, r0 * 1.05, t_start, max_iters=15000)
        r2, ok2 = _score(p2)
        if ok2 and r2 > best_r:
            best_r = r2
            best_pts = p2.copy()

    # Final polish on the best
    if time.time() < t_start + _DEADLINE - _MARGIN - 2.0:
        best_pts = _local_search(best_pts, best_r * 1.02, t_start, max_iters=20000)
        r_final, ok = _score(best_pts)
        if ok:
            best_r = r_final

    # Ensure validity: clamp strictly inside
    rad = np.linalg.norm(best_pts, axis=1)
    for i in range(N):
        if rad[i] > 1.0 - 1e-6:
            best_pts[i] *= (1.0 - 1e-6) / rad[i]

    return best_pts
