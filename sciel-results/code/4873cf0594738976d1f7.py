import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0
SAFETY = 8.0

def _hex_verts(cx, cy, ang):
    """Vertices of a unit hexagon centered at (cx,cy) with rotation ang."""
    a = ang + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(a), cy + np.sin(a)])

def _all_verts(centers, angs):
    """All 6*N vertices stacked."""
    out = np.empty((N, 6, 2))
    for i in range(N):
        out[i] = _hex_verts(centers[i, 0], centers[i, 1], angs[i])
    return out

def _outer_side_from_points(pts, outer_ang):
    """Min side length of regular hexagon (angle outer_ang) enclosing pts."""
    ka = np.arange(6) * math.pi / 3.0 + outer_ang
    normals = np.column_stack([np.cos(ka), np.sin(ka)])
    d = pts @ normals.T  # (M, 6)
    ad = np.abs(d)
    max_proj = ad.max(axis=0)  # (6,)
    m = max_proj.max()
    return (2.0 / math.sqrt(3.0)) * m

def _sat_overlap(v1, v2):
    """Return penetration depth if hexes v1,v2 (each 6x2) overlap, else 0."""
    min_ov = float('inf')
    for v in (v1, v2):
        for k in range(6):
            p1 = v[k]
            p2 = v[(k + 1) % 6]
            e = p2 - p1
            L = np.linalg.norm(e)
            if L < 1e-12:
                continue
            n = np.array([-e[1], e[0]]) / L
            d1 = v1 @ n
            d2 = v2 @ n
            ov = min(d1.max(), d2.max()) - max(d1.min(), d2.min())
            if ov <= 1e-9:
                return 0.0
            if ov < min_ov:
                min_ov = ov
    return min_ov

def _separation_penalty(verts):
    """Sum of squared penetrations over all pairs."""
    tot = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ov = _sat_overlap(verts[i], verts[j])
            tot += ov * ov
    return tot

def _objective(x, outer_ang, mu):
    """x = [c0x,c0y,a0, c1x,c1y,a1, ..., c10x,c10y,a10] (33 vars)."""
    centers = x[0::3].reshape(N, 2)
    angs = x[1::3].reshape(N)
    verts = _all_verts(centers, angs)
    pts = verts.reshape(-1, 2)
    S = _outer_side_from_points(pts, outer_ang)
    sep = _separation_penalty(verts)
    return S + mu * sep

def _refine(x0, outer_ang, deadline, mu_init=500.0):
    """Adaptive-penalty local optimization. Returns (best_x, best_S) or None."""
    t0 = time.time()
    mu = mu_init
    best_x = x0.copy()
    best_S = float('inf')
    for it in range(60):
        if time.time() > deadline - SAFETY:
            break
        res = minimize(
            _objective, best_x, args=(outer_ang, mu),
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        )
        cx = res.x
        centers = cx[0::3].reshape(N, 2)
        angs = cx[1::3].reshape(N)
        verts = _all_verts(centers, angs)
        sep = _separation_penalty(verts)
        pts = verts.reshape(-1, 2)
        S = _outer_side_from_points(pts, outer_ang)
        if sep < 1e-6:
            if S < best_S:
                best_S = S
                best_x = cx.copy()
            # feasible: try to tighten with lower mu
            mu *= 0.3
        else:
            mu *= 2.0
            if mu > 1e6:
                break
    return best_x, best_S

def _make_seed(outer_ang):
    """Seed: hexagonal close-packed cluster, centered, with small jitter."""
    rng = np.random.default_rng(0)
    axial = [
        (0, 0),
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ]
    d = math.sqrt(3.0)
    centers = np.empty((N, 2))
    for i, (q, r) in enumerate(axial):
        centers[i, 0] = d * (q + r / 2.0)
        centers[i, 1] = d * (math.sqrt(3.0) / 2.0) * r
    centers -= centers.mean(axis=0)
    # rotate cluster to align with outer hexagon
    R = np.array([[math.cos(outer_ang), -math.sin(outer_ang)],
                  [math.sin(outer_ang), math.cos(outer_ang)]])
    centers = centers @ R.T
    angs = np.zeros(N)
    x = np.empty(3 * N)
    x[0::3] = centers[:, 0]
    x[1::3] = centers[:, 1]
    x[2::3] = angs
    return x

def _make_seed_scatter(outer_ang, seed):
    """Random scattered seed within a disk."""
    rng = np.random.default_rng(seed)
    R0 = 3.0
    centers = np.column_stack([
        rng.uniform(-R0, R0, N),
        rng.uniform(-R0, R0, N),
    ])
    angs = rng.uniform(0, 2 * math.pi, N)
    x = np.empty(3 * N)
    x[0::3] = centers[:, 0]
    x[1::3] = centers[:, 1]
    x[2::3] = angs
    return x

def _make_seed_line(outer_ang, seed):
    """Line seed along x-axis."""
    rng = np.random.default_rng(seed)
    spacing = math.sqrt(3.0) + 0.01
    centers = np.empty((N, 2))
    for i in range(N):
        centers[i, 0] = (i - N // 2) * spacing
        centers[i, 1] = rng.normal(0, 0.05)
    centers -= centers.mean(axis=0)
    angs = rng.uniform(0, 2 * math.pi, N)
    x = np.empty(3 * N)
    x[0::3] = centers[:, 0]
    x[1::3] = centers[:, 1]
    x[2::3] = angs
    return x

def construct():
    start = time.time()
    deadline = start + DEADLINE

    outer_angles = [0.0, math.pi / 6, math.pi / 12, -math.pi / 6,
                    math.pi / 3, math.pi / 4, math.pi / 2, 0.1, -0.1]

    best_S = float('inf')
    best_x = None
    best_oa = 0.0

    # Phase 1: hexagonal cluster seeds for each outer angle
    for oa in outer_angles:
        if time.time() > deadline - SAFETY:
            break
        x0 = _make_seed(oa)
        res = _refine(x0, oa, deadline)
        if res is not None:
            bx, bS = res
            if bS < best_S:
                best_S = bS
                best_x = bx
                best_oa = oa

    # Phase 2: scattered seeds
    if time.time() < deadline - SAFETY:
        for seed_idx in range(12):
            if time.time() > deadline - SAFETY:
                break
            oa = outer_angles[seed_idx % len(outer_angles)]
            x0 = _make_seed_scatter(oa, seed_idx * 100 + 7)
            res = _refine(x0, oa, deadline)
            if res is not None:
                bx, bS = res
                if bS < best_S:
                    best_S = bS
                    best_x = bx
                    best_oa = oa

    # Phase 3: line seeds
    if time.time() < deadline - SAFETY:
        for seed_idx in range(6):
            if time.time() > deadline - SAFETY:
                break
            oa = outer_angles[(seed_idx + 3) % len(outer_angles)]
            x0 = _make_seed_line(oa, seed_idx * 200 + 13)
            res = _refine(x0, oa, deadline)
            if res is not None:
                bx, bS = res
                if bS < best_S:
                    best_S = bS
                    best_x = bx
                    best_oa = oa

    # Phase 4: final polish on best
    if best_x is not None and time.time() < deadline - SAFETY:
        # small local refinement
        res = _refine(best_x, best_oa, deadline, mu_init=1000.0)
        if res is not None:
            bx, bS = res
            if bS < best_S:
                best_S = bS
                best_x = bx

    if best_x is not None:
        centers = best_x[0::3].reshape(N, 2)
        angs = best_x[1::3].reshape(N)
        # Verify feasibility with tight tolerance
        verts = _all_verts(centers, angs)
        sep = _separation_penalty(verts)
        pts = verts.reshape(-1, 2)
        S_check = _outer_side_from_points(pts, best_oa)
        if sep > 1e-5:
            # Not feasible; fall back to a safe arrangement
            inner = [[(i - N // 2) * 3.5, 0.0, 0.0] for i in range(N)]
            return inner, (0.0, 0.0), 30.0, 0.0
        # Use the computed S (slightly padded for safety)
        S_final = S_check * 1.0001
        inner = [[float(centers[i, 0]), float(centers[i, 1]),
                  math.degrees(float(angs[i]))] for i in range(N)]
        return inner, (0.0, 0.0), float(S_final), math.degrees(float(best_oa))

    # Ultimate fallback: widely separated hexagons
    inner = [[(i - N // 2) * 4.0, 0.0, 0.0] for i in range(N)]
    return inner, (0.0, 0.0), 40.0, 0.0
