"""Structural redesign: physics-based packing of 12 unit hexagons inside a regular hexagon."""
import time
import math
import numpy as np

N = 12
_TIME_LIMIT = 120.0

# Vertices of a unit regular hexagon (circumradius = 1), angle measured in degrees.
_UNIT_HEX = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
# 6 fixed outward normals of a regular hexagon (invariant under rotation), pointing to edge midpoints.
_NORMALS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 390, 60)])


def _hex_vertices(x, y, ang_deg):
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return x + _UNIT_HEX @ R.T


def _hex_widths(v):
    """Width of a hexagon along each of the 6 fixed normals."""
    return np.max(v @ _NORMALS.T, axis=1) - np.min(v @ _NORMALS.T, axis=1)


def _outer_size(v):
    """Minimal regular-hexagon side length (any rotation) that contains polygon v (centered at 0)."""
    # For a given hexagon rotation, required side = max over normals of (max projection + width/2).
    # Optimize over rotation; the objective is 1-periodic in 30 deg so use a fine grid.
    best = float('inf')
    for k in range(0, 60):
        th = math.radians(k)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        vn = v @ R.T
        proj = vn @ _NORMALS.T
        w = proj.max(axis=1) - proj.min(axis=1)
        size = np.max(proj.max(axis=1) + 0.5 * w)
        if size < best:
            best = size
    return best


def _outer_size_coarse(v):
    best = float('inf')
    for k in range(0, 12):
        th = math.radians(k * 5.0)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        vn = v @ R.T
        proj = vn @ _NORMALS.T
        w = proj.max(axis=1) - proj.min(axis=1)
        size = np.max(proj.max(axis=1) + 0.5 * w)
        if size < best:
            best = size
    return best


def _build_initial(seed):
    rng = np.random.default_rng(seed)
    # Base hexagonal ring packing (center + 6 inner + 6 outer = 13), drop one random hexagon -> 12.
    base = [(0.0, 0.0, 0.0)]
    for a in range(0, 360, 60):
        r = math.radians(a)
        base.append((2.0 * math.cos(r), 2.0 * math.sin(r), 0.0))
        base.append((2.0 * math.cos(r + math.radians(60)), 2.0 * math.sin(r + math.radians(60)), 0.0))
    drop = int(rng.integers(1, len(base)))
    pts = base[:drop] + base[drop + 1:]
    pos = np.array([(p[0], p[1]) for p in pts], dtype=float)
    ang = np.array([p[2] for p in pts], dtype=float)
    pos += 0.02 * rng.standard_normal((N, 2))
    ang += 0.02 * rng.standard_normal(N)
    return pos, ang


def _evaluate(pos, ang):
    verts = [_hex_vertices(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
    # pairwise minimum distance (collision check)
    min_gap = float('inf')
    for i in range(N):
        vi = verts[i]
        for j in range(i + 1, N):
            vj = verts[j]
            d = np.min(np.sqrt(((vi[:, None, :] - vj[None, :, :]) ** 2).sum(axis=2)))
            if d < min_gap:
                min_gap = d
    allv = np.vstack(verts)
    size = _outer_size_coarse(allv)
    return min_gap, size


def _physics_iterate(pos, ang, n_iters, t0, deadline):
    pos = pos.copy()
    ang = ang.copy()
    dt = 0.05
    for it in range(n_iters):
        if time.time() > deadline:
            break
        # forces
        fp = np.zeros_like(pos)
        fa = np.zeros_like(ang)
        for i in range(N):
            for j in range(i + 1, N):
                d = pos[i] - pos[j]
                dist = float(np.linalg.norm(d))
                if dist < 1e-6:
                    d = np.array([1e-6 * (i - j), 1e-6])
                    dist = 1e-6
                # repulsion when too close (center distance < 2.0 means likely overlap)
                if dist < 2.0:
                    f = 8.0 * (2.0 - dist) / dist
                    fp[i] += f * d
                    fp[j] -= f * d
        # containment: pull each hexagon inward if its vertex exceeds a soft radius
        Rsoft = 3.5
        for i in range(N):
            v = _hex_vertices(pos[i, 0], pos[i, 1], ang[i])
            rmax = float(np.max(np.linalg.norm(v, axis=1)))
            if rmax > Rsoft:
                center_r = float(np.linalg.norm(pos[i]))
                if center_r > 1e-6:
                    fp[i] -= 2.0 * (rmax - Rsoft) * (pos[i] / center_r)
                else:
                    fp[i] -= 2.0 * (rmax - Rsoft) * np.array([1.0, 0.0])
        # integrate
        pos = pos + dt * fp
        ang = ang + dt * fa
    return pos, ang


def construct():
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT - 15.0
    best_pos, best_ang, best_score = None, None, float('inf')

    def record(pos, ang, score):
        nonlocal best_pos, best_ang, best_score
        if score < best_score:
            best_pos, best_ang, best_score = pos.copy(), ang.copy(), score

    for seed in range(0, 200):
        if time.time() > deadline:
            break
        pos, ang = _build_initial(seed)
        # run physics to relax into a compact packing
        pos, ang = _physics_iterate(pos, ang, 400, t0, deadline)
        if time.time() > deadline:
            break
        min_gap, size = _evaluate(pos, ang)
        if min_gap <= 1e-6:
            continue  # still colliding; skip
        record(pos, ang, size)

    if best_pos is None:
        # fallback: return a valid (if not optimal) layout
        pos, ang = _build_initial(0)
        pos, ang = _physics_iterate(pos, ang, 200, t0, deadline)
        best_pos, best_ang = pos, ang

    # final precise outer size
    verts = [_hex_vertices(best_pos[i, 0], best_pos[i, 1], best_ang[i]) for i in range(N)]
    allv = np.vstack(verts)
    side = _outer_size(allv)
    # center at origin
    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(side), 0.0
