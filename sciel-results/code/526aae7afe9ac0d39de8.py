"""Combined: structured high-symmetry seeds + multi-start local relax + basin-hopping refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 325.0
_T0 = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_verts():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n

def _cuboctahedron_verts():
    v = []
    for x in (-1, 1):
        for y in (-1, 1):
            v.append([x, y, 0.0])
            v.append([x, 0.0, y])
            v.append([0.0, x, y])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _rhombicubocta_verts():
    t = 1.0 + 2.0 ** 0.5
    v = []
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                v.append([a, b, c * t])
                v.append([a, b * t, c])
                v.append([a * t, b, c])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _fcc_lattice(n):
    R = 2.5
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
                for shift in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]:
                    p = np.array([x + shift[0], y + shift[1], z + shift[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    if len(pts) > n:
        pts = pts[:n]
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _bcc_lattice(n):
    R = 2.5
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
                for shift in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]:
                    p = np.array([x + shift[0], y + shift[1], z + shift[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    if len(pts) > n:
        pts = pts[:n]
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _random_cluster(size, rng, spacing=0.72, min_dist=0.85):
    radius = spacing * size ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < size and attempts < size * size * 40:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        attempts += 1
    while len(pts) < size:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:size])

def _shells(n):
    cands = []
    ico = _icosahedron_verts() * 0.60
    cubo = _cuboctahedron_verts() * 0.60
    rho = _rhombicubocta_verts() * 0.62
    shells = [ico * (0.60 + 0.55 * k) for k in range(1, 6)]
    cands.append(_fcc_lattice(n))
    cands.append(_bcc_lattice(n))
    for shell_set in [
        [ico, cubo, rho],
        [ico, rho],
        [ico, cubo],
        [cubo, rho],
        [ico],
        [cubo],
        [rho],
        shells[0], shells[1], shells[2], shells[3],
    ]:
        pts = np.vstack([s for s in shell_set if s is not None and len(s)])
        if len(pts) > 1:
            keep = [0]
            for i in range(1, len(pts)):
                if np.min(np.linalg.norm(pts[i] - pts[keep], axis=1)) > 0.3:
                    keep.append(i)
            pts = pts[keep]
        if len(pts) < n:
            continue
        pts = pts[:n]
        pts = pts - pts.mean(axis=0)
        cands.append(pts)
    return cands

def _sanitize(pos, size):
    """Ensure finite coords and min distance > 0.35."""
    if not np.all(np.isfinite(pos)):
        return None
    if size < 2:
        return pos
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if d.min() < 0.35:
        scale = 0.35 / d.min() + 0.05
        pos = pos * scale
    return pos

def _basin_hopping(pos, potential, kind, rng, n_iter=12, step_scale=0.25):
    """Simple basin-hopping loop around a relaxed structure."""
    best_energy, best_pos = None, None
    try:
        e0, p0 = relax(pos, potential, kind)
        if np.all(np.isfinite(p0)):
            best_energy = float(e0)
            best_pos = np.asarray(p0, float)
    except Exception:
        return pos

    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        noise = step_scale * (1.0 - 0.5 * it / n_iter)
        perturbed = best_pos + rng.normal(0, noise, size=(best_pos.shape[0], 3))
        perturbed = _sanitize(perturbed, best_pos.shape[0])
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        if best_energy is None or energy < best_energy:
            best_energy = float(energy)
            best_pos = positions
    return best_pos

def construct(instance):
    global _T0
    _T0 = time.time()
    size = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)
    rng = np.random.default_rng(hash((potential, kind, size)) & 0xFFFFFFFF)

    # Build candidate seeds
    seeds = []
    for c in _shells(size):
        if c is None or len(c) == 0:
            continue
        c = np.asarray(c, float)
        if len(c) != size:
            continue
        c = _sanitize(c, size)
        if c is not None:
            seeds.append(c)
    for _ in range(8):
        seeds.append(_random_cluster(size, rng, spacing, 0.85))

    best = None
    for seed in seeds:
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(np.asarray(seed, float), potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        if best is None or energy < best[0]:
            best = (float(energy), positions)

    if best is None:
        for _ in range(10):
            if _time_left() < 6.0:
                break
            seed = _random_cluster(size, rng, spacing, 0.9)
            try:
                energy, positions = relax(seed, potential, kind)
                if np.all(np.isfinite(positions)):
                    if best is None or energy < best[0]:
                        best = (float(energy), np.asarray(positions, float))
            except Exception:
                continue

    if best is None:
        pts = _random_cluster(size, rng, spacing, 1.0)
        return np.asarray(pts, float)

    # Basin hopping refinement: stochastic global search to escape local minima
    best_energy, best_pos = best
    bh_result = _basin_hopping(best_pos, potential, kind, rng, n_iter=15, step_scale=0.3)
    if bh_result is not None and np.all(np.isfinite(bh_result)):
        try:
            e_bh, _ = sutton_chen_energy_gradient(bh_result, kind)
            if np.isfinite(e_bh) and e_bh < best_energy:
                best_energy = float(e_bh)
                best_pos = bh_result
        except Exception:
            pass

    # Additional perturbation refinement with decreasing noise
    for trial in range(8):
        if _time_left() < 12.0:
            break
        noise_scale = 0.18 * (1.0 - trial / 8.0) + 0.02
        perturbed = best_pos + rng.normal(0, noise_scale, size=(size, 3))
        perturbed = _sanitize(perturbed, size)
        if perturbed is None:
            continue
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
        positions = np.asarray(positions, float)
        if not np.all(np.isfinite(positions)):
            continue
        if energy < best_energy:
            best_energy = float(energy)
            best_pos = positions

    return best_pos
# EVOLVE-BLOCK-END
