"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

_T0 = time.time()
_DEADLINE = 322.0  # safety margin below the 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _icosahedron_verts():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _cuboctahedron_verts():
    v = np.array([
        [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
        [0.0, 1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0],
        [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0],
    ], dtype=float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _rhombicubocta_verts():
    t = 0.4142135623730951  # sqrt(2)-1
    v = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                v.append([sx, sy, sz])
                v.append([sx, sy, t])
                v.append([sx, t, sy])
                v.append([t, sx, sy])
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _fcc_lattice(n, spacing):
    side = int(np.ceil(n ** (1.0 / 3.0))) + 1
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                pts.append((i, j, k))
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    pts *= spacing
    order = np.argsort(np.linalg.norm(pts, axis=1))
    return pts[order[:n]]


def _bcc_lattice(n, spacing):
    side = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                pts.append((i, j, k))
                pts.append((i + 0.5, j + 0.5, k + 0.5))
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    pts *= spacing
    order = np.argsort(np.linalg.norm(pts, axis=1))
    return pts[order[:n]]


def _shells(n, spacing):
    """Concentric icosahedral shells (12-30-42...) scaled to the spacing."""
    counts = [12, 30, 42, 60, 80, 120]
    shells = [
        _icosahedron_verts(),
        _icosahedron_verts() * 1.0,
        _rhombicubocta_verts(),
        _rhombicubocta_verts() * 1.0,
        _cuboctahedron_verts(),
        _cuboctahedron_verts() * 1.0,
    ]
    radii = [1.0, 1.6, 2.1, 2.6, 3.1, 3.6]
    pts = [np.array([0.0, 0.0, 0.0])]
    need = n
    idx = 0
    while need > 0 and idx < len(shells):
        take = min(need, counts[idx])
        pts.append(shells[idx][:take] * radii[idx])
        need -= take
        idx += 1
    if need > 0:
        base = _icosahedron_verts()
        extra = np.repeat(base, need // 12 + 1, axis=0)[:need] * (radii[-1] + 0.5)
        pts.append(extra)
    out = np.vstack(pts)
    out *= spacing
    return out[:n]


def _random_cluster(n, spacing, rng):
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 40000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.55 * spacing:
                    ok = False
                    break
            if ok:
                pts.append(p)
        attempts += 1
    # fill any shortfall by nudging on a sphere
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts, dtype=float)


def _sanitize(pos):
    pos = np.asarray(pos, dtype=float)
    pos[~np.isfinite(pos)] = 0.0
    return pos


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 0.72 if potential == "sutton_chen" else 1.0
    rng = np.random.default_rng(size * 7 + 1)

    seeds = []
    seeds.append(_shells(size, spacing))
    seeds.append(_fcc_lattice(size, spacing))
    seeds.append(_bcc_lattice(size, spacing))
    for s in (0.9, 1.1):
        seeds.append(_shells(size, spacing * s))
    for k in range(4):
        seeds.append(_random_cluster(size, spacing, rng))

    best = None
    for seed in seeds:
        if _time_left() < 6.0:
            break
        try:
            energy, positions = relax(_sanitize(seed), potential, parameter)
            if not np.isfinite(energy):
                continue
            if best is None or energy < best[0]:
                best = (energy, positions)
        except Exception:
            continue

    if best is None:
        return _sanitize(_shells(size, spacing))

    # Perturbation refinement around the best structure while time allows.
    best_energy, best_pos = best
    for _ in range(40):
        if _time_left() < 8.0:
            break
        noise = rng.normal(0.0, 0.06, best_pos.shape)
        cand = _sanitize(best_pos + noise)
        try:
            energy, positions = relax(cand, potential, parameter)
            if np.isfinite(energy) and energy < best_energy:
                best_energy, best_pos = energy, positions
        except Exception:
            continue

    return _sanitize(best_pos)
# EVOLVE-BLOCK-END
