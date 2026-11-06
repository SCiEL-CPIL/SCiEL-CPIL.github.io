"""Morse cluster global-minima search: structured multi-start + time-adaptive basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin well before the 330 s kill / 300 s target


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _morse_energy(pos, rho):
    n = len(pos)
    if n < 2:
        return 0.0
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    np.fill_diagonal(r, 1e10)
    u = rho * (1.0 - r)
    e = np.exp(u)
    return 0.5 * np.sum(e * (e - 2.0))


def _fix_min_dist(pos, min_d=0.501):
    """Push apart any pair closer than min_d (a few sweeps)."""
    n = len(pos)
    pos = pos.copy()
    for _ in range(6):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                v = pos[i] - pos[j]
                d = np.linalg.norm(v)
                if d < min_d:
                    if d < 1e-8:
                        v = np.array([0.01, 0.01, 0.02])
                        d = np.linalg.norm(v)
                    push = (min_d - d) / d
                    v = v / d * (0.5 * min_d + 0.5 * (min_d - d))
                    pos[i] += v
                    pos[j] -= v
                    moved = True
        if not moved:
            break
    return pos


def _fcc_start(n, spacing=1.0):
    """FCC lattice, trimmed to n points, centered."""
    pts = []
    # generate FCC points in a cube
    L = max(4, int(np.ceil(n ** (1.0 / 3.0))) + 2)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                # FCC lattice points: (i,j,k) and face-centers
                pts.append(np.array([i, j, k]))
                if i < L - 1:
                    pts.append(np.array([i + 0.5, j + 0.5, k]))
                if j < L - 1:
                    pts.append(np.array([i, j + 0.5, k + 0.5]))
                if k < L - 1:
                    pts.append(np.array([i + 0.5, j, k + 0.5]))
    pts = np.array(pts[: max(n, 1)]) * spacing
    pts = pts - pts.mean(axis=0)
    return pts[:n]


def _icosahedral_start(n, spacing=1.0):
    """Concentric icosahedral shells (icosahedron + inner 13-atom icosahedron motif)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])  # unit icosahedron
    shells = [0.0, 1.0, np.sqrt(3.0), np.sqrt(5.0)]  # radii (relative)
    pts = [np.array([0.0, 0.0, 0.0])]
    for rad in shells[1:]:
        pts.extend((verts * rad).tolist())
        if len(pts) >= n:
            break
    pts = np.array(pts[:n]) * spacing
    pts = pts - pts.mean(axis=0)
    return pts


def _relax_with_check(pos, potential, rho):
    """Local relaxation; return (energy, positions)."""
    try:
        energy, positions = relax(pos, potential, rho)
    except Exception:
        energy = _morse_energy(pos, rho)
        positions = pos
    return energy, positions


def construct(instance):
    """Structured multi-start + time-adaptive basin hopping for Morse clusters."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size * 131 + 7)

    # --- Build a diverse pool of starting configurations -------------------
    starts = []
    for gen in (_fcc_start, _icosahedral_start):
        try:
            starts.append(gen(size))
        except Exception:
            pass
    # A couple of compact random starts for robustness
    radius = 1.0 * size ** (1.0 / 3.0) * 0.8
    for _ in range(2):
        pts = []
        while len(pts) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.8 for q in pts
            ):
                pts.append(p)
        if len(pts) == size:
            starts.append(np.array(pts))

    # --- Relax each start, keep the best, seed basin hopping ---------------
    best_energy = np.inf
    best_pos = None
    for s in starts:
        if _remaining_time() < 25.0:
            break
        e, p = _relax_with_check(s, potential, rho)
        if e < best_energy:
            best_energy = e
            best_pos = p.copy()

    if best_pos is None:  # fallback
        best_pos = _fcc_start(size)
        best_energy = _morse_energy(best_pos, rho)

    # --- Time-adaptive basin hopping ---------------------------------------
    # The number of exploratory kicks is driven by the remaining budget:
    # a faster relaxation lets the loop run more kicks before the deadline.
    kick_scale = 0.35
    min_kick = 0.05
    temp = 1.0
    min_temp = 1e-4
    while _remaining_time() > 20.0:
        kick = kick_scale * (1.0 + 0.5 * (1.0 - kick_scale / 0.35))
        kick = max(kick, min_kick)
        perturbed = best_pos + rng.normal(0.0, kick, size=best_pos.shape)
        e2, p2 = _relax_with_check(perturbed, potential, rho)
        if e2 < best_energy or (
            rng.random() < np.exp((best_energy - e2) / max(temp, min_temp))
        ):
            best_energy = e2
            best_pos = p2.copy()
        # anneal: shrink kick and temperature each accepted/total step
        kick_scale = max(min_kick, kick_scale * 0.995)
        temp = max(min_temp, temp * 0.99)

    # --- Guarantee the separation constraint -------------------------------
    best_pos = _fix_min_dist(best_pos)
    return best_pos
# EVOLVE-BLOCK-END
