"""Rewritten Dzugutov cluster global-minima search with polytetrahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_START_TIME = time.time()
_TOTAL_BUDGET = 290.0  # safety margin below 300s


def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)


def _icosahedron_vertices(scale=1.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    v = v / np.linalg.norm(v[0]) * scale
    return v


def _make_polytetrahedral_seed(n, spacing=1.1, rng=None):
    """Build a polytetrahedral (icosahedral) cluster of n atoms.

    Uses a central atom + icosahedral shells, trimming or adding to reach n.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Central atom
    pts = [np.zeros(3)]

    # First icosahedral shell (12 atoms) at radius ~ spacing
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p for p in shell1])

    # Second shell: octahedral directions (6) + icosahedral face-centers (20)
    # Octahedral (6) at radius ~ 1.618*spacing (golden ratio)
    octa = np.array([
        [1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0],
        [0, 0, 1.0], [0, 0, -1.0],
    ], dtype=float)
    octa = octa * (spacing * 1.618)
    pts.extend([p for p in octa])

    # Icosahedral face centers (20) at radius ~ 1.382*spacing
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 20 face-centers of icosahedron (normalized)
    face_centers = np.array([
        [1.0, 1.0, phi], [1.0, -1.0, phi], [1.0, 1.0, -phi], [1.0, -1.0, -phi],
        [-1.0, 1.0, phi], [-1.0, -1.0, phi], [-1.0, 1.0, -phi], [-1.0, -1.0, -phi],
        [1.0, phi, 1.0], [-1.0, phi, 1.0], [1.0, phi, -1.0], [-1.0, phi, -1.0],
        [1.0, -phi, 1.0], [-1.0, -phi, 1.0], [1.0, -phi, -1.0], [-1.0, -phi, -1.0],
        [phi, 1.0, 1.0], [-phi, 1.0, 1.0], [phi, -1.0, 1.0], [-phi, -1.0, 1.0],
    ], dtype=float)
    face_centers = face_centers / np.linalg.norm(face_centers[0]) * (spacing * 1.382)
    pts.extend([p for p in face_centers])

    # Third shell: add more icosahedral directions at larger radius
    shell3 = _icosahedron_vertices(scale=spacing * 2.0)
    pts.extend([p for p in shell3])
    shell3b = _icosahedron_vertices(scale=spacing * 2.618)
    pts.extend([p for p in shell3b])

    pts = np.array(pts, dtype=float)

    if len(pts) >= n:
        # Keep the n closest to center (most bound)
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:n]
        pts = pts[idx]
    else:
        # Need to add more atoms: place near existing ones with small jitter
        while len(pts) < n:
            base = pts[rng.integers(0, len(pts))]
            p = base + rng.normal(0, 0.3, 3)
            # ensure not too close to any existing
            if np.all(np.linalg.norm(p - pts, axis=1) > 0.55):
                pts = np.vstack([pts, p])
            else:
                p = base + rng.normal(0, 0.6, 3)
                if np.all(np.linalg.norm(p - pts, axis=1) > 0.55):
                    pts = np.vstack([pts, p])
                else:
                    pts = np.vstack([pts, p])  # fallback, relax will fix
    return pts


def _make_random_compact_seed(n, spacing=1.1, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 200:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in pts):
            pts.append(p)
    if len(pts) < n:
        # fill remaining with jittered duplicates
        for _ in range(n - len(pts)):
            base = pts[rng.integers(0, len(pts))]
            pts.append(base + rng.normal(0, 0.5, 3))
    return np.array(pts, dtype=float)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))

    # Per-instance time budget
    instance_budget = min(48.0, _remaining_time() - 5.0)
    if instance_budget < 3.0:
        instance_budget = 3.0

    t0 = time.time()
    best_energy = np.inf
    best_positions = None

    # Build candidate seeds: polytetrahedral variants + random
    seeds = []
    rng = np.random.default_rng(size)

    # Primary polytetrahedral seeds with slightly varied spacing
    for sp in (1.05, 1.10, 1.15, 1.08):
        seeds.append(_make_polytetrahedral_seed(size, spacing=sp, rng=rng))
    # Random compact seeds
    for _ in range(3):
        seeds.append(_make_random_compact_seed(size, spacing=1.1, rng=rng))

    # Relax each seed, keep best, respecting time budget
    for seed in seeds:
        if (time.time() - t0) > instance_budget:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
        except Exception:
            continue
        if energy < best_energy:
            best_energy = energy
            best_positions = positions

    if best_positions is None:
        # Fallback: a simple seed
        best_positions = _make_random_compact_seed(size, spacing=1.1, rng=rng)
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
