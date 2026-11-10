"""
Sutton-Chen cluster global minima search using basin hopping with diverse seeds.
Strategy:
  1. Generate multiple diverse initial configurations (FCC/HCP/BCC truncations,
     random with min-distance, gradient-insertion seeds).
  2. Basin hopping: kick + relax, accept by Metropolis criterion.
  3. Keep best configuration found.
"""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6":  (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}

_DEADLINE = 320.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _sc_energy(positions, n_exp, m_exp, c):
    """Full Sutton-Chen energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r_inv = 1.0 / np.sqrt(r2)
    repulsive = 0.5 * np.sum(r_inv ** n_exp)
    r_inv_m = r_inv ** m_exp
    np.fill_diagonal(r_inv_m, 0.0)
    sqrt_sums = np.sqrt(np.sum(r_inv_m, axis=1))
    attractive = c * np.sum(sqrt_sums)
    return repulsive - attractive


def _valid(positions, min_dist=0.35):
    """Check positions are finite and min distance constraint satisfied."""
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.min(r2) >= min_dist ** 2


def _fcc_seed(n, rng):
    """FCC lattice truncation to n atoms."""
    # FCC lattice vectors
    a = 1.0
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ]) * a
    # Generate FCC points in a cube, sort by distance from center, take n
    pts = []
    extent = int(np.ceil((n / 4) ** (1/3))) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    pts.append(p)
    pts = np.array(pts)
    # Center
    center = pts.mean(axis=0)
    pts -= center
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]].copy()


def _hcp_seed(n, rng):
    """HCP-like seed: layered hexagonal packing."""
    # Simple approach: place atoms in layers with hexagonal arrangement
    pts = []
    a = 1.0  # in-plane spacing
    c = 1.633 * a  # c/a ratio for HCP
    layer_idx = 0
    while len(pts) < n:
        # Hexagonal layer with radius growing
        r = int(np.ceil(np.sqrt(len(pts) / (np.pi * 3)))) + 1
        layer_pts = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    if i * i + j * j + k * k <= r * r and (i + j + k) % 2 == 0:
                        x = a * (i + 0.5 * j)
                        y = a * np.sqrt(3) / 2 * j
                        z = c * layer_idx
                        if layer_idx % 2 == 1:
                            x += 0.5 * a
                        layer_pts.append([x, y, z])
        if not layer_pts:
            layer_pts = [[0, 0, c * layer_idx]]
        for p in layer_pts:
            if len(pts) >= n:
                break
            # Check min distance
            if all(np.linalg.norm(np.array(p) - np.array(q)) >= 0.38 for q in pts):
                pts.append(p)
        layer_idx += 1
        if layer_idx > 20:
            break
    # Fill remaining randomly
    while len(pts) < n:
        p = rng.uniform(-2, 2, 3)
        if all(np.linalg.norm(p - np.array(q)) >= 0.38 for q in pts):
            pts.append(p)
    return np.array(pts[:n])


def _bcc_seed(n, rng):
    """BCC lattice truncation."""
    a = 1.0
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * a
    pts = []
    extent = int(np.ceil((n / 2) ** (1/3))) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    pts.append(p)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]].copy()


def _random_seed(n, rng, min_dist=0.45):
    """Random placement with minimum distance."""
    pts = []
    radius = 0.8 * n ** (1.0/3.0)
    attempts = 0
    while len(pts) < n and attempts < 100000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    # Fill if needed
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:n])


def _gradient_insertion_seed(n, n_exp, m_exp, c, rng):
    """Build cluster atom-by-atom minimizing incremental energy."""
    positions = [np.zeros(3)]
    for _ in range(n - 1):
        best_pos = None
        best_e = np.inf
        pos_arr = np.array(positions)
        # Sample candidates on shells
        centroid = pos_arr.mean(axis=0)
        max_r = np.max(np.linalg.norm(pos_arr - centroid, axis=1))
        for _trial in range(200):
            r = rng.uniform(0.5, max_r + 1.0)
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * rng.uniform(-1, 1))
            p = centroid + r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            if len(positions) > 0 and np.min(np.linalg.norm(p - pos_arr, axis=1)) < 0.38:
                continue
            # Incremental energy
            diff = p - pos_arr
            r2 = np.sum(diff ** 2, axis=1)
            r2 = np.maximum(r2, 1e-12)
            r_inv = 1.0 / np.sqrt(r2)
            rep = np.sum(r_inv ** n_exp)
            att = -c * np.sqrt(np.sum(r_inv ** m_exp))
            # Approximate existing atom changes
            e = rep + att
            if e < best_e:
                best_e = e
                best_pos = p
        if best_pos is None:
            best_pos = rng.uniform(-1, 1, 3)
            while len(positions) > 0 and np.min(np.linalg.norm(best_pos - pos_arr, axis=1)) < 0.38:
                best_pos = rng.uniform(-1, 1, 3)
        positions.append(best_pos)
    return np.array(positions)


def _kick(positions, kick_size, rng):
    """Random perturbation."""
    return positions + rng.normal(0, kick_size, positions.shape)


def _relax(positions, potential, kind):
    """Relax with L-BFGS-B."""
    try:
        energy, relaxed = relax(positions, potential, kind)
        if not _valid(relaxed):
            return np.inf, positions
        return energy, relaxed
    except Exception:
        return np.inf, positions


def construct(instance):
    global _START
    _START = time.time()

    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    n_exp, m_exp, c = _PARAMS[kind]
    rng = np.random.default_rng(42)

    best_energy = np.inf
    best_positions = None

    # Generate diverse initial seeds
    seeds = []
    seeds.append(_fcc_seed(size, rng))
    seeds.append(_hcp_seed(size, rng))
    seeds.append(_bcc_seed(size, rng))
    seeds.append(_random_seed(size, rng))
    if _remaining() > 60:
        seeds.append(_gradient_insertion_seed(size, n_exp, m_exp, c, rng))
    if _remaining() > 40:
        seeds.append(_random_seed(size, np.random.default_rng(123), min_dist=0.5))
    if _remaining() > 30:
        seeds.append(_random_seed(size, np.random.default_rng(456), min_dist=0.55))

    # Basin hopping on each seed
    for seed_idx, seed in enumerate(seeds):
        if _remaining() < 15:
            break
        if not _valid(seed):
            continue

        # Initial relaxation
        energy, positions = _relax(seed, potential, kind)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()

        # Basin hopping
        kick_size = 0.5
        n_kicks = 30  # adaptive
        for kick_i in range(n_kicks):
            if _remaining() < 10:
                break
            kicked = _kick(positions, kick_size, rng)
            if not _valid(kicked):
                continue
            e_new, pos_new = _relax(kicked, potential, kind)
            if not np.isfinite(e_new):
                continue

            # Metropolis acceptance
            if e_new < energy or (energy > 0 and rng.random() < np.exp(-(e_new - energy) / 0.1)):
                energy = e_new
                positions = pos_new
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            else:
                # Keep current
                pass

            # Adapt kick size
            if kick_i % 10 == 9:
                kick_size = max(0.1, kick_size * 0.85)

    # Final: if we have time, do one more round of basin hopping from best
    if _remaining() > 30 and best_positions is not None:
        positions = best_positions.copy()
        energy = best_energy
        kick_size = 0.3
        for _ in range(50):
            if _remaining() < 10:
                break
            kicked = _kick(positions, kick_size, rng)
            if not _valid(kicked):
                continue
            e_new, pos_new = _relax(kicked, potential, kind)
            if np.isfinite(e_new) and (e_new < energy or rng.random() < np.exp(-(e_new - energy) / 0.05)):
                energy = e_new
                positions = pos_new
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            kick_size = max(0.1, kick_size * 0.95)

    if best_positions is None:
        # Absolute fallback
        best_positions = _random_seed(size, rng)
        if not _valid(best_positions):
            # Force valid
            pts = []
            for _ in range(size):
                p = rng.uniform(-1, 1, 3)
                pts.append(p)
            best_positions = np.array(pts)

    return best_positions
