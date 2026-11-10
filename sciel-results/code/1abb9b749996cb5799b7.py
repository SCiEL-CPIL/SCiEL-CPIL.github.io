"""Coarse-to-fine grid seeding with gradient-based insertion for Sutton-Chen clusters."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6":  (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}

_DEADLINE = 320.0  # wall-clock seconds; evaluator kills at 330, we guard earlier
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _sc_full_energy(positions, n_exp, m_exp, c):
    """Full Sutton-Chen energy, vectorized."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero; diagonal terms will be masked
    r2 = np.maximum(r2, 1e-12)
    r_inv = 1.0 / np.sqrt(r2)
    repulsive = 0.5 * np.sum(r_inv ** n_exp)
    # For attractive: for each i, sqrt(sum_{j!=i} r_ij^{-m})
    r_inv_m = r_inv ** m_exp
    np.fill_diagonal(r_inv_m, 0.0)
    sqrt_sums = np.sqrt(np.sum(r_inv_m, axis=1))
    attractive = c * np.sum(sqrt_sums)
    return repulsive - attractive


def _incremental_energy_cached(positions, S, new_pos, n_exp, m_exp, c):
    """Delta E with precomputed S = sum_{j!=i} r_ij^{-m} for each existing atom."""
    n = len(positions)
    if n == 0:
        return 0.0
    diff = new_pos - positions
    r2 = np.sum(diff ** 2, axis=1)
    r2 = np.maximum(r2, 1e-12)
    r_inv = 1.0 / np.sqrt(r2)
    r_inv_n = r_inv ** n_exp
    r_inv_m = r_inv ** m_exp

    rep = np.sum(r_inv_n)
    new_att = -c * np.sqrt(np.sum(r_inv_m))

    S_new = S + r_inv_m
    existing_att_delta = -c * np.sum(np.sqrt(S_new) - np.sqrt(S))

    return rep + new_att + existing_att_delta


def _compute_S(positions, m_exp):
    """Compute S_i = sum_{j!=i} r_ij^{-m} for all i."""
    n = len(positions)
    if n < 2:
        return np.zeros(n)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r_inv = 1.0 / np.sqrt(r2)
    r_inv_m = r_inv ** m_exp
    np.fill_diagonal(r_inv_m, 0.0)
    return np.sum(r_inv_m, axis=1)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.min(r2) >= min_dist ** 2


def _grid_seed_core(n_atoms, n_exp, m_exp, c, rng, radius=1.5, grid_spacing=0.2):
    """Place n_atoms on a grid within a sphere, pick lowest-energy subset via greedy insertion."""
    # Generate candidate grid points
    points = []
    r = radius
    for x in np.arange(-r, r + grid_spacing, grid_spacing):
        for y in np.arange(-r, r + grid_spacing, grid_spacing):
            for z in np.arange(-r, r + grid_spacing, grid_spacing):
                if np.sqrt(x*x + y*y + z*z) <= r:
                    points.append(np.array([x, y, z]))
    points = np.array(points)
    # Shuffle for diversity
    idx = rng.permutation(len(points))
    points = points[idx]

    core = []
    S = np.array([])
    for _ in range(n_atoms):
        if len(core) == 0:
            # First atom: pick center or random
            core.append(np.zeros(3))
            S = np.array([0.0])
            continue
        core_arr = np.array(core)
        best_pos = None
        best_e = np.inf
        # Evaluate all candidates for small cores, subset for larger
        n_cand = min(len(points), 3000)
        cand_idx = rng.choice(len(points), size=n_cand, replace=False)
        for ci in cand_idx:
            p = points[ci]
            # Check min distance
            d = np.linalg.norm(p - core_arr, axis=1)
            if np.min(d) < 0.4:
                continue
            e = _incremental_energy_cached(core_arr, S, p, n_exp, m_exp, c)
            if e < best_e:
                best_e = e
                best_pos = p
        if best_pos is None:
            # Fallback: random position
            best_pos = rng.uniform(-radius, radius, 3)
            while len(core) > 0 and np.min(np.linalg.norm(best_pos - np.array(core), axis=1)) < 0.4:
                best_pos = rng.uniform(-radius, radius, 3)
        core.append(best_pos)
        # Update S
        S = _compute_S(np.array(core), m_exp)

    return np.array(core)


def _insert_atom(positions, S, n_exp, m_exp, c, rng, n_candidates=2000):
    """Insert one atom at the best incremental-energy position."""
    n = len(positions)
    if n == 0:
        return np.zeros(3), np.array([0.0])

    # Sample candidate positions on shells around the cluster
    centroid = np.mean(positions, axis=0)
    max_r = np.max(np.linalg.norm(positions - centroid, axis=1))
    shell_radii = np.linspace(0.5, max_r + 1.5, 5)

    candidates = []
    for sr in shell_radii:
        # Random points on sphere of radius sr
        n_per_shell = max(1, n_candidates // len(shell_radii))
        theta = rng.uniform(0, 2 * np.pi, n_per_shell)
        phi = np.arccos(2 * rng.uniform(-1, 1, n_per_shell))
        xs = sr * np.sin(phi) * np.cos(theta)
        ys = sr * np.sin(phi) * np.sin(theta)
        zs = sr * np.cos(phi)
        candidates.append(np.column_stack([xs, ys, zs]))

    candidates = np.vstack(candidates)
    # Also add some random points in a box
    box_r = max_r + 1.0
    candidates = np.vstack([candidates, rng.uniform(-box_r, box_r, (n_candidates // 2, 3))])

    best_pos = None
    best_e = np.inf
    pos_arr = np.array(positions)
    for p in candidates:
        d = np.linalg.norm(p - pos_arr, axis=1)
        if np.min(d) < 0.38:
            continue
        e = _incremental_energy_cached(pos_arr, S, p, n_exp, m_exp, c)
        if e < best_e:
            best_e = e
            best_pos = p

    if best_pos is None:
        # Fallback
        best_pos = rng.uniform(-2, 2, 3)
        while len(positions) > 0 and np.min(np.linalg.norm(best_pos - np.array(positions), axis=1)) < 0.38:
            best_pos = rng.uniform(-2, 2, 3)

    new_positions = np.vstack([pos_arr, best_pos])
    new_S = _compute_S(new_positions, m_exp)
    return new_positions, new_S


def _relax_and_refine(positions, potential, kind):
    """Relax positions, returning (energy, positions)."""
    try:
        energy, relaxed = relax(positions, potential, kind)
        if not _is_valid(relaxed):
            return np.inf, positions
        return energy, relaxed
    except Exception:
        return np.inf, positions


def _basin_hopping(positions, potential, kind, n_exp, m_exp, c, rng, n_kicks=20, kick_scale=0.1):
    """Simple basin hopping: kick + relax, keep best."""
    best_e, best_pos = _relax_and_refine(positions, potential, kind)
    
    for _ in range(n_kicks):
        if _remaining() < 10.0:
            break
        # Kick
        kicked = best_pos + rng.normal(0, kick_scale, best_pos.shape)
        if not _is_valid(kicked):
            continue
        e, relaxed = _relax_and_refine(kicked, potential, kind)
        if e < best_e:
            best_e, best_pos = e, relaxed
    
    return best_e, best_pos


def construct(instance):
    global _START
    _START = time.time()

    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    n_exp, m_exp, c = _PARAMS[kind]

    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(42)

    # Strategy: multiple restarts with different core sizes and insertion sampling
    core_sizes = [6, 8, 10]
    n_restarts_per_core = 2

    for core_size in core_sizes:
        if _remaining() < 30.0:
            break
        for restart in range(n_restarts_per_core):
            if _remaining() < 20.0:
                break
            rng_r = np.random.default_rng(1000 + core_size * 10 + restart)

            # Grid-seed the core
            core = _grid_seed_core(core_size, n_exp, m_exp, c, rng_r, radius=1.2, grid_spacing=0.2)

            # Validate core
            if len(core) >= 2 and not _is_valid(core):
                continue

            # Insert remaining atoms one by one
            positions = core.copy()
            S = _compute_S(positions, m_exp)

            for i in range(core_size, size):
                if _remaining() < 15.0:
                    break
                positions, S = _insert_atom(positions, S, n_exp, m_exp, c, rng_r, n_candidates=1500)

            if len(positions) < size:
                # Fill remaining randomly
                while len(positions) < size:
                    p = rng_r.uniform(-2, 2, 3)
                    if len(positions) > 0 and np.min(np.linalg.norm(p - positions, axis=1)) < 0.38:
                        continue
                    positions = np.vstack([positions, p])
                S = _compute_S(positions, m_exp)

            # Relax
            energy, relaxed = _relax_and_refine(positions, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed

            # If we have time, do basin hopping from the best found so far
            if _remaining() > 45.0 and best_positions is not None:
                n_kicks = max(5, int(_remaining() // 5))
                e_bh, pos_bh = _basin_hopping(best_positions, potential, kind, n_exp, m_exp, c, rng_r, n_kicks=n_kicks, kick_scale=0.08)
                if e_bh < best_energy:
                    best_energy = e_bh
                    best_positions = pos_bh

    # Final fallback: if nothing valid, use a simple random cluster
    if best_positions is None:
        radius = 0.72 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.4 for q in points):
                    points.append(p)
        positions = np.array(points)
        energy, relaxed = _relax_and_refine(positions, potential, kind)
        if _is_valid(relaxed):
            best_positions = relaxed
        else:
            best_positions = positions

    return best_positions
# EVOLVE-BLOCK-END
