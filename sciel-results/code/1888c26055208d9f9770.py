"""Sutton-Chen transition-metal cluster global minima: basin-hopping + multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_START_TIME = None
_TIME_LIMIT = 315.0  # Leave 15s safety margin before 330s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _make_fcc(n, spacing=0.72):
    """Generate n atoms in an FCC-like arrangement."""
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
    max_idx = int(r_max / (a * 0.5)) + 2
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a * 0.5
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
                else:
                    p = (np.array([i, j, k]) + np.array([1, 1, 1])) * a * 0.25
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
    candidates = np.array(candidates)
    if len(candidates) < n:
        return None
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_bcc(n, spacing=0.72):
    """Generate n atoms in a BCC-like arrangement."""
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
    max_idx = int(r_max / a) + 2
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p_corner = np.array([i, j, k]) * a
                if np.linalg.norm(p_corner) <= r_max:
                    candidates.append(p_corner)
                p_body = (np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a
                if np.linalg.norm(p_body) <= r_max:
                    candidates.append(p_body)
    candidates = np.array(candidates)
    if len(candidates) < n:
        return None
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_icosahedral(n, spacing=0.72):
    """Generate n atoms in a Mackay icosahedral arrangement."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]
    r1 = spacing * 0.65
    points.extend(verts * r1)
    
    # Edge midpoints (30 edges)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(verts[i] - verts[j]) < 1.05:
                edges.append((i, j))
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        points.append(mid * r1 * 1.15)
    
    if n <= len(points):
        return np.array(points[:n])
    
    # Fill remaining with Fibonacci sphere on larger shells
    remaining = n - len(points)
    r3 = spacing * 1.4
    golden = (1 + np.sqrt(5)) / 2
    for i in range(remaining):
        y = 1 - (i + 0.5) / remaining * 2
        r = np.sqrt(1 - y*y)
        theta = golden * i * 2 * np.pi
        p = np.array([r * np.cos(theta), y, r * np.sin(theta)]) * r3
        points.append(p)
    return np.array(points[:n])

def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact cluster with minimum distance constraint (vectorized)."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    max_attempts = 30000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: allow closer packing
        points = []
        attempts = 0
        while len(points) < n and attempts < 50000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if len(points) == 0:
                    points.append(p)
                else:
                    arr = np.array(points)
                    dists = np.linalg.norm(arr - p, axis=1)
                    if np.min(dists) >= 0.36:
                        points.append(p)
            attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _make_spherical_shells(n, spacing=0.72, seed=None):
    """Generate n atoms on concentric spherical shells."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    remaining = n - 1
    if remaining <= 0:
        return np.array(points)
    
    r1 = spacing * 0.7
    r2 = spacing * 1.3
    r3 = spacing * 1.9
    weights = [r1**2, r2**2, r3**2]
    total_weight = sum(weights)
    counts = [int(remaining * w / total_weight) for w in weights]
    diff = remaining - sum(counts)
    counts[-1] += diff
    
    for r, count in zip([r1, r2, r3], counts):
        if count <= 0:
            continue
        pts = rng.normal(0, 1, (count, 3))
        norms = np.linalg.norm(pts, axis=1)
        pts /= norms[:, np.newaxis]
        pts *= r
        points.extend(pts)
    return np.array(points[:n])

def _make_deformed(base_positions, spacing=0.72, seed=None, scale=0.08):
    """Create a deformed version of base positions."""
    rng = np.random.default_rng(seed)
    positions = base_positions.copy()
    noise = rng.normal(0, scale * spacing, positions.shape)
    positions += noise
    # Ensure no atoms are too close
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.36 and d > 1e-10:
                direction = (positions[i] - positions[j]) / d
                positions[i] += direction * (0.36 - d) / 2
                positions[j] -= direction * (0.36 - d) / 2
    return positions

def _basin_hopping(positions, potential, kind, n_perturb=15, base_scale=0.08, seed=0):
    """Simple basin hopping: perturb and relax, keep best."""
    rng = np.random.default_rng(seed)
    best_e, best_pos = sutton_chen_energy_gradient(positions, kind)
    if not np.isfinite(best_e):
        return best_e, best_pos
    
    for i in range(n_perturb):
        if _time_left() < 3.0:
            break
        scale = base_scale * (1.0 + 0.3 * np.sin(i * 0.7))
        perturbed = _make_deformed(best_pos, spacing=0.72, seed=seed + i, scale=scale)
        try:
            e, relaxed = relax(perturbed, potential, kind)
            if np.isfinite(e) and e < best_e:
                dists = cdist(relaxed, relaxed)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.35:
                    best_e = e
                    best_pos = relaxed
        except Exception:
            pass
    return best_e, best_pos

def construct(instance):
    """
    Construct a low-energy Sutton-Chen cluster using basin-hopping + multi-start.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance.get("potential", "sutton_chen")
    spacing = 0.72
    
    best_energy = np.inf
    best_positions = None
    
    def _try_relax(positions, label=""):
        nonlocal best_energy, best_positions
        if _time_left() < 2.0:
            return
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                dists = cdist(relaxed, relaxed)
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                if min_d >= 0.35:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Generate diverse starting structures
    start_structures = []
    
    for factory, kwargs in [
        (_make_fcc, {'spacing': spacing}),
        (_make_bcc, {'spacing': spacing}),
        (_make_icosahedral, {'spacing': spacing}),
        (_make_spherical_shells, {'spacing': spacing, 'seed': n+5}),
        (_make_spherical_shells, {'spacing': spacing, 'seed': n+6}),
        (_make_random_compact, {'spacing': spacing, 'seed': n, 'min_dist': 0.5}),
        (_make_random_compact, {'spacing': spacing, 'seed': n+1, 'min_dist': 0.5}),
        (_make_random_compact, {'spacing': spacing, 'seed': n+2, 'min_dist': 0.45}),
        (_make_random_compact, {'spacing': spacing, 'seed': n+3, 'min_dist': 0.4}),
    ]:
        try:
            struct = factory(n, **kwargs)
            if struct is not None and len(struct) == n and np.all(np.isfinite(struct)):
                start_structures.append(struct)
        except Exception:
            pass
    
    # Relax all starting structures
    for i, structure in enumerate(start_structures):
        if _time_left() < 3.0:
            break
        _try_relax(structure, f"struct_{i}")
    
    # Basin hopping from the best structure found
    if best_positions is not None and _time_left() > 15.0:
        n_perturb = max(10, int(_time_left() // 3))
        best_e, best_pos = _basin_hopping(
            best_positions, potential, kind,
            n_perturb=n_perturb, base_scale=0.08, seed=n+100
        )
        if np.isfinite(best_e) and best_e < best_energy:
            dists = cdist(best_pos, best_pos)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.35:
                best_energy = best_e
                best_positions = best_pos
    
    # Additional basin hopping from second-best if available
    if best_positions is not None and _time_left() > 10.0:
        # Try basin hopping from a slightly perturbed version
        perturbed = _make_deformed(best_positions, spacing=spacing, seed=n+200, scale=0.12)
        n_perturb = max(5, int(_time_left() // 4))
        best_e2, best_pos2 = _basin_hopping(
            perturbed, potential, kind,
            n_perturb=n_perturb, base_scale=0.1, seed=n+300
        )
        if np.isfinite(best_e2) and best_e2 < best_energy:
            dists = cdist(best_pos2, best_pos2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.35:
                best_energy = best_e2
                best_positions = best_pos2
    
    # If still time, try more random starts with relaxation
    extra_starts = 3
    for k in range(extra_starts):
        if _time_left() < 5.0:
            break
        seed = n + 400 + k
        try:
            rnd = _make_random_compact(n, spacing, seed=seed, min_dist=0.45)
            if len(rnd) == n:
                _try_relax(rnd, f"extra_{k}")
        except Exception:
            pass
    
    # Final fallback
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        for i in range(n):
            theta = i * 2.399
            r = spacing * 0.3 * np.sqrt(i)
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0.1 * i * 0.05]
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                best_positions = relaxed
        except Exception:
            best_positions = positions
    
    # Ensure validity
    if not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        side = int(np.ceil(n ** (1/3)))
        for i in range(n):
            ix = i % side
            iy = (i // side) % side
            iz = i // (side * side)
            positions[i] = [ix * 0.5, iy * 0.5, iz * 0.5]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
