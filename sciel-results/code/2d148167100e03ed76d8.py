"""Sutton-Chen transition-metal cluster global minima: Multi-Start Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Per-kind parameters: (n_exp, m_exp, c)
_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

# Spacing per kind for seed generation
_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}


def _validate_positions(pos):
    """Vectorized validity check: finite coords and min pair distance >= 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    """Generate n-atom FCC-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites)
    sites = sites * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    """Generate n-atom HCP-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                p1 = base + np.array([0, 0, z])
                p2 = base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2])
                sites.append(p1)
                sites.append(p2)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _bcc_cluster(n, spacing=0.72):
    """Generate n-atom BCC-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    """Generate n-atom icosahedral-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    ico_norms = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_norms[i] - ico_norms[j])
            if 1.0 < d < 2.0:
                mid = (ico_norms[i] + ico_norms[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Use a simple check for speed, full validation later
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempt += 1
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def construct(instance):
    """
    Multi-Start Local Search with strict time budgeting.
    1. Generate diverse seeds (FCC, HCP, BCC, Icosahedral, Random).
    2. Relax each seed using the provided `relax` helper.
    3. Use remaining time to perturb the best solution and relax again.
    """
    start_time = time.time()
    # Leave a safety margin of 20 seconds to ensure we return well before 330s limit
    deadline = start_time + 310.0
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)
    
    # Generate seeds
    seeds = []
    scales = [0.85, 0.95, 1.0, 1.05, 1.15]
    
    for scale in scales:
        s = spacing * scale
        seeds.append(_fcc_cluster(size, s))
        seeds.append(_hcp_cluster(size, s))
        seeds.append(_bcc_cluster(size, s))
        seeds.append(_icosahedral_cluster(size, s))
        
    # Add random seeds
    rng = np.random.default_rng(42)
    for i in range(10):
        seeds.append(_random_compact(size, spacing, seed=42 + i))

    # Remove duplicates and validate
    unique_seeds = []
    seen = set()
    for s in seeds:
        if s.shape[0] != size:
            continue
        if not _validate_positions(s):
            continue
        key = tuple(np.round(s, 2).flatten())
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)
            
    best_energy = np.inf
    best_positions = None
    
    # Relax each seed
    # We don't know exactly how long relax takes, but for n<100 it should be fast (<1s).
    # We will check time before each call.
    for seed in unique_seeds:
        if time.time() > deadline - 10.0:
            break
        
        try:
            # relax returns (energy, positions)
            energy, positions = relax(seed, potential, kind)
            
            if np.isfinite(energy) and _validate_positions(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue

    # Fallback if no valid result
    if best_positions is None:
        # Create a simple linear chain or random valid cluster
        best_positions = _random_compact(size, spacing, seed=999)
        if not _validate_positions(best_positions):
            # Last resort: grid
            best_positions = np.zeros((size, 3))
            best_positions[:, 0] = np.arange(size) * spacing
        # Try to relax this fallback
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except Exception:
            pass

    # Polish phase: Use remaining time to perturb the best result
    if best_positions is not None:
        # Estimate remaining time
        remaining = deadline - time.time()
        
        # If we have more than 5 seconds, do some perturbations
        if remaining > 5.0:
            # Try perturbations of decreasing magnitude
            # We'll do a few quick relaxations
            perturb_scales = [0.1, 0.2, 0.3, 0.5]
            
            for scale in perturb_scales:
                if time.time() > deadline - 5.0:
                    break
                
                perturbed = best_positions + rng.normal(0, scale * spacing, best_positions.shape)
                if _validate_positions(perturbed):
                    try:
                        energy, positions = relax(perturbed, potential, kind)
                        if np.isfinite(energy) and _validate_positions(positions):
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = positions
                    except Exception:
                        pass

    # Final validation
    if not _validate_positions(best_positions):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * spacing
        
    return best_positions
# EVOLVE-BLOCK-END
