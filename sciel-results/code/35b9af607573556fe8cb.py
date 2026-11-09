"""Sutton-Chen transition-metal cluster global minima: multi-start structured seeds + L-BFGS-B relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
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
                    p = np.array([i, j, k]) * spacing + b * spacing
                    sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)[:n]
    return sites[idx]


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                # hcp positions
                p1 = np.array([i * a, j * a * np.sqrt(3), k * c])
                p2 = np.array([(i + 0.5) * a, (j + 0.5) * a * np.sqrt(3), k * c + c / 2])
                sites.append(p1)
                sites.append(p2)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)[:n]
    return sites[idx]


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral-like cluster (Mackay icosahedron shells)."""
    if n <= 0:
        return np.empty((0, 3))
    # Icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0]) * 2 * phi / np.sqrt(1 + phi**2)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    
    # Shell radii: 1, 2, 3...
    shell_radii = [1.0, 2.0, 3.0, 4.0]
    all_points = [np.zeros(3)]  # center
    for r in shell_radii:
        # Add vertices scaled by r
        all_points.extend(verts * r)
        # Add edge midpoints for shell 2+
        if r >= 2:
            for i in range(12):
                for j in range(i + 1, 12):
                    d = np.linalg.norm(verts[i] - verts[j])
                    if 1.0 < d < 2.0:  # adjacent vertices
                        mid = (verts[i] + verts[j]) / 2
                        mid = mid / np.linalg.norm(mid) * r
                        all_points.append(mid)
    
    all_points = np.array(all_points)
    # Scale by spacing
    all_points *= spacing / 1.0  # unit radius
    
    # If we have more than n, take closest to origin
    if len(all_points) >= n:
        dists = np.linalg.norm(all_points, axis=1)
        idx = np.argsort(dists)[:n]
        return all_points[idx]
    else:
        # Need more points, add random points on sphere
        rng = np.random.default_rng(42)
        while len(all_points) < n:
            p = rng.normal(size=3)
            p /= np.linalg.norm(p)
            p *= spacing * (1 + rng.uniform(0, 1))
            all_points = np.vstack([all_points, p])
        return all_points[:n]


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 1000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
    # If not enough, just add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def construct(instance):
    """Multi-start structured seeds + L-BFGS-B relaxation with time budget."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Generate diverse starting geometries
    seeds = []
    seeds.append(_fcc_cluster(size))
    seeds.append(_hcp_cluster(size))
    seeds.append(_icosahedral_cluster(size))
    seeds.append(_random_compact(size, seed=size))
    seeds.append(_random_compact(size, seed=size + 1))
    seeds.append(_random_compact(size, seed=size + 2))
    
    best_energy = np.inf
    best_positions = None
    
    for i, seed_pos in enumerate(seeds):
        if time.time() > deadline:
            break
        
        # Validate seed
        if not np.all(np.isfinite(seed_pos)):
            continue
        if len(seed_pos) != size:
            continue
        # Check minimum distance
        min_dist = np.inf
        for j in range(size):
            for k in range(j + 1, size):
                d = np.linalg.norm(seed_pos[j] - seed_pos[k])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.35:
            # Perturb to fix
            rng = np.random.default_rng(i + 100)
            for j in range(size):
                for k in range(j + 1, size):
                    if np.linalg.norm(seed_pos[j] - seed_pos[k]) < 0.35:
                        seed_pos[k] += rng.uniform(-0.5, 0.5, 3)
        
        try:
            energy, positions = relax(seed_pos, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: return random compact
        best_positions = _random_compact(size, seed=size + 10)
    
    # Final validation
    if not np.all(np.isfinite(best_positions)):
        best_positions = _random_compact(size, seed=size + 20)
    
    return best_positions
# EVOLVE-BLOCK-END
