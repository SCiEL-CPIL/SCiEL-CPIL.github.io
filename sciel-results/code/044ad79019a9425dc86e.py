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
                    pos = np.array([i, j, k]) + b
                    sites.append(pos * spacing)
    sites = np.array(sites)
    # Sort by distance from origin
    dists = np.linalg.norm(sites, axis=1)
    order = np.argsort(dists)
    sites = sites[order[:n]]
    return sites


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    # HCP: two interleaved triangular lattices
    a = spacing
    c = a * np.sqrt(2.0 / 3.0)  # c/a ratio for ideal HCP
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = i * a + (j % 2) * a * 0.5
                y = j * a * np.sqrt(3.0) / 2.0
                if layer % 2 == 0:
                    pos = np.array([x, y, z])
                else:
                    pos = np.array([x + a / 3.0, y + a * np.sqrt(3.0) / 6.0, z])
                sites.append(pos)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    order = np.argsort(dists)
    sites = sites[order[:n]]
    return sites


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral-like cluster (Mackay icosahedron)."""
    if n <= 0:
        return np.empty((0, 3))
    # Mackay icosahedron: shells of 1, 12, 42, 92, ...
    # We'll generate points on concentric icosahedral shells
    phi = (1 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices (12 vertices)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Generate points by scaling icosahedron vertices at different radii
    sites = [np.zeros(3)]  # center
    shell_radii = [1.0, 1.618, 2.414]  # approximate shell radii
    shell_counts = [12, 42, 92]
    
    for r, count in zip(shell_radii, shell_counts):
        # Add scaled icosahedron vertices
        for v in ico_verts:
            sites.append(v * r * spacing)
        # For higher shells, add midpoints and other positions
        if count > 12:
            # Add edge midpoints and face centers for denser shells
            # This is a simplified approach
            for i in range(len(ico_verts)):
                for j in range(i + 1, len(ico_verts)):
                    mid = (ico_verts[i] + ico_verts[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 0.1:
                        mid = mid / mid_norm * r * spacing
                        sites.append(mid)
    
    sites = np.array(sites)
    # Deduplicate and take n closest to origin
    if len(sites) > n:
        dists = np.linalg.norm(sites, axis=1)
        order = np.argsort(dists)
        sites = sites[order[:n]]
    elif len(sites) < n:
        # Fill remaining with random points near the cluster
        rng = np.random.default_rng(42)
        radius = spacing * n ** (1.0 / 3.0) * 0.8
        while len(sites) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                sites = np.vstack([sites, p])
    return sites


def _random_compact(n, spacing=0.72, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place enough, fill with random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _validate(positions, min_dist=0.35):
    """Check if positions are valid (finite, no two atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized check
    for i in range(n):
        dists = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(dists < min_dist):
            return False
    return True


def construct(instance):
    """Multi-start structured seeds + L-BFGS-B relaxation with time budget."""
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    spacing = 0.72  # typical equilibrium distance for Sutton-Chen
    
    # Generate diverse starting configurations
    starts = []
    
    # FCC at different scales
    for scale in [0.9, 1.0, 1.1]:
        starts.append(_fcc_cluster(size, spacing * scale))
    
    # HCP at different scales
    for scale in [0.9, 1.0, 1.1]:
        starts.append(_hcp_cluster(size, spacing * scale))
    
    # Icosahedral
    starts.append(_icosahedral_cluster(size, spacing))
    starts.append(_icosahedral_cluster(size, spacing * 0.9))
    starts.append(_icosahedral_cluster(size, spacing * 1.1))
    
    # Random compact at different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        starts.append(_random_compact(size, spacing, seed))
    
    # Relax each start, track best
    best_energy = np.inf
    best_positions = None
    
    for i, start_pos in enumerate(starts):
        if time.time() > deadline:
            break
        
        if not _validate(start_pos):
            continue
        
        try:
            energy, positions = relax(start_pos, potential, kind)
        except Exception:
            continue
        
        if not np.all(np.isfinite(positions)):
            continue
        
        if not _validate(positions):
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # Polish pass: if we have time, try to refine the best result
    if best_positions is not None and time.time() < deadline - 30.0:
        try:
            # Do a second relaxation from the best result with slight perturbation
            rng = np.random.default_rng(123)
            perturbed = best_positions + rng.normal(0, 0.01, best_positions.shape)
            if _validate(perturbed):
                energy2, positions2 = relax(perturbed, potential, kind)
                if _validate(positions2) and energy2 < best_energy:
                    best_energy = energy2
                    best_positions = positions2
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: return a simple random cluster
        rng = np.random.default_rng(999)
        radius = spacing * size ** (1.0 / 3.0) * 0.8
        best_positions = rng.uniform(-radius, radius, (size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
