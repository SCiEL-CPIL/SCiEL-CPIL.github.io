"""Sutton-Chen transition-metal cluster global minima - multi-start with diverse initial geometries."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n):
    """Generate a truncated FCC cluster with n atoms."""
    # FCC lattice points: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a/2
    # Nearest neighbor distance in FCC is a/sqrt(2), so if we want nn ~ 1.0, a = sqrt(2)
    # Let's use a = 1.414 (sqrt(2)) so nn = 1.0
    a = 1.414
    # Generate FCC points in a cube
    points = []
    # Estimate cube size needed for n atoms
    # FCC density: 4 atoms per a^3, so volume ~ n * a^3 / 4
    # Side ~ (n * a^3 / 4)^(1/3)
    side = (n * a**3 / 4.0) ** (1.0/3.0) * 1.5  # add margin
    n_half = int(side / (a/2)) + 2
    
    fcc_bases = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
    candidates = []
    for i in range(-n_half, n_half+1):
        for j in range(-n_half, n_half+1):
            for k in range(-n_half, n_half+1):
                for bi, bj, bk in fcc_bases:
                    x = (i + bi) * a
                    y = (j + bj) * a
                    z = (k + bk) * a
                    r = np.sqrt(x*x + y*y + z*z)
                    candidates.append((r, np.array([x, y, z])))
    
    # Sort by distance from center and take first n
    candidates.sort(key=lambda t: t[0])
    points = [c[1] for c in candidates[:n]]
    
    # Center the cluster
    if len(points) > 0:
        center = np.mean(points, axis=0)
        points = [p - center for p in points]
    
    return np.array(points)


def _hcp_cluster(n):
    """Generate a truncated HCP cluster with n atoms."""
    # HCP: two interpenetrating FCC-like lattices
    a = 1.414  # same as FCC for similar nn distance
    c = a * np.sqrt(8.0/3.0)  # ideal c/a ratio
    
    points = []
    n_half = int((n * a**3 / 4.0) ** (1.0/3.0) * 1.5 / (a/2)) + 2
    
    # Layer 1: hexagonal close packing
    for layer in range(-n_half, n_half+1):
        z = layer * c
        for i in range(-n_half, n_half+1):
            for j in range(-n_half, n_half+1):
                # Hexagonal lattice in xy plane
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                r = np.sqrt(x*x + y*y + z*z)
                points.append((r, np.array([x, y, z])))
    
    points.sort(key=lambda t: t[0])
    points = [p[1] for p in points[:n]]
    
    if len(points) > 0:
        center = np.mean(points, axis=0)
        points = [p - center for p in points]
    
    return np.array(points)


def _spherical_cluster(n):
    """Generate a spherical cluster with atoms on concentric shells."""
    # Mackay icosahedral-like: shells at radii ~ 1, 2, 3, ...
    # Number of atoms per shell ~ 4*pi*r^2 / (nn^2 * sqrt(3)/2)
    points = []
    shell_radii = [0.0]  # center
    shell_counts = [1]
    
    r = 1.0
    while len(points) < n:
        # Approximate number of atoms on shell of radius r
        area = 4 * np.pi * r**2
        # Packing density for close-packed surface ~ 0.907
        count = int(area / (0.75 * 1.0**2)) + 1  # rough estimate
        if len(points) + count > n:
            count = n - len(points)
        if count <= 0:
            break
        
        # Distribute points on sphere using Fibonacci sphere
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(count):
            z = 1 - 2 * (i + 0.5) / count
            radius_xy = np.sqrt(1 - z*z)
            theta = golden_angle * i
            x = np.cos(theta) * radius_xy
            y = np.sin(theta) * radius_xy
            points.append(np.array([x, y, z]) * r)
        
        shell_radii.append(r)
        r += 1.0
    
    if len(points) > 0:
        center = np.mean(points, axis=0)
        points = [p - center for p in points]
    
    return np.array(points)


def _random_compact(n, rng, spacing=0.8):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points]) if points else np.inf
            if min_dist >= 0.7 * spacing:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, just add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _perturbed_cluster(base, n, rng, scale=0.1):
    """Create a perturbed version of a base cluster."""
    if len(base) != n:
        return base
    perturbed = base + rng.normal(0, scale, size=base.shape)
    return perturbed


def construct(instance):
    """Multi-start optimization with diverse initial geometries."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC-based
    try:
        fcc = _fcc_cluster(size)
        if len(fcc) == size and np.all(np.isfinite(fcc)):
            initial_configs.append(fcc)
    except:
        pass
    
    # 2. HCP-based
    try:
        hcp = _hcp_cluster(size)
        if len(hcp) == size and np.all(np.isfinite(hcp)):
            initial_configs.append(hcp)
    except:
        pass
    
    # 3. Spherical
    try:
        sph = _spherical_cluster(size)
        if len(sph) == size and np.all(np.isfinite(sph)):
            initial_configs.append(sph)
    except:
        pass
    
    # 4-8. Random compact with different seeds
    for seed_offset in range(5):
        rng = np.random.default_rng(size + seed_offset * 1000)
        try:
            rand = _random_compact(size, rng)
            if len(rand) == size and np.all(np.isfinite(rand)):
                initial_configs.append(rand)
        except:
            pass
    
    # 9-12. Perturbed versions of best initial configs
    for i, base in enumerate(initial_configs[:4]):
        if time.time() > deadline - 60:
            break
        rng = np.random.default_rng(size + 5000 + i)
        try:
            pert = _perturbed_cluster(base, size, rng, scale=0.15)
            if len(pert) == size and np.all(np.isfinite(pert)):
                initial_configs.append(pert)
        except:
            pass
    
    if not initial_configs:
        # Fallback: simple random
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0/3.0) * 0.75
        points = rng.uniform(-radius, radius, (size, 3))
        initial_configs.append(points)
    
    # Relax each initial configuration
    best_energy = np.inf
    best_positions = initial_configs[0]
    
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        try:
            energy, positions = relax(config, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            continue
    
    # Final check: ensure validity
    if not np.all(np.isfinite(best_positions)):
        # Fallback to a simple valid configuration
        rng = np.random.default_rng(12345)
        radius = 0.72 * size ** (1.0/3.0) * 0.75
        best_positions = rng.uniform(-radius, radius, (size, 3))
        try:
            energy, positions = relax(best_positions, potential, kind)
            if np.isfinite(energy):
                best_positions = positions
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
