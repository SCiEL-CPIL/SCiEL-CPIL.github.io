"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites by distance from center."""
    if n <= 0:
        return np.empty((0, 3))
    # Generate fcc lattice points
    sites = []
    # fcc basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    # Generate cubic lattice points
    max_shell = int(np.ceil(n ** (1.0/3.0))) + 2
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * 1.0 + b
                    sites.append(p)
    sites = np.array(sites)
    # Remove duplicates
    sites = np.unique(np.round(sites, 10), axis=0)
    # Sort by distance from center
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    sites = sites[idx]
    # Center the cluster
    sites = sites - sites[:n].mean(axis=0)
    # Take first n
    sites = sites[:n] * spacing
    return sites


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites by distance from center."""
    if n <= 0:
        return np.empty((0, 3))
    # Generate hcp lattice points
    # hcp: two atoms per unit cell, close-packed layers
    # Use a simple approach: generate points in close-packed arrangement
    sites = []
    a = 1.0  # in-plane spacing
    c = 1.633 * a  # c/a ratio for ideal hcp
    max_shell = int(np.ceil(n ** (1.0/3.0))) + 2
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c / 2.0
        # Each layer is a triangular lattice
        offset = (layer % 2) * 0.5
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = (i + offset) * a
                y = j * a * np.sqrt(3) / 2.0
                sites.append([x, y, z])
    sites = np.array(sites)
    sites = np.unique(np.round(sites, 10), axis=0)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    sites = sites[idx]
    sites = sites - sites[:n].mean(axis=0)
    sites = sites[:n] * spacing
    return sites


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral cluster with concentric shells."""
    if n <= 0:
        return np.empty((0, 3))
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2.0
    ico_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    # Add center
    points = [np.zeros(3)]
    
    # First shell: 12 icosahedron vertices
    for v in ico_verts:
        points.append(v)
    
    # If we need more, add second shell (octahedron-like or more icosahedral)
    if n > 13:
        # Second shell: 30 points (edges of icosahedron midpoints) or use a larger icosahedron
        # Simpler: add points at distance ~2 from center on icosahedral directions
        # Use a dodecahedron-like shell or just scale up
        # For simplicity, add more icosahedral vertices at larger radius
        shell2_scale = 1.8
        for v in ico_verts:
            if len(points) < n:
                points.append(v * shell2_scale)
        # If still need more, add a third shell
        if n > 25:
            shell3_scale = 2.5
            for v in ico_verts:
                if len(points) < n:
                    points.append(v * shell3_scale)
            # Add more points by rotating
            if n > 37:
                # Add points at intermediate positions
                for i in range(len(ico_verts)):
                    if len(points) >= n:
                        break
                    v1 = ico_verts[i]
                    v2 = ico_verts[(i+1) % len(ico_verts)]
                    mid = (v1 + v2) / np.linalg.norm(v1 + v2) * shell3_scale
                    points.append(mid)
    
    points = np.array(points[:n])
    points = points - points.mean(axis=0)
    points = points * spacing
    return points


def _random_cluster(n, seed, min_dist=0.6, spacing=0.72):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    max_radius = spacing * n ** (1.0 / 3.0) * 1.2
    points = []
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        points.append(p)
    points = np.array(points)
    return points


def construct(instance):
    """Best of multiple structured and random starting clusters, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    # Time budget management
    t_start = time.time()
    total_budget = 280.0  # Leave 50s safety margin for 330s kill
    per_attempt_budget = 45.0  # Max time per relaxation attempt
    
    # Check if this is a Sutton-Chen potential
    if potential == "sutton_chen":
        spacing = 0.72
    elif potential == "morse":
        spacing = 1.0
    elif potential == "dzugutov":
        spacing = 1.1
    else:
        spacing = 0.72
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. FCC-like cluster
    try:
        fcc_pts = _fcc_cluster(size, spacing)
        if fcc_pts.shape[0] == size and np.all(np.isfinite(fcc_pts)):
            initial_configs.append(fcc_pts)
    except Exception:
        pass
    
    # 2. HCP-like cluster
    try:
        hcp_pts = _hcp_cluster(size, spacing)
        if hcp_pts.shape[0] == size and np.all(np.isfinite(hcp_pts)):
            initial_configs.append(hcp_pts)
    except Exception:
        pass
    
    # 3. Icosahedral cluster
    try:
        ico_pts = _icosahedral_cluster(size, spacing)
        if ico_pts.shape[0] == size and np.all(np.isfinite(ico_pts)):
            initial_configs.append(ico_pts)
    except Exception:
        pass
    
    # 4-7. Random clusters with different seeds
    for seed in [size, size + 1000, size + 2000, size + 3000]:
        try:
            rand_pts = _random_cluster(size, seed, min_dist=0.6, spacing=spacing)
            if rand_pts.shape[0] == size and np.all(np.isfinite(rand_pts)):
                initial_configs.append(rand_pts)
        except Exception:
            pass
    
    # Remove duplicates (approximately)
    unique_configs = []
    for cfg in initial_configs:
        is_dup = False
        for ucfg in unique_configs:
            if np.allclose(cfg, ucfg, atol=0.1):
                is_dup = True
                break
        if not is_dup:
            unique_configs.append(cfg)
    
    initial_configs = unique_configs
    
    # Relax each configuration and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, cfg in enumerate(initial_configs):
        # Check time
        elapsed = time.time() - t_start
        if elapsed > total_budget - 5.0:
            break
        
        # Check minimum distance
        if cfg.shape[0] > 1:
            from scipy.spatial.distance import pdist
            min_d = pdist(cfg, metric='euclidean').min()
            if min_d < 0.35:
                continue
        
        try:
            energy, positions = relax(cfg, potential, kind)
            
            # Validate result
            if not np.all(np.isfinite(positions)):
                continue
            if positions.shape[0] != size:
                continue
            if positions.shape[1] != 3:
                continue
            
            # Check minimum distance in result
            if positions.shape[0] > 1:
                from scipy.spatial.distance import pdist
                min_d = pdist(positions, metric='euclidean').min()
                if min_d < 0.35:
                    continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                
        except Exception:
            continue
    
    # If no valid result, return a simple valid configuration
    if best_positions is None:
        # Fallback: linear arrangement
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * spacing
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
