"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    """Return normalized icosahedron vertices (12 points)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def _icosahedron_edges():
    """Return edge midpoint directions (30 edges, normalized)."""
    verts = _icosahedron_vertices()
    dists = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    edge_pairs = [(dists[k][1], dists[k][2]) for k in range(30)]
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        edge_dirs.append(mid / np.linalg.norm(mid))
    return np.array(edge_dirs)


def _icosahedron_face_centers():
    """Return 20 face center directions (normalized)."""
    verts = _icosahedron_vertices()
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 1],
        [1, 6, 5], [1, 5, 7], [1, 7, 2], [1, 2, 0],
        [2, 7, 5], [2, 5, 9], [2, 9, 3], [2, 3, 1],
        [3, 9, 4], [3, 4, 8], [3, 8, 11], [3, 11, 7], [3, 7, 2],
        [4, 9, 11], [4, 11, 8],
    ]
    centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return np.array(centers)


def _build_icosahedral_cluster(n, seed=0, radii_scale=1.0):
    """Build a cluster with icosahedral core + surface atoms at high-symmetry sites."""
    rng = np.random.default_rng(seed)
    positions = []
    
    # Core icosahedron: 12 atoms + 1 center atom if n >= 13
    core_radius = 0.72 * radii_scale
    core = _icosahedron_vertices() * core_radius
    positions.extend(core)
    
    # Add center atom if n >= 13
    if n >= 13:
        positions.append([0, 0, 0])
    
    n_remaining = n - len(positions)
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell: 30 edge-midpoint positions
    edge_dirs = _icosahedron_edges()
    shell2_radius = 1.05 * radii_scale
    shell2 = edge_dirs * shell2_radius
    
    n_add2 = min(n_remaining, 30)
    positions.extend(shell2[:n_add2])
    n_remaining -= n_add2
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Third shell: 20 face centers
    face_dirs = _icosahedron_face_centers()
    shell3_radius = 1.35 * radii_scale
    shell3 = face_dirs * shell3_radius
    
    n_add3 = min(n_remaining, 20)
    positions.extend(shell3[:n_add3])
    n_remaining -= n_add3
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Fourth shell: Fibonacci sphere for remaining atoms
    if n_remaining > 0:
        indices = np.arange(n_remaining)
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_remaining)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        shell4_radius = 1.55 * radii_scale
        shell4 = np.column_stack([
            shell4_radius * np.sin(phi) * np.cos(theta),
            shell4_radius * np.sin(phi) * np.sin(theta),
            shell4_radius * np.cos(phi)
        ])
        positions.extend(shell4)
    
    return np.array(positions[:n])


def _build_fcc_fragment(n, seed=0):
    """Build FCC lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)[:n]
    positions = candidates[indices]
    positions -= np.mean(positions, axis=0)
    return positions


def _build_hcp_fragment(n, seed=0):
    """Build HCP lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    c = a * np.sqrt(8.0/3.0)
    layer_spacing = c / 2.0
    
    positions = []
    n_layers = int(np.ceil(np.sqrt(n / 3.0))) + 2
    
    for layer in range(n_layers):
        z = layer * layer_spacing
        offset = np.array([0, 0, 0]) if layer % 2 == 0 else np.array([a/2, a/(2*np.sqrt(3)), 0])
        
        max_r = int(np.ceil(np.sqrt(n / (3 * np.pi / (a**2 * 0.866)))))
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                x = a * (i + j / 2) + offset[0]
                y = a * np.sqrt(3) / 2 * j + offset[1]
                p = np.array([x, y, z])
                if len(positions) < n:
                    positions.append(p)
                else:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _build_random_cluster(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _validate(positions, min_dist=0.35):
    """Check if positions are valid (finite, no overlaps)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
        if np.any(dists < min_dist):
            return False
    return True


def _adaptive_basin_hopping(positions, potential, kind, rng=None, deadline=None, max_iter=50):
    """
    Adaptive Basin Hopping: 
    - Starts with larger steps, decreases as energy converges.
    - Stops if no improvement for a while or deadline reached.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initial relaxation
    energy, pos = relax(positions, potential, kind)
    
    step_size = 0.5
    min_step = 0.05
    no_improve_count = 0
    max_no_improve = 10
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        perturbed = pos + rng.uniform(-step_size, step_size, pos.shape)
        
        if not _validate(perturbed):
            continue
        
        try:
            e2, pos2 = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if not _validate(pos2):
            continue
        
        if e2 < energy:
            energy = e2
            pos = pos2
            no_improve_count = 0
            # Decay step size slowly
            step_size *= 0.95
        else:
            no_improve_count += 1
            # If stuck, try a bigger step occasionally
            if no_improve_count % 5 == 0:
                step_size = max(min_step, step_size * 0.8)
        
        if no_improve_count > max_no_improve:
            break
        
        if step_size < min_step:
            break

    return energy, pos


def construct(instance):
    """Structure prediction with adaptive basin hopping for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    start_time = time.time()
    # We have ~50s per instance if we split 300s by 6. 
    # But we don't know how many instances are called in a row. 
    # The prompt says "all calls together within 300 s". 
    # So we should track a global deadline? No, construct is called per instance.
    # We assume each call has its own budget, but the total is 300s.
    # To be safe, we'll use a fixed 45s budget per instance.
    deadline = start_time + 45.0
    
    best_energy = np.inf
    best_positions = None
    
    def try_strategy(positions, seed_offset=0):
        nonlocal best_energy, best_positions
        if not _validate(positions):
            return
        
        rng = np.random.default_rng(seed_offset)
        try:
            energy, relaxed = _adaptive_basin_hopping(positions, potential, kind, 
                                                      rng=rng, deadline=deadline, max_iter=30)
        except Exception:
            return
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 1: Icosahedral shells with varied radii
    radii_scales = [0.9, 1.0, 1.1]
    for rs in radii_scales:
        if time.time() > deadline:
            break
        positions = _build_icosahedral_cluster(size, seed=0, radii_scale=rs)
        try_strategy(positions, seed_offset=100 + int(rs*10))
        
    # Strategy 2: FCC fragments
    for seed in range(2):
        if time.time() > deadline:
            break
        positions = _build_fcc_fragment(size, seed=seed)
        try_strategy(positions, seed_offset=200 + seed)
        
    # Strategy 3: HCP fragments
    if time.time() < deadline - 10:
        for seed in range(2):
            if time.time() > deadline:
                break
            positions = _build_hcp_fragment(size, seed=seed)
            try_strategy(positions, seed_offset=300 + seed)
            
    # Strategy 4: Random compact
    if time.time() < deadline - 10:
        for seed in range(2):
            if time.time() > deadline:
                break
            positions = _build_random_cluster(size, seed=seed)
            try_strategy(positions, seed_offset=400 + seed)
            
    # Fallback
    if best_positions is None:
        positions = _build_icosahedral_cluster(size, seed=0)
        if not _validate(positions):
            positions = _build_random_cluster(size, seed=42)
        try:
            energy, best_positions = relax(positions, potential, kind)
        except Exception:
            best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
