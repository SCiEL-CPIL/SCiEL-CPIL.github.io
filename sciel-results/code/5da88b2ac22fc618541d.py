"""Sutton-Chen cluster global minima search using structured templates and parallel relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, a=1.0):
    """Generate n positions from an FCC lattice fragment, compacted to a sphere."""
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    # Estimate side length
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    candidates = np.array(candidates)
    # Center
    center = candidates.mean(axis=0)
    candidates -= center
    # Sort by distance from center, take closest n
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n]
    points = candidates[idx]
    # Scale to approximate cluster radius
    if n > 0:
        max_r = np.max(np.linalg.norm(points, axis=1))
        if max_r > 0:
            target_r = 0.72 * n ** (1.0/3.0) * 0.75
            points *= target_r / max_r
    return points


def _hcp_positions(n, a=1.0):
    """Generate n positions from an HCP lattice fragment."""
    # HCP basis in 2D hexagonal layer, stacked
    # Simplified: use a hexagonal close-packed arrangement
    points = []
    # Generate layers
    layer_spacing = a * np.sqrt(2/3)
    hex_spacing = a
    # Number of atoms per layer
    layers = max(1, int(np.ceil(n ** (1/3))))
    per_layer = max(1, n // layers)
    
    for layer in range(layers):
        z = layer * layer_spacing
        # Hexagonal arrangement in each layer
        max_r = int(np.ceil(np.sqrt(per_layer / np.pi * 4 / np.sqrt(3)))) + 1
        layer_points = []
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                # Hexagonal grid
                x = hex_spacing * (i + 0.5 * (j % 2))
                y = hex_spacing * np.sqrt(3) / 2 * j
                r = np.sqrt(x**2 + y**2)
                if r <= hex_spacing * max_r * 0.8:
                    layer_points.append([x, y, z])
        
        if layer % 2 == 1:
            # Offset layer
            offset = hex_spacing / 2
            layer_points = [[p[0] + offset, p[1], p[2]] for p in layer_points]
        
        points.extend(layer_points)
    
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 3))
    
    # Center
    center = points.mean(axis=0)
    points -= center
    
    # Take closest n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:min(n, len(points))]
    points = points[idx]
    
    # If we have too few, pad with random
    while len(points) < n:
        r = 0.72 * n ** (1.0/3.0)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-np.pi/2, np.pi/2)
        p = [r * np.cos(phi) * np.cos(theta), r * np.cos(phi) * np.sin(theta), r * np.sin(phi)]
        points = np.vstack([points, [p]])
    
    points = points[:n]
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        target_r = 0.72 * n ** (1.0/3.0) * 0.75
        points *= target_r / max_r
    return points


def _icosahedron_positions(n, a=1.0):
    """Generate positions on concentric icosahedral shells."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]  # Center
    shells = [1.0, 1.8, 2.6, 3.4, 4.2]  # Shell radii
    
    shell_idx = 0
    while len(points) < n and shell_idx < len(shells):
        r = shells[shell_idx]
        # Add vertices of icosahedron at this radius
        for v in verts:
            if len(points) >= n:
                break
            points.append(r * v)
        shell_idx += 1
    
    # If still not enough, add more points on outer shell
    while len(points) < n:
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-np.pi/2, np.pi/2)
        r = shells[-1] * 1.2
        p = [r * np.cos(phi) * np.cos(theta), r * np.cos(phi) * np.sin(theta), r * np.sin(phi)]
        points.append(p)
    
    points = np.array(points[:n])
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        target_r = 0.72 * n ** (1.0/3.0) * 0.75
        points *= target_r / max_r
    return points


def _random_compact(n, spacing=0.72, seed=None):
    """Generate random compact cluster positions."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            min_dist = np.inf
            if points:
                pts = np.array(points)
                min_dist = np.min(np.linalg.norm(pts - p, axis=1))
            if min_dist >= 0.8 * spacing or len(points) == 0:
                points.append(p)
        attempts += 1
    
    # If not enough, fill with random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _relax_one(positions, potential, parameter, kind):
    """Relax a single cluster and return (energy, positions)."""
    try:
        energy, new_positions = relax(positions, potential, parameter)
        # Validate
        if not np.all(np.isfinite(new_positions)):
            return (np.inf, positions)
        # Check minimum distance
        n = len(new_positions)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(new_positions[i] - new_positions[j])
                if d < 0.35:
                    return (np.inf, positions)
        return (energy, new_positions)
    except Exception:
        return (np.inf, positions)


def construct(instance):
    """Search for global minimum using structured templates and parallel relaxation."""
    start_time = time.time()
    deadline = start_time + 280  # 280s deadline for safety (330s total limit)
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    parameter = kind
    
    # Generate diverse starting configurations
    starts = []
    
    # FCC-based
    starts.append(_fcc_positions(n))
    
    # HCP-based
    starts.append(_hcp_positions(n))
    
    # Icosahedral
    starts.append(_icosahedron_positions(n))
    
    # Random compact (multiple seeds)
    for seed in [n, n+1, n+2, n+3, n+4, n+5]:
        starts.append(_random_compact(n, seed=seed))
    
    # Parallel relaxation
    best_energy = np.inf
    best_positions = starts[0].copy()
    
    def process_start(idx_pos):
        idx, pos = idx_pos
        return _relax_one(pos, potential, parameter, kind)
    
    # Use ThreadPoolExecutor with up to 4 threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_start, (i, pos)): i for i, pos in enumerate(starts)}
        
        for future in as_completed(futures):
            if time.time() > deadline:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                pass
    
    # If we have time, do more random starts
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        for seed in range(n+10, n+30):
            if time.time() > deadline:
                break
            pos = _random_compact(n, seed=seed)
            energy, positions = _relax_one(pos, potential, parameter, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    # Final validation
    if not np.all(np.isfinite(best_positions)):
        # Fallback to a simple valid configuration
        best_positions = _random_compact(n, seed=n)
    
    # Ensure no two atoms are closer than 0.35
    n_pts = len(best_positions)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < 0.35:
                # Nudge apart
                mid = (best_positions[i] + best_positions[j]) / 2
                diff = best_positions[i] - best_positions[j]
                if np.linalg.norm(diff) < 1e-10:
                    diff = np.array([1.0, 0, 0])
                diff /= np.linalg.norm(diff)
                best_positions[i] = mid + diff * 0.18
                best_positions[j] = mid - diff * 0.18
    
    return best_positions
# EVOLVE-BLOCK-END
