"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global time management
_GLOBAL_START_TIME = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START_TIME + 290.0  # 10s safety margin

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
    # Standard icosahedron faces
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 1],
        [1, 6, 5], [1, 5, 7], [1, 7, 2], [1, 2, 0],
        [2, 7, 5], [2, 5, 9], [2, 9, 3], [2, 3, 1],
        [3, 9, 4], [3, 4, 8], [3, 8, 11], [3, 11, 7], [3, 7, 2],
        [4, 9, 11], [4, 11, 8]
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
    
    # Core icosahedron: 12 atoms
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

def _basin_hopping_worker(initial_pos, potential, kind, seed, deadline):
    """
    Worker function for parallel basin hopping.
    Returns (energy, positions) of the best found in this worker.
    """
    rng = np.random.default_rng(seed)
    
    # Initial relaxation
    try:
        energy, pos = relax(initial_pos, potential, kind)
    except Exception:
        return np.inf, None
    
    if not _validate(pos):
        return np.inf, None

    best_energy = energy
    best_pos = pos.copy()
    
    step_size = 0.5
    min_step = 0.05
    no_improve_count = 0
    max_no_improve = 15
    
    while time.time() < deadline:
        # Perturb
        perturbed = pos + rng.uniform(-step_size, step_size, pos.shape)
        
        if not _validate(perturbed):
            # If invalid, try a smaller perturbation or reset
            perturbed = pos + rng.uniform(-0.1, 0.1, pos.shape)
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
            step_size *= 0.98
            if step_size < min_step:
                step_size = min_step
            
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        else:
            no_improve_count += 1
            # If stuck, kick the step size
            if no_improve_count >= max_no_improve:
                step_size = 0.5
                no_improve_count = 0
            else:
                step_size *= 0.99
                if step_size < min_step:
                    step_size = min_step

    return best_energy, best_pos

def construct(instance):
    """Structure prediction with parallel basin hopping for Sutton-Chen clusters."""
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    # Reset global time if this is a new session? 
    # The prompt implies construct is called for multiple instances.
    # We should probably track time per instance or use a global budget.
    # "all calls together within 300 s" implies a global budget for the whole process.
    # However, construct is the entry point. If it's called 6 times, we need to manage the total time.
    # Since we don't know if it's the first call or the 6th, we should assume a shared budget.
    # But we can't easily share state between calls if they are separate processes?
    # Usually, in these tasks, the function is called sequentially in the same process.
    # Let's assume the global deadline is set on the first call and maintained.
    if not hasattr(construct, '_global_deadline_set'):
        construct._global_deadline_set = True
        construct._global_start = time.time()
        construct._global_deadline = construct._global_start + 290.0
        construct._instance_count = 0
    
    construct._instance_count += 1
    
    # Estimate remaining time per instance based on how many are left?
    # We don't know how many total. The prompt lists 6 specific cases.
    # Let's assume there are 6 calls.
    # If we have used T time, we have (290 - T) left.
    # We should allocate roughly (290 - T) / (6 - construct._instance_count + 1) to this instance.
    # But to be safe, let's just use the global deadline and ensure we don't exceed it.
    
    deadline = construct._global_deadline
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    # Generate diverse initial structures
    inits = []
    inits.append(_build_icosahedral_cluster(size, seed=0))
    inits.append(_build_fcc_fragment(size, seed=0))
    inits.append(_build_hcp_fragment(size, seed=0))
    inits.append(_build_random_cluster(size, seed=0))
    
    # Add a few more random/structured variants for diversity if time permits
    # For now, 4 workers is the limit.
    
    results = []
    
    # Use ThreadPoolExecutor with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, init in enumerate(inits):
            # Assign a unique seed to each worker
            seed = 1000 + i
            future = executor.submit(_basin_hopping_worker, init, potential, kind, seed, deadline)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, pos = future.result()
                if pos is not None and _validate(pos):
                    results.append((energy, pos))
            except Exception:
                pass
    
    if not results:
        # Fallback: just relax the first initialization
        try:
            energy, pos = relax(inits[0], potential, kind)
            if _validate(pos):
                return pos
        except Exception:
            return inits[0]
    
    # Find the best result
    best_energy, best_pos = min(results, key=lambda x: x[0])
    
    return best_pos

# EVOLVE-BLOCK-END
