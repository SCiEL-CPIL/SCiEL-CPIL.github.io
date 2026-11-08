"""Multi-start search with geometric templates and adaptive perturbation for Sutton-Chen clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_fcc_template(n, rng):
    """Generate FCC-like cluster by taking n closest points to center."""
    points = []
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz])
                    points.append(p)
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points

def _generate_hcp_template(n, rng):
    """Generate HCP-like cluster."""
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # AB stacking
                offset = (0.5, 0.0, 0.0) if k % 2 == 0 else (0.5/np.sqrt(3), 0.5/np.sqrt(3), 0.0)
                p = np.array([i + offset[0], j + offset[1], k * c / a + offset[2]])
                points.append(p)
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points

def _generate_bcc_template(n, rng):
    """Generate BCC-like cluster."""
    points = []
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                p = np.array([i, j, k])
                points.append(p)
                # BCC center
                p_center = np.array([i + 0.5, j + 0.5, k + 0.5])
                points.append(p_center)
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points

def _generate_icosahedral_template(n, rng):
    """Generate icosahedral cluster with shells."""
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    shell = 1
    while len(points) < n:
        for v in verts:
            if len(points) < n:
                points.append(shell * v)
        if len(points) < n:
            # Add edge midpoints
            for i in range(12):
                v1 = verts[i]
                v2 = verts[(i + 1) % 12]
                for t in np.linspace(0.25, 0.75, 3):
                    if len(points) < n:
                        p = shell * (v1 * (1 - t) + v2 * t)
                        p = p / np.linalg.norm(p) * shell
                        points.append(p)
        shell += 1
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points

def _generate_random_compact(n, rng, min_dist=0.5):
    """Generate random compact cluster."""
    points = []
    radius = 0.7 * n ** (1.0 / 3.0) + 1.0
    max_attempts = 2000 * n
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        if not points:
            points.append(p)
        else:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
            else:
                attempts += 1
    if len(points) < n:
        points = np.array(_generate_fcc_template(n, rng))
    else:
        points = np.array(points)
        points -= points.mean(axis=0)
    return points

def _adaptive_perturbation(pos, rng, base_step=0.15, scale_factor=1.0):
    """Perturb positions with adaptive step size based on local density."""
    n = len(pos)
    # Compute local density for each atom
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(pos, pos)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Local density: number of neighbors within 1.5 units
    neighbor_counts = (dist_matrix < 1.5).sum(axis=1)
    avg_density = neighbor_counts.mean()
    
    # Atoms in dense regions get smaller steps, sparse regions get larger
    density_factor = 1.0 + 0.3 * (neighbor_counts - avg_density) / (avg_density + 1e-10)
    density_factor = np.clip(density_factor, 0.5, 2.0)
    
    steps = base_step * scale_factor * density_factor
    perturbed = pos + rng.normal(0, 1, pos.shape) * steps
    return perturbed

def _worker_chain(initial_pos, kind, rng_seed, time_limit):
    """
    Run a search chain: relax, then adaptive perturbation + relax iterations.
    Returns (energy, positions).
    """
    start_time = time.time()
    rng = np.random.default_rng(rng_seed)
    
    # Initial relaxation
    current_energy, current_pos = relax(initial_pos, 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    base_step = 0.15
    scale = 1.0
    
    while time.time() - start_time < time_limit:
        # Adaptive perturbation
        perturbed = _adaptive_perturbation(current_pos, rng, base_step, scale)
        
        # Relax
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            scale *= 0.9  # Contract on improvement
        else:
            delta_e = new_energy - current_energy
            # Accept with probability based on energy difference
            accept_prob = np.exp(-delta_e / 0.05)
            if rng.random() < accept_prob:
                current_energy = new_energy
                current_pos = new_pos
                scale *= 1.1  # Expand on uphill acceptance
            else:
                scale *= 0.95
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
        
        # Validity check
        min_dist = pdist(current_pos).min()
        if min_dist < 0.35:
            # Spread out
            current_pos += rng.normal(0, 0.1, current_pos.shape)
            current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
    
    return best_energy, best_pos

def construct(instance):
    """
    Multi-start search with geometric templates and adaptive perturbation.
    """
    start_time = time.time()
    deadline = start_time + 285  # 15s safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    rng_master = np.random.default_rng(42)
    
    # Generate diverse initial structures
    templates = [
        _generate_fcc_template(n, rng_master),
        _generate_hcp_template(n, rng_master),
        _generate_bcc_template(n, rng_master),
        _generate_icosahedral_template(n, rng_master),
        _generate_random_compact(n, rng_master, 0.6),
        _generate_random_compact(n, rng_master, 0.5),
    ]
    
    # We have 4 threads. Use each thread for a different template.
    # Each thread runs one chain with adaptive perturbation.
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            start_pos = templates[i]
            worker_time = (deadline - time.time()) / 4 - 3
            if worker_time < 10:
                worker_time = 10
            seed = 200 + i
            futures.append(executor.submit(_worker_chain, start_pos, kind, seed, worker_time))
        
        for future in futures:
            try:
                energy, pos = future.result()
                if np.isfinite(energy):
                    results.append((energy, pos))
            except Exception:
                pass
    
    if not results:
        return _generate_fcc_template(n, rng_master)
    
    best_energy, best_pos = min(results, key=lambda x: x[0])
    
    # Final validity check
    min_dist = pdist(best_pos).min()
    if min_dist < 0.35:
        for _ in range(20):
            best_pos += np.random.default_rng(123).normal(0, 0.05, best_pos.shape)
            min_dist = pdist(best_pos).min()
            if min_dist >= 0.35:
                break
        try:
            best_energy, best_pos = relax(best_pos, 'sutton_chen', kind)
        except:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
