# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 300.0  # 300s total budget, return before kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_icosahedral(n, rng):
    """Generate an icosahedral-like cluster for polytetrahedral packing."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Build shells using icosahedral directions
    directions = list(ico_verts)
    
    # Add edge midpoints (normalized)
    for i in range(12):
        for j in range(i+1, 12):
            d = ico_verts[i] + ico_verts[j]
            d = d / np.linalg.norm(d)
            directions.append(d)
    
    # Add face centers (approximate by summing 3 vertices)
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d = ico_verts[i] + ico_verts[j] + ico_verts[k]
                norm = np.linalg.norm(d)
                if norm > 0.1:
                    directions.append(d / norm)
    
    # Deduplicate directions
    unique_dirs = [directions[0]]
    for d in directions[1:]:
        if all(np.linalg.norm(d - ud) > 0.1 for ud in unique_dirs):
            unique_dirs.append(d)
    directions = unique_dirs
    
    # Build shells
    shell_radii = [1.0, 1.8, 2.6, 3.4, 4.2, 5.0]
    dir_idx = 0
    for shell_r in shell_radii:
        for d in directions:
            if len(points) >= n:
                break
            noise = rng.normal(0, 0.05, 3)
            p = shell_r * d + noise
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        if len(points) >= n:
            break
    
    # If we still need more, add random points
    while len(points) < n:
        p = rng.uniform(-6, 6, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_fcc(n, rng):
    """Generate an FCC-like cluster."""
    a = 2.0
    points = []
    max_coord = int((n / 4.0) ** (1.0/3.0) * a) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for basis in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a, j*a, k*a]) + np.array(basis)
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    while len(points) < n:
        p = rng.uniform(-6, 6, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_random(n, rng):
    """Generate a random compact cluster."""
    radius = 1.2 * n ** (1.0/3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        attempts += 1
    
    # Fallback: fill with slightly perturbed positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Nudge to satisfy minimum distance
            for _ in range(100):
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
                    break
                p = p + rng.normal(0, 0.1, 3)
            else:
                p = np.array([len(points)*0.5, 0, 0])
                points.append(p)
    
    return np.array(points[:n])


def _simulated_annealing(positions, potential, rng, time_limit=40.0):
    """Simulated Annealing with local relaxation."""
    current_energy, current_pos = relax(positions, potential)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Cooling schedule parameters
    T_initial = 5.0
    T_final = 0.01
    # Estimate number of steps based on time limit
    # Assume each step (perturb + relax) takes ~0.1-0.5s depending on n
    # We will run until time limit is reached
    start_time = time.time()
    
    # Adaptive step size
    base_step = 0.5
    
    while (time.time() - start_time) < time_limit:
        # Cooldown temperature
        elapsed = time.time() - start_time
        # Exponential cooling
        T = T_initial * np.exp(-elapsed / (time_limit / 5.0))
        if T < T_final:
            T = T_final
            
        # Perturbation
        # Use larger steps initially, smaller later
        step_size = base_step * (T / T_initial + 0.2)
        delta = rng.normal(0, step_size, positions.shape)
        perturbed = current_pos + delta
        
        # Check minimum distance and fix if necessary
        diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        if np.min(dists) < 0.5:
            # Simple fix: move close pairs apart
            for _ in range(50):
                if np.min(dists) >= 0.5:
                    break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                i_idx, j_idx = min_idx
                if i_idx == j_idx:
                    continue
                diff = perturbed[i_idx] - perturbed[j_idx]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = rng.normal(0, 0.1, 3)
                push = 0.5 * (diff / dist)
                perturbed[i_idx] += push
                perturbed[j_idx] -= push
                diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                np.fill_diagonal(dists, np.inf)
        
        new_energy, new_pos = relax(perturbed, potential)
        
        # Acceptance criterion
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
        else:
            # Metropolis acceptance
            delta_e = new_energy - current_energy
            if T > 0 and rng.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
    return best_energy, best_pos


def _run_sa_worker(args):
    """Worker function for parallel SA."""
    positions, potential, seed, time_limit = args
    rng = np.random.default_rng(seed)
    return _simulated_annealing(positions, potential, rng, time_limit)


def construct(instance):
    """Find low-energy structures using parallel Simulated Annealing."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial structures
    generators = [
        lambda: _generate_icosahedral(size, rng),
        lambda: _generate_fcc(size, rng),
        lambda: _generate_random(size, rng),
        lambda: _generate_random(size, rng),
        lambda: _generate_random(size, rng),
        lambda: _generate_icosahedral(size, rng),
        lambda: _generate_fcc(size, rng),
        lambda: _generate_random(size, rng),
    ]
    
    # Prepare seeds
    seeds = []
    for gen in generators:
        try:
            pos = gen()
            if pos.shape[0] == size:
                seeds.append(pos)
        except Exception:
            pass
    
    # Ensure we have at least one seed
    if not seeds:
        seeds.append(_generate_random(size, rng))
    
    # Determine time per worker
    # We have ~50s per instance. Let's use 4 threads.
    # Each thread gets ~45s to be safe.
    time_per_worker = 45.0
    
    # Run in parallel
    best_energy = np.inf
    best_positions = None
    
    # Create tasks
    tasks = []
    for i, pos in enumerate(seeds):
        # Use different seeds for each worker to ensure diversity
        task_seed = size * 1000 + i
        tasks.append((pos, potential, task_seed, time_per_worker))
    
    # Limit to 4 workers
    max_workers = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_sa_worker, task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
    
    # Final relaxation
    if best_positions is not None:
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback
        best_positions = _generate_random(size, rng)
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
