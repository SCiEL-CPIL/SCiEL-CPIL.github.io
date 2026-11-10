"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
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
    # Shell 1: 12 vertices at r=1
    # Shell 2: 30 vertices (midpoints of edges) at r~1.05
    # Shell 3: 20 vertices (face centers) at r~0.95
    # Then repeat at larger radii
    
    # Generate more directions by combining icosahedral vertices
    directions = list(ico_verts)
    
    # Add edge midpoints (normalized)
    for i in range(12):
        for j in range(i+1, 12):
            d = ico_verts[i] + ico_verts[j]
            d = d / np.linalg.norm(d)
            directions.append(d)
    
    # Add face centers
    ico_faces = [
        [0,1,8],[0,8,4],[0,4,2],[0,2,6],[0,6,1],
        [1,6,9],[1,9,3],[1,3,5],[1,5,0],[2,4,10],[2,10,7],[2,7,6],
        [3,9,11],[3,11,5],[3,5,0],[4,8,10],[4,10,7],[4,7,2],
        [5,11,9],[5,9,1],[5,0,3],[6,7,10],[6,10,8],[6,1,9],
        [7,10,11],[7,11,3],[7,2,6],[8,11,10],[8,3,9],[8,0,4],
        [9,11,3],[9,2,7],[9,6,1],[10,11,7],[10,8,4],[10,5,9],
        [11,8,0],[11,5,3],[11,7,2],[8,1,5],[8,4,2],[8,0,3]
    ]
    # Just use all possible triples for simplicity
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


def _basin_hopping(positions, potential, rng, max_iterations=50, delta_e=0.15, T=0.3):
    """Basin hopping with adaptive step size."""
    current_energy, current_pos = relax(positions, potential)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for i in range(max_iterations):
        # Adaptive step size: start larger, shrink over iterations
        step = delta_e * (1.0 + 0.5 * np.cos(np.pi * i / max_iterations))
        
        # Random perturbation
        delta = rng.normal(0, step, positions.shape)
        perturbed = current_pos + delta
        
        # Check minimum distance vectorized
        diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        if np.min(dists) < 0.5:
            # Try to fix by moving close pairs apart
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
        
        # Accept if lower energy or with Boltzmann probability
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
        elif rng.random() < np.exp(-(new_energy - current_energy) / T):
            current_energy = new_energy
            current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy structures using basin hopping with diverse seeds."""
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
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Determine iterations per seed based on time budget
    time_per_seed = 25  # seconds
    max_iter_per_seed = 40
    
    # Try each generator with basin hopping
    for gi, gen in enumerate(generators):
        if _get_time_remaining() < time_per_seed:
            break
        
        try:
            positions = gen()
            if positions.shape[0] != size:
                continue
            
            # Scale iterations with remaining time
            remaining = _get_time_remaining()
            n_iter = min(max_iter_per_seed, int(remaining / (len(generators) - gi) * 0.8))
            n_iter = max(n_iter, 10)
            
            energy, pos = _basin_hopping(positions, potential, rng, 
                                         max_iterations=n_iter, delta_e=0.15)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            continue
    
    # If we have time, do more basin hopping from the best
    if best_positions is not None and _get_time_remaining() > 40:
        try:
            # Multiple restarts from best with varying step sizes
            for delta in [0.1, 0.2, 0.05]:
                if _get_time_remaining() < 20:
                    break
                energy, pos = _basin_hopping(best_positions, potential, rng,
                                             max_iterations=30, delta_e=delta)
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
        # Fallback to random
        best_positions = _generate_random(size, rng)
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
