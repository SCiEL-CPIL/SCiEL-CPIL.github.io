# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 325.0  # 325s total budget (330s kill, 5s safety)


def _remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _rotmat(rng):
    """Generate a random rotation matrix."""
    # Use quaternion-based rotation for uniform distribution
    u1, u2, u3, u4 = rng.normal(0, 1, 4)
    norm = np.sqrt(u1**2 + u2**2 + u3**2 + u4**2)
    u1, u2, u3, u4 = u1/norm, u2/norm, u3/norm, u4/norm
    return np.array([
        [1 - 2*(u2**2 + u3**2), 2*(u1*u2 - u3*u4), 2*(u1*u3 + u2*u4)],
        [2*(u1*u2 + u3*u4), 1 - 2*(u1**2 + u3**2), 2*(u2*u3 - u1*u4)],
        [2*(u1*u3 - u2*u4), 2*(u2*u3 + u1*u4), 1 - 2*(u1**2 + u2**2)]
    ])


def _icosahedron_positions(n, rng):
    """Generate positions for an n-atom icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (12 vertices)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    if n <= 13:
        # Center + up to 12 vertices
        pts = [np.array([0.0, 0.0, 0.0])]
        for i in range(min(n-1, 12)):
            pts.append(verts[i])
        return np.array(pts)
    
    # For larger clusters, build shells
    # Shell 1: 12 vertices at radius r1
    # Shell 2: 30 edge midpoints at radius r2
    # Shell 3: 20 face centers at radius r3
    # etc.
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Generate all candidate directions
    directions = list(verts)
    
    # Edge midpoints (30 edges)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = verts[i] + verts[j]
            norm = np.linalg.norm(d)
            if norm > 0.1:
                directions.append(d / norm)
                edges.append((i, j))
    
    # Face centers (20 faces)
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d = verts[i] + verts[j] + verts[k]
                norm = np.linalg.norm(d)
                if norm > 0.1:
                    directions.append(d / norm)
    
    # Deduplicate
    unique_dirs = [directions[0]]
    for d in directions[1:]:
        if all(np.linalg.norm(d - ud) > 0.15 for ud in unique_dirs):
            unique_dirs.append(d)
    directions = unique_dirs
    
    # Shell radii (tuned for Dzugutov potential)
    shell_radii = [1.0, 1.75, 2.5, 3.25, 4.0, 4.75, 5.5]
    
    dir_idx = 0
    for shell_r in shell_radii:
        for d in directions:
            if len(points) >= n:
                break
            p = shell_r * d
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        if len(points) >= n:
            break
    
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-6, 6, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _assemble_subclusters(subcluster_positions, rng):
    """
    Divide-and-conquer assembly: place sub-clusters relative to each other
    and optimize their positions/orientations.
    """
    n_sub = len(subcluster_positions)
    if n_sub == 1:
        return subcluster_positions[0].copy()
    
    # Initialize: place sub-clusters on a sphere
    total_atoms = sum(len(sc) for sc in subcluster_positions)
    radius = 0.8 * (total_atoms / n_sub) ** (1/3)
    
    # Place sub-cluster centers
    centers = []
    for i in range(n_sub):
        angle = 2 * np.pi * i / n_sub
        centers.append(np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0]))
    
    # Build full cluster
    full_pos = []
    for i, sc in enumerate(subcluster_positions):
        R = _rotmat(rng)
        rotated = (sc - np.mean(sc, axis=0)) @ R.T + centers[i]
        full_pos.append(rotated)
    
    full_pos = np.vstack(full_pos)
    
    # Relax the whole thing
    try:
        _, full_pos = relax(full_pos, 'dzugutov')
    except:
        pass
    
    return full_pos


def _generate_seed(n, rng, method='icosahedral'):
    """Generate a seed structure."""
    if method == 'icosahedral':
        return _icosahedron_positions(n, rng)
    elif method == 'random':
        radius = 0.8 * n ** (1/3)
        points = []
        attempts = 0
        while len(points) < n and attempts < 20000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
            attempts += 1
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            for _ in range(50):
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    break
                p = p + rng.normal(0, 0.1, 3)
            points.append(p)
        return np.array(points[:n])
    elif method == 'divide_conquer':
        # Split into sub-clusters of ~13 atoms each
        sub_sizes = []
        remaining = n
        while remaining > 0:
            s = min(13, remaining)
            sub_sizes.append(s)
            remaining -= s
        
        subclusters = []
        for s in sub_sizes:
            sc = _icosahedron_positions(s, rng)
            subclusters.append(sc)
        
        return _assemble_subclusters(subclusters, rng)
    else:
        return _icosahedron_positions(n, rng)


def _basin_hopping(positions, potential, rng, time_limit=30.0):
    """Basin Hopping with adaptive step size and subset moves."""
    try:
        current_energy, current_pos = relax(positions, potential)
    except:
        return np.inf, positions
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    start_time = time.time()
    n_atoms = len(current_pos)
    
    # Adaptive step size
    base_step = 0.4 + 0.4 * (38.0 / n_atoms)
    step_size = base_step
    consecutive_rejects = 0
    
    while (time.time() - start_time) < time_limit:
        if _remaining() < 10:  # Safety margin
            break
        
        # Perturbation: move a random subset
        n_move = max(2, n_atoms // 5)
        move_idx = rng.choice(n_atoms, n_move, replace=False)
        
        delta = rng.normal(0, step_size, (n_move, 3))
        perturbed = current_pos.copy()
        perturbed[move_idx] += delta
        
        # Fix minimum distance violations
        diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        if np.min(dists) < 0.5:
            fixed = False
            for _ in range(30):
                if np.min(dists) >= 0.5:
                    fixed = True
                    break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                i_idx, j_idx = min_idx
                if i_idx == j_idx:
                    continue
                diff = perturbed[i_idx] - perturbed[j_idx]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = rng.normal(0, 0.1, 3)
                    dist = np.linalg.norm(diff)
                push = 0.5 * (diff / dist)
                perturbed[i_idx] += push
                perturbed[j_idx] -= push
                diffs = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                np.fill_diagonal(dists, np.inf)
            
            if not fixed:
                continue
        
        try:
            new_energy, new_pos = relax(perturbed, potential)
        except:
            continue
        
        # Metropolis acceptance
        T = 0.5
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            consecutive_rejects = 0
        else:
            delta_e = new_energy - current_energy
            if rng.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
        
        # Adapt step size
        if consecutive_rejects > 10:
            step_size *= 0.8
            consecutive_rejects = 0
        elif consecutive_rejects == 0 and step_size < base_step:
            step_size *= 1.05
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy structures using divide-and-conquer + basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(hash(instance["n"]) % (2**32))
    
    best_energy = np.inf
    best_positions = None
    
    # Time allocation: ~50s per instance, use 45s for search
    time_per_instance = 45.0
    
    # Strategy: multiple seeds with different methods
    seed_methods = [
        'icosahedral',
        'divide_conquer',
        'random',
        'icosahedral',
        'divide_conquer',
    ]
    
    # Time per seed (leave 5s for final relaxation)
    time_per_seed = (time_per_instance - 5.0) / len(seed_methods)
    
    for method in seed_methods:
        if _remaining() < 15:
            break
        
        try:
            pos = _generate_seed(size, rng, method)
            if pos.shape[0] == size:
                energy, pos_result = _basin_hopping(pos, potential, rng, time_limit=time_per_seed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos_result
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
        best_positions = _generate_seed(size, rng, 'icosahedral')
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
