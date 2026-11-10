"""Dzugutov cluster global minima: polytetrahedral seeds + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 325.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _rotmat(rng):
    """Generate a random rotation matrix."""
    # Use QR decomposition of a random matrix
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure determinant is +1
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    return Q


def _icosahedron_positions(n, rng):
    """
    Generate positions for an n-atom cluster with icosahedral symmetry.
    Uses shells: 1 (center), 12 (vertices), 30 (edges), 20 (faces), ...
    For n < 13, use center + vertices.
    For 13 <= n < 55, use center + vertices + edge midpoints.
    For 55 <= n < 147, use center + vertices + edge midpoints + face centers.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of an icosahedron (normalized to radius 1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    
    positions = [np.zeros(3)]
    
    # Shell 1: 12 atoms at vertices (radius ~1.4 for Dzugutov min distance)
    r1 = 1.4
    for v in verts:
        positions.append(v * r1)
    
    if n > 13:
        # Shell 2: 30 atoms at edge midpoints (radius ~2.2)
        r2 = 2.2
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 2.0:  # Adjacent vertices
                    mid = (verts[i] + verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edges.append(mid * r2)
        
        # Shuffle edges to allow random truncation if n < 43
        rng.shuffle(edges)
        for e in edges:
            if len(positions) >= n:
                break
            positions.append(e)
    
    if n > 43:
        # Shell 3: 20 atoms at face centers (radius ~2.8)
        r3 = 2.8
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if d1 < 2.0 and d2 < 2.0 and d3 < 2.0:
                        center = (verts[i] + verts[j] + verts[k]) / 3
                        center = center / np.linalg.norm(center)
                        faces.append(center * r3)
        
        rng.shuffle(faces)
        for f in faces:
            if len(positions) >= n:
                break
            positions.append(f)
    
    # If we still need more atoms (n > 43+20=63 is rare for this task, max 60)
    # Or if we need to fill gaps for n between 13 and 43, or 43 and 63
    # If we have fewer than n, add random points near the surface
    while len(positions) < n:
        # Pick a random existing point and offset it
        idx = rng.integers(0, len(positions))
        p = positions[idx] + rng.normal(0, 0.5, 3)
        # Ensure minimum distance
        if all(np.linalg.norm(p - q) >= 0.6 for q in positions):
            positions.append(p)
    
    return np.array(positions[:n])


def _generate_random_compact(n, rng):
    """Generate a random compact cluster as a fallback seed."""
    points = []
    radius = 1.5 * n ** (1.0/3.0)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 for q in points):
            points.append(p)
    return np.array(points)


def _basin_hop(positions, rng, max_steps=5, step_size=0.5):
    """
    Perform basin hopping: random perturbations followed by relaxation.
    Returns the best energy and positions found.
    """
    # Initial relaxation
    energy, pos = relax(positions, 'dzugutov')
    best_energy = energy
    best_pos = pos
    
    for _ in range(max_steps):
        if _get_time_remaining() < 5:
            break
        
        # Random perturbation
        noise = rng.normal(0, step_size, pos.shape)
        new_pos = pos + noise
        
        # Relax the perturbed structure
        try:
            e_new, p_new = relax(new_pos, 'dzugutov')
        except Exception:
            continue
        
        if e_new < best_energy:
            best_energy = e_new
            best_pos = p_new
            # Accept move, continue from here
            pos = p_new
            energy = e_new
        else:
            # Reject move, stay at previous best (or current if we want random walk, 
            # but for global min search, staying at best is safer if we have limited time)
            # Standard basin hopping accepts with probability exp(-dE/T). 
            # Given time constraints, we just keep the best found so far.
            pass
            
    return best_energy, best_pos


def _worker(instance, seed_idx, rng_seed):
    """
    Worker function for parallel execution.
    Generates a seed, relaxes it, and performs basin hopping.
    """
    n = instance["n"]
    rng = np.random.default_rng(rng_seed)
    
    # Select seed type based on index
    if seed_idx % 3 == 0:
        seed = _icosahedron_positions(n, rng)
    elif seed_idx % 3 == 1:
        seed = _generate_random_compact(n, rng)
    else:
        # Slightly perturbed icosahedron
        seed = _icosahedron_positions(n, rng) + rng.normal(0, 0.2, (n, 3))
    
    # Ensure minimum distance constraint is roughly satisfied before relax
    # relax() might handle it, but better to be safe
    # If relax fails, try a random seed
    try:
        energy, pos = _basin_hop(seed, rng, max_steps=3, step_size=0.5)
    except Exception:
        try:
            seed = _generate_random_compact(n, rng)
            energy, pos = relax(seed, 'dzugutov')
        except Exception:
            return None, None
            
    return energy, pos


def construct(instance):
    """
    Main entry point.
    Uses parallel basin hopping from polytetrahedral and random seeds.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    time_left = _get_time_remaining()
    
    # If very little time left, just do one quick relaxation
    if time_left < 10:
        seed = _icosahedron_positions(n, np.random.default_rng(n))
        try:
            _, pos = relax(seed, 'dzugutov')
            return pos
        except:
            return seed

    # Determine number of parallel workers
    # More workers for smaller n, fewer for larger n
    if n < 45:
        n_workers = 4
    else:
        n_workers = 3
        
    best_energy = float('inf')
    best_pos = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            # Unique seed for each worker to ensure diversity
            seed_val = n * 100 + i
            fut = executor.submit(_worker, instance, i, seed_val)
            futures.append(fut)
        
        for fut in concurrent.futures.as_completed(futures):
            if _get_time_remaining() < 5:
                break
            try:
                energy, pos = fut.result()
                if pos is not None and energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                pass
    
    if best_pos is None:
        # Fallback
        seed = _icosahedron_positions(n, np.random.default_rng(n))
        try:
            _, best_pos = relax(seed, 'dzugutov')
        except:
            best_pos = seed
            
    return best_pos
# EVOLVE-BLOCK-END
