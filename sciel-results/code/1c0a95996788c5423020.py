"""
Improved Dzugutov cluster search using parallel basin hopping with Mackay icosahedral seeds.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit
N_THREADS = 4

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _get_icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _get_icosahedron_edges(verts):
    # Identify edges by distance (should be 2.0 in unnormalized coords, or specific normalized distance)
    # Normalized distance between connected vertices is 2 / sqrt(1+phi^2) = 2 / sqrt((5+sqrt(5))/2)
    # Let's just compute distances and pick the smallest 30
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    edges = []
    # The 30 edges are the shortest distances
    # There are 66 pairs. 30 are edges, 36 are diagonals.
    # We can sort them.
    pairs = []
    for i in range(12):
        for j in range(i+1, 12):
            pairs.append((dists[i, j], i, j))
    pairs.sort()
    # The first 30 should be the edges
    for d, i, j in pairs[:30]:
        edges.append((i, j))
    return edges

def _get_icosahedron_faces(verts, edges):
    # Faces are triangles. We can find them by looking for cycles of length 3 in the edge graph
    # Or simpler: we know the 20 faces of an icosahedron.
    # Let's just generate them programmatically or hardcode indices for the standard icosahedron.
    # Standard faces for the vertex order above:
    faces = [
        [0, 1, 8], [0, 8, 4], [0, 4, 2], [0, 2, 6], [0, 6, 1],
        [1, 6, 9], [1, 9, 5], [1, 5, 3], [1, 3, 8],
        [2, 4, 10], [2, 10, 7], [2, 7, 6],
        [3, 5, 11], [3, 11, 10], [3, 10, 4], [3, 4, 8],
        [5, 9, 11], [5, 11, 7], [5, 7, 3],
        [7, 11, 10], [7, 10, 2], [7, 2, 6], [7, 6, 9], [7, 9, 5]
    ]
    # Verify these are valid (all pairs in face are edges)
    edge_set = set()
    for i, j in edges:
        edge_set.add((i, j))
        edge_set.add((j, i))
    
    valid_faces = []
    for f in faces:
        if all((f[k], f[(k+1)%3]) in edge_set for k in range(3)):
            valid_faces.append(f)
            
    return valid_faces

def _generate_mackay_seed(n, scale=1.0, seed=0):
    """
    Generate a Mackay icosahedral seed for n atoms.
    Shells:
    0: 1 atom (center)
    1: 12 atoms (icosahedron vertices)
    2: 30 atoms (edge midpoints)
    3: 12 atoms (face centers)
    Total 55.
    For n > 55, add random surface atoms.
    For n < 55, take first n atoms from the list.
    """
    verts = _get_icosahedron_vertices()
    edges = _get_icosahedron_edges(verts)
    faces = _get_icosahedron_faces(verts, edges)
    
    points = []
    
    # Shell 0
    if n > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: 12 vertices
    if n > 12:
        r1 = 0.55 * scale
        points.extend((verts * r1).tolist())
    
    # Shell 2: 30 edge midpoints
    if n > 42:
        r2 = 0.85 * scale
        edge_mids = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            edge_mids.append(mid * r2)
        points.extend(edge_mids)
        
    # Shell 3: 12 face centers
    if n > 54:
        r3 = 0.75 * scale
        face_centers = []
        for f in faces:
            fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc * r3)
        points.extend(face_centers)
        
    points = np.array(points)
    
    # If we have more than n, trim
    if len(points) >= n:
        return points[:n].copy()
        
    # If we have fewer (n > 55), add random atoms on the surface
    rng = np.random.default_rng(seed)
    while len(points) < n:
        # Pick a random direction from the existing vertices or faces
        idx = rng.integers(0, 12)
        dir_vec = verts[idx]
        # Place slightly outside
        r = 1.1 * scale * rng.uniform(0.9, 1.1)
        p = dir_vec * r
        # Add small random perturbation to avoid overlap
        p += rng.normal(0, 0.1, 3)
        points = np.vstack([points, p])
        
    return points[:n].copy()

def _generate_random_seed(n, seed=0):
    """Generate a random compact seed."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1.0/3.0) * 0.8
    points = []
    min_dist = 0.8
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    # Fallback if random placement failed (shouldn't happen with min_dist=0.8 and large radius)
    while len(points) < n:
        p = rng.uniform(-radius*0.5, radius*0.5, 3)
        points.append(p)
        
    return np.array(points[:n])

def _kick_and_relax(positions, potential, parameter, kick_sigma, seed, time_limit_check):
    """
    Perform a single kick and relax step.
    Returns (energy, positions).
    """
    if time_limit_check():
        return None, None
        
    rng = np.random.default_rng(seed)
    perturbed = positions + rng.normal(0, kick_sigma, positions.shape)
    
    try:
        energy, relaxed = relax(perturbed, potential, parameter)
        return energy, relaxed
    except Exception as e:
        return np.inf, positions

def _basin_hopping_trajectory(seed_pos, potential, parameter, n_kicks, seed, time_limit_check):
    """
    Run a basin hopping trajectory starting from seed_pos.
    """
    best_energy, best_pos = np.inf, seed_pos.copy()
    
    # Initial relax
    if not time_limit_check():
        try:
            e0, p0 = relax(seed_pos, potential, parameter)
            best_energy = e0
            best_pos = p0
        except:
            return np.inf, seed_pos
            
    current_pos = best_pos.copy()
    
    for i in range(n_kicks):
        if time_limit_check():
            break
            
        # Adaptive kick size: start large, get smaller
        # sigma = 0.2 * (0.9 ** i)
        sigma = 0.15 + 0.1 * np.cos(i * np.pi / n_kicks)
        
        energy, new_pos = _kick_and_relax(current_pos, potential, parameter, sigma, seed + i, time_limit_check)
        
        if energy is not None:
            if energy < best_energy:
                best_energy = energy
                best_pos = new_pos
            # Metropolis acceptance? No, just keep best (greedy basin hopping)
            # But to escape, we should sometimes accept worse? 
            # For global min, greedy is often fine if kicks are large enough.
            # Let's stick to greedy for simplicity and speed, but ensure kicks are varied.
            current_pos = new_pos
            
    return best_energy, best_pos

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Mackay Icosahedron (Primary)
    base_scale = size ** (1.0/3.0) * 0.45
    seeds.append(("mackay_base", _generate_mackay_seed(size, scale=base_scale, seed=1)))
    
    # 2. Deformed Mackay
    for s in range(3):
        mackay = _generate_mackay_seed(size, scale=base_scale, seed=1)
        rng = np.random.default_rng(100 + s)
        deformed = mackay + rng.normal(0, 0.2, mackay.shape)
        seeds.append((f"mackay_def_{s}", deformed))
        
    # 3. Different Scales
    seeds.append(("mackay_small", _generate_mackay_seed(size, scale=base_scale*0.9, seed=2)))
    seeds.append(("mackay_large", _generate_mackay_seed(size, scale=base_scale*1.1, seed=3)))
    
    # 4. Random Compact
    for s in range(4):
        seeds.append((f"random_{s}", _generate_random_seed(size, seed=200 + s)))
        
    # 5. Random Sphere Surface (for quasicrystal-like?)
    for s in range(2):
        rng = np.random.default_rng(300 + s)
        radius = base_scale * 1.2
        pts = []
        for _ in range(size):
            theta = rng.uniform(0, 2*np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            p = np.array([r*np.cos(theta), r*np.sin(theta), z]) * radius
            pts.append(p)
        seeds.append((f"surface_{s}", np.array(pts)))

    best_energy = np.inf
    best_positions = None
    
    def time_check():
        return _time_remaining() < 10.0

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {}
        
        # Submit all seeds
        for name, seed_pos in seeds:
            if time_check():
                break
            # Allocate a certain number of kicks per seed
            # Total kicks ~ 50-100. With 10 seeds, ~5-10 kicks each.
            # Let's do 10 kicks per seed.
            n_kicks = 10
            future = executor.submit(_basin_hopping_trajectory, seed_pos, potential, parameter, n_kicks, hash(name) % 10000, time_check)
            futures[future] = name
            
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            if time_check():
                # Cancel remaining
                for f in futures:
                    f.cancel()
                break
                
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception as e:
                pass
                
    # If we still have time, do more kicks on the best position found
    if best_positions is not None and _time_remaining() > 20.0:
        # Continue basin hopping on the best found
        n_extra_kicks = 20
        for i in range(n_extra_kicks):
            if _time_remaining() < 10.0:
                break
            sigma = 0.1
            energy, new_pos = _kick_and_relax(best_positions, potential, parameter, sigma, 1000 + i, time_check)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = new_pos.copy()
                
    return best_positions
# EVOLVE-BLOCK-END
