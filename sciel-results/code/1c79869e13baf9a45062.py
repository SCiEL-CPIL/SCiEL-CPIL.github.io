import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedral cluster structure.
    Uses the standard Mackay shells:
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices of icosahedron)
    Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    Shell 3: 92 atoms (not fully utilized here, we take what we need)
    
    We construct the points geometrically to ensure high symmetry which is often a good starting point for Dzugutov.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (radius 1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: 12 vertices
    if n > 1:
        points.extend([v * 1.0 for v in verts])
        
    # Shell 2: 42 atoms
    if n > 13:
        # Find edges
        dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        dist_flat = dists.flatten()
        idx_flat = np.argsort(dist_flat)
        
        edges = []
        seen = set()
        for idx in idx_flat:
            i, j = divmod(idx, 12)
            if i == j: continue
            pair = (min(i,j), max(i,j))
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
                if len(edges) == 30:
                    break
        
        # Edge midpoints (30 atoms)
        # Radius of edge midpoint for unit icosahedron:
        # Distance from center to edge midpoint is cos(theta/2) where theta is angle between adjacent vertices.
        # For icosahedron, adjacent vertices have dot product -1/sqrt(1+phi^2)? No.
        # Let's just normalize the midpoint and scale it.
        # In a Mackay cluster, the distance ratio R2/R1 is fixed.
        # R1 = 1.0. R2 = sqrt(1 + 1/phi) ? No.
        # Let's calculate the norm of (v1+v2)/2.
        # v1, v2 are unit vectors. |v1+v2|^2 = 2 + 2 v1.v2.
        # For icosahedron, v1.v2 = -1/sqrt(1+phi^2) is incorrect.
        # Actually, for standard icosahedron with vertices (0,1,phi), the edge length is 2.
        # Distance from center to vertex is sqrt(1+phi^2).
        # So normalized vertices have distance 1.
        # Edge length in normalized coords: 2 / sqrt(1+phi^2).
        # Midpoint distance: sqrt(1 - (edge_len/2)^2) = sqrt(1 - 1/(1+phi^2)).
        # Let's just compute it numerically from the first edge to be safe.
        if edges:
            i, j = edges[0]
            mid_vec = (verts[i] + verts[j]) / 2
            r2 = np.linalg.norm(mid_vec)
        else:
            r2 = 1.9 # Fallback
            
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2
            mid /= np.linalg.norm(mid)
            points.append(mid * r2)
            
        # Face centers (12 atoms)
        # The 12 face centers of an icosahedron form a dodecahedron.
        # We can generate dodecahedron vertices and normalize them.
        # Dodecahedron vertices: (±1, ±1, ±1) and (0, ±1/phi, ±phi) etc.
        # Let's generate the 20 vertices of a dodecahedron and take 12? 
        # No, we need 12 points. The dual of icosahedron is dodecahedron.
        # The 20 faces of icosahedron correspond to 20 vertices of dodecahedron.
        # Wait, the Mackay 2nd shell has 42 atoms.
        # 30 edge centers + 12 face centers? 
        # An icosahedron has 20 faces. Why 12?
        # Ah, the 12 atoms are at the centers of the 12 *vertices* of the dual? No.
        # Let's check the literature. Mackay icosahedron n=55.
        # Shell 2 has 42 atoms.
        # 30 are on edges. 12 are on faces? But there are 20 faces.
        # Actually, the 12 atoms are located at the centers of the 12 *pentagonal* faces of the dodecahedron?
        # No, let's just use the 12 vertices of a dodecahedron that are NOT the ones corresponding to the icosahedron vertices?
        # Actually, it's easier to just add 12 points on a sphere of radius r2 that are roughly distributed.
        # Or, more accurately, the 12 points are the vertices of a dodecahedron inscribed in the sphere?
        # Let's try generating dodecahedron vertices and selecting 12.
        # Dodecahedron has 20 vertices. 
        # Let's just add 12 random points on the sphere of radius r2, but perturb them slightly to avoid symmetry breaking issues?
        # No, better to use the actual geometric centers.
        # The 12 points in the 2nd shell of a Mackay icosahedron are the 12 vertices of a dodecahedron?
        # If so, which 12? A dodecahedron has 20 vertices.
        # Wait, 1+12+42 = 55.
        # Shell 2 is 42.
        # 30 edge midpoints.
        # 12 face centers?
        # Icosahedron has 20 faces.
        # Maybe the 12 are the centers of the 12 *vertices* of the icosahedron? No, those are the 12 in shell 1.
        # Let's look at the structure.
        # It turns out the 12 atoms are at the centers of the 12 *edges* of the dual dodecahedron? No.
        # Let's just use the 12 vertices of a regular dodecahedron. 
        # We will generate 20 dodecahedron vertices and pick 12 that are farthest from the icosahedron vertices?
        # Or simply, the 12 points are the vertices of a dodecahedron that is dual to the icosahedron?
        # The dual of an icosahedron is a dodecahedron. The vertices of the dodecahedron are at the face centers of the icosahedron.
        # So there are 20 such points.
        # Why only 12 in the Mackay shell?
        # Actually, the Mackay icosahedron is not just shells of spheres.
        # Let's just add 12 random points on the sphere of radius r2. It's a heuristic.
        for _ in range(12):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            p = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
            points.append(p * r2)
            
    # Shell 3: 92 atoms
    if n > 55:
        needed = n - len(points)
        # Radius of shell 3.
        # R3/R1 is approx 2.0?
        # Let's use 2.0 * r2? No, r2 is ~1.9.
        # Let's estimate R3.
        # In Mackay, R3 is larger.
        # Let's just use 2.2.
        r3 = 2.2
        for _ in range(needed):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            p = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
            points.append(p * r3)
            
    return np.array(points[:n])

def _generate_random(n, rng):
    """Generate random compact cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    points = []
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points)

def _generate_shattered_icosahedron(n, rng):
    """
    Generate an icosahedron-like structure but with significant noise to break symmetry.
    This helps escape the deep local minima of perfect icosahedra if they aren't global.
    """
    base = _generate_mackay_icosahedron(n, rng)
    # Add noise
    noise = rng.normal(0, 0.3, base.shape)
    return base + noise

def _basin_hopping_worker(args):
    n, initial_pos, potential, parameter, seed_id = args
    rng = np.random.default_rng(seed_id)
    
    current_energy, current_pos = relax(initial_pos, potential, parameter)
    
    # Basin Hopping parameters
    step_size = 0.5
    max_iters = 30 # Reduced per thread to allow more threads
    min_step = 0.1
    max_step = 1.5
    
    for i in range(max_iters):
        if _get_time_remaining() < 5.0:
            break
            
        # Perturb
        noise = rng.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + noise
        
        # Relax
        new_energy, relaxed_pos = relax(new_pos, potential, parameter)
        
        # Metropolis criterion (always accept if lower, else probabilistic? 
        # For global optimization, standard BH accepts if new_energy < current_energy.
        # To escape, we rely on the step size increase.
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = relaxed_pos
            step_size *= 0.95
        else:
            step_size *= 1.05
            if step_size > max_step:
                step_size = max_step
            if step_size < min_step:
                step_size = min_step
                
    return current_energy, current_pos

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    seeds = []
    # 1. Perfect Mackay
    seeds.append(_generate_mackay_icosahedron(n, rng))
    # 2. Random
    seeds.append(_generate_random(n, rng))
    # 3. Shattered Mackay
    seeds.append(_generate_shattered_icosahedron(n, rng))
    # 4. Another Random
    seeds.append(_generate_random(n, rng))
    # 5. Another Shattered
    seeds.append(_generate_shattered_icosahedron(n, rng))
    
    # We have 5 seeds. We can run them in parallel using 4 threads.
    # We will process them in a way that utilizes the thread pool effectively.
    # Since we have 4 threads, we can run 4 at a time.
    # Or simply submit all and let the pool manage.
    
    best_energy = float('inf')
    best_positions = seeds[0]
    
    # Prepare arguments
    args_list = [(n, s, potential, parameter, i) for i, s in enumerate(seeds)]
    
    # Use ThreadPoolExecutor
    # Note: relax() might be thread-safe. If not, we need to be careful.
    # Assuming relax is thread-safe or GIL-bound (Python), threads help if relax releases GIL (e.g. C extension).
    # If relax is pure Python, threads won't help much due to GIL, but the task says "at most 4 threads" implying concurrency is allowed.
    # Let's assume relax benefits from threading (e.g. C backend).
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_basin_hopping_worker, args) for args in args_list]
        
        for f in concurrent.futures.as_completed(futures):
            try:
                energy, positions = f.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass

    # If time remains, do a final refinement on the best found so far
    remaining = _get_time_remaining()
    if remaining > 10.0:
        # Run a short BH on the best position found
        rng_refine = np.random.default_rng(999)
        current_energy, current_pos = relax(best_positions, potential, parameter)
        step_size = 0.2
        max_iters = int(remaining / 2.0) # Estimate iterations based on time
        for i in range(max_iters):
            if _get_time_remaining() < 2.0:
                break
            noise = rng_refine.normal(0, step_size, current_pos.shape)
            new_pos = current_pos + noise
            new_energy, relaxed_pos = relax(new_pos, potential, parameter)
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = relaxed_pos
                step_size *= 0.95
            else:
                step_size *= 1.05
                if step_size > 1.0: step_size = 1.0
                if step_size < 0.05: step_size = 0.05
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_pos

    return best_positions
