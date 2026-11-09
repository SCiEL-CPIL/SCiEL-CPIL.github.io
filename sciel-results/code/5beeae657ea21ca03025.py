"""Dzugutov cluster global minima via Basin Hopping with geometric seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill

def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Edge length of this icosahedron: distance between [0,1,phi] and [1,phi,0]
    d = np.linalg.norm(verts[0] - verts[4])
    verts = verts * (scale / d)
    return verts

def _generate_icosahedral_shell(n, scale=1.1):
    """
    Generate an approximate n-atom cluster based on icosahedral symmetry.
    1. Center atom
    2. 12 vertices of icosahedron
    3. Second shell: points on edges and faces of the icosahedron
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    verts = _icosahedron_vertices(scale)
    points = [np.zeros(3)] # Center
    
    # Add first shell (12 vertices)
    for v in verts:
        if len(points) < n:
            points.append(v)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Add second shell candidates
    # The 42 atoms in a Mackay icosahedron (55 total) are located at:
    # - 30 edge midpoints (scaled)
    # - 12 face centers (scaled)
    # We approximate this by sampling points on the surface of the icosahedron.
    
    # Find edges (pairs of vertices with distance ~ edge length)
    edge_len = np.linalg.norm(verts[0] - verts[4])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(verts[i] - verts[j]) < 1.1 * edge_len:
                edges.append((i, j))
    
    # Generate points along edges
    edge_points = []
    for i, j in edges:
        # Place 2 points per edge (1/3 and 2/3) to get density
        for frac in [0.33, 0.66]:
            p = verts[i] * (1 - frac) + verts[j] * frac
            # Push out slightly to avoid overlap with core
            p = p * 1.15 
            edge_points.append(p)
            
    # Generate face centers
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(verts[i] - verts[j])
                djk = np.linalg.norm(verts[j] - verts[k])
                dki = np.linalg.norm(verts[k] - verts[i])
                if dij < 1.1*edge_len and djk < 1.1*edge_len and dki < 1.1*edge_len:
                    faces.append((i, j, k))
                    
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc * 1.2 # Push out
        face_centers.append(fc)
        
    second_shell = edge_points + face_centers
    np.random.shuffle(second_shell)
    
    for p in second_shell:
        if len(points) < n:
            points.append(p)
            
    # If still not enough, add random points near the surface
    if len(points) < n:
        for _ in range(n - len(points)):
            # Pick a random vertex and add noise
            v = verts[np.random.randint(12)]
            p = v * np.random.uniform(1.0, 1.5)
            points.append(p)
            
    return np.array(points[:n])

def _random_compact(n, rng, spacing=1.05):
    """Random compact cluster with min separation."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    pts = []
    max_tries = 1000 * n
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if pts:
                dists = np.linalg.norm(pts - p, axis=1)
                if np.min(dists) >= 0.7 * spacing:
                    pts.append(p)
            else:
                pts.append(p)
        tries += 1
    # Fill remaining with jittered points if packing failed
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3) + rng.normal(0, 0.1, 3)
        pts.append(p)
    return np.array(pts)

def _basin_hopping(n, initial_pos, rng, max_iters=50, step_size=0.25):
    """
    Perform Basin Hopping starting from initial_pos.
    Returns best position and energy found.
    """
    best_pos = initial_pos.copy()
    best_e, best_pos = relax(initial_pos, 'dzugutov')
    
    current_pos = best_pos.copy()
    current_e = best_e
    
    for i in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Perturb
        new_pos = current_pos + rng.normal(0, step_size, (n, 3))
        
        # Relax
        e_new, pos_new = relax(new_pos, 'dzugutov')
        
        # Metropolis acceptance (T is effectively high to explore)
        # For global minima search, we often just accept if better, or use a temperature
        # Here we use a simple greedy + occasional uphill move
        if e_new < current_e:
            current_pos = pos_new
            current_e = e_new
            if e_new < best_e:
                best_e = e_new
                best_pos = pos_new
        else:
            # Accept with probability exp(-dE/T)
            # T is scaled by problem size
            T = 0.5 * np.sqrt(n)
            delta_e = e_new - current_e
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_pos = pos_new
                current_e = e_new
            else:
                # Reject, keep current
                pass
                
    return best_pos, best_e

def construct(instance):
    """
    Find low-energy Dzugutov cluster using parallel Basin Hopping
    from diverse geometric seeds.
    """
    size = instance["n"]
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse initial structures
    seeds = []
    
    # 1. Icosahedral based seeds (multiple scales)
    for scale in [1.0, 1.1, 1.2]:
        try:
            seeds.append(_generate_icosahedral_shell(size, scale))
        except Exception:
            pass
            
    # 2. Random compact seeds
    for _ in range(4):
        seeds.append(_random_compact(size, rng))
        
    # 3. Distorted Icosahedral (perturb the icosahedral seed)
    if seeds:
        base = seeds[0].copy()
        base += rng.normal(0, 0.2, base.shape)
        seeds.append(base)

    best_pos_global = seeds[0]
    best_e_global = np.inf
    
    # Time management
    # We have ~280s. Let's allocate time to each seed run.
    # If we have ~10 seeds, ~25s each.
    # Basin Hopping iterations are expensive due to relax.
    # Let's run them sequentially but with strict time checks.
    
    num_seeds = len(seeds)
    time_per_seed = 20.0 # seconds
    
    for i, seed in enumerate(seeds):
        if time.time() > _DEADLINE - 5:
            break
            
        # Run Basin Hopping
        # Adjust max_iters based on time
        start_t = time.time()
        # Estimate iterations: relax takes ~0.1-0.5s? 
        # Let's just loop until time budget for this seed is up.
        max_iters = 100 # cap for logic, but time controls it
        
        pos, e = _basin_hopping(size, seed, rng, max_iters=max_iters)
        
        if e < best_e_global:
            best_e_global = e
            best_pos_global = pos
            
    return best_pos_global
# EVOLVE-BLOCK-END
