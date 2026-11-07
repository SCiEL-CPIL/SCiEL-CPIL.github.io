"""Basin Hopping search for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Efficient distance check using broadcasting in chunks to avoid huge memory
    # We check all pairs, but in a vectorized way for speed if n is small, 
    # otherwise loop. For n<=80, a full distance matrix is 80x80=6400 floats, very cheap.
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < min_dist):
        return False
    return True

def _generate_bcc_cluster(n, rng):
    """Generate a BCC-like cluster."""
    points = []
    # Central atom
    points.append(np.zeros(3))
    # Neighbors in BCC lattice
    coords = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if i == j == k == 0:
                    continue
                # BCC: integer coords with all even or all odd
                if (i + j + k) % 2 == 0:
                    coords.append([i, j, k])
    
    # Sort by distance
    coords = sorted(coords, key=lambda c: np.dot(c, c))
    for c in coords:
        if len(points) >= n:
            break
        points.append(np.array(c, dtype=float) * 0.5)  
    while len(points) < n:
        p = rng.uniform(-3, 3, 3)
        points.append(p)
    return np.array(points[:n])

def _generate_fcc_cluster(n, rng):
    """Generate an FCC-like cluster."""
    points = []
    coords = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC logic: (i,j,k) where i,j,k are integers and i+j+k is even 
                # is actually BCC in this coordinate system if scaled by 0.5?
                # Let's use standard FCC basis vectors.
                pass
    
    # Simpler: generate points in a cubic grid and keep those that fit FCC symmetry
    # Or just use a dense random sphere with slight bias towards lattice points.
    # For simplicity and robustness, let's generate a "quasi-FCC" by taking 
    # points from a BCC lattice but scaling differently or just using random.
    # Actually, for TM clusters, BCC and ICOS are the main competitors.
    # Let's make this a "deformed BCC" or just another random cluster.
    
    # Let's try a different lattice: simple cubic or just random.
    # To be distinct, let's generate a random cluster with a tighter packing.
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 100 * n
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _generate_icosahedron(n, rng):
    """Generate points based on icosahedral shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    # Shell 1: 12 vertices
    if n >= 12:
        points.extend([v.copy() for v in verts])
        n -= 12
    
    # Shell 2: 30 vertices (midpoints of edges of icosahedron, scaled)
    # Approximate by adding random points on a sphere of radius ~1.5
    if n > 0:
        radius2 = 1.5
        num_to_add = min(n, 30)
        for _ in range(num_to_add):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = radius2 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            points.append(p)
        n -= num_to_add
        
    # Fill rest
    while n > 0:
        radius3 = 2.0
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        p = radius3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        points.append(p)
        n -= 1
        
    return np.array(points)

def _bh_worker(args):
    """
    Worker for Basin Hopping.
    args: (initial_positions, potential, kind, n_steps, seed)
    """
    initial_positions, potential, kind, n_steps, seed = args
    rng = np.random.default_rng(seed)
    current_pos = initial_positions.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    
    if not np.isfinite(current_energy):
        return (np.inf, None)
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Relax the initial position first to ensure we start at a local minimum
    try:
        current_energy, current_pos = relax(current_pos, potential, kind)
        if np.isfinite(current_energy) and _is_valid(current_pos):
            best_energy = current_energy
            best_pos = current_pos.copy()
        else:
            return (np.inf, None)
    except Exception:
        return (np.inf, None)

    for step in range(n_steps):
        # Perturb
        # The perturbation scale should be small enough to stay in the basin 
        # or jump to a neighbor, but large enough to explore.
        # Typical BH perturbation is ~0.01-0.1 in reduced units.
        scale = 0.1
        perturbed = current_pos + rng.normal(0, scale, current_pos.shape)
        
        # Relax perturbed
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(new_energy) or not _is_valid(new_pos):
            continue

        # Metropolis acceptance (T=0 for greedy, or small T)
        # For global minimum search, greedy (downhill only) is often sufficient 
        # if the basin is wide, but Metropolis helps escape. 
        # Given the tight time budget, greedy is faster and safer for "lowest found".
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            # Optional: Metropolis step
            # delta_E = new_energy - current_energy
            # if np.log(rng.uniform()) < -delta_E / T:
            #     current_energy = new_energy
            #     current_pos = new_pos
            pass

    return (best_energy, best_pos)

def construct(instance):
    """
    Basin Hopping search: 
    1. Generate diverse initial candidates.
    2. Relax them.
    3. Run parallel Basin Hopping chains from the best candidates.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Generate initial candidates
    rng = np.random.default_rng(size * 1000 + 42)
    
    candidates = []
    # 2 BCC, 2 FCC-like (random dense), 2 Icosahedral
    for _ in range(2):
        candidates.append(_generate_bcc_cluster(size, rng))
    for _ in range(2):
        candidates.append(_generate_fcc_cluster(size, rng))
    for _ in range(2):
        candidates.append(_generate_icosahedron(size, rng))
        
    # Relax initial candidates to get local minima
    local_minima = []
    for c in candidates:
        try:
            e, p = relax(c, potential, kind)
            if np.isfinite(e) and _is_valid(p):
                local_minima.append((e, p))
        except Exception:
            pass
            
    if not local_minima:
        # Fallback: just return a valid random configuration
        pts = rng.uniform(-2, 2, (size, 3))
        return pts

    # Sort by energy and pick the best few to start BH
    local_minima.sort(key=lambda x: x[0])
    # Use up to 4 best starting points for BH chains
    n_chains = min(4, len(local_minima))
    start_points = [p for e, p in local_minima[:n_chains]]
    
    # 2. Determine budget for BH
    # Each BH step involves a relax call. 
    # Relax takes ~0.1-0.5s depending on n.
    # Let's assume 0.2s per relax.
    # Total time allowed for BH: ~250s (leaving 40s for overhead/safety)
    # With 4 threads, we can run ~1250 relaxations in total.
    # So ~300 steps per chain.
    
    # Estimate steps per chain dynamically? No, fixed is safer.
    # Let's try 200 steps per chain. 4 chains * 200 steps * 4 threads = 800 relaxations.
    # If each is 0.3s, that's 240s. Fits in budget.
    
    n_steps_per_chain = 200
    seeds = [i * 1000 for i in range(n_chains)]
    
    args_list = [
        (sp, potential, kind, n_steps_per_chain, seed) 
        for sp, seed in zip(start_points, seeds)
    ]
    
    best_energy = min(e for e, p in local_minima)
    best_positions = min(local_minima, key=lambda x: x[0])[1]
    
    # 3. Run BH in parallel
    n_threads = min(4, cpu_count(), n_chains)
    
    try:
        with Pool(processes=n_threads) as pool:
            # We need to monitor time. Pool.map is blocking.
            # To allow early termination, we could use async, but Pool.map is simpler.
            # Given the fixed n_steps, it should finish within time.
            # If it's too slow, we might timeout. 
            # To be safe, let's reduce steps if n is large.
            if size > 60:
                # Reduce steps for larger clusters
                n_steps_per_chain = 100
                args_list = [
                    (sp, potential, kind, n_steps_per_chain, seed) 
                    for sp, seed in zip(start_points, seeds)
                ]
            
            results = pool.map(_bh_worker, args_list)
            
            for energy, positions in results:
                if positions is not None and np.isfinite(energy):
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                        
    except Exception as e:
        # If multiprocessing fails, fallback to sequential BH on the best candidate
        print(f"Multiprocessing failed: {e}")
        if best_positions is not None:
             # Just do a quick sequential BH on the best one found so far
             sp = best_positions
             try:
                 e_b, p_b = _bh_worker((sp, potential, kind, 50, 12345))
                 if p_b is not None and e_b < best_energy:
                     best_energy = e_b
                     best_positions = p_b
             except Exception:
                 pass

    # Final validation
    if best_positions is not None and _is_valid(best_positions):
        return best_positions
    else:
        # Return the best valid local minimum found in step 1
        return local_minima[0][1]

# EVOLVE-BLOCK-END
