"""
Strategy: Structure-Guided Basin Hopping with Rigid-Body Rotations.
1. Initialization: 
   - Create a stable core (FCC/ICosa fragment of size ~n/2).
   - Embed remaining atoms into interstitial sites of the core.
   - Also generate standard FCC/HCP/BCC/Random starts for diversity.
2. Optimization:
   - Parallel Multi-Start.
   - Basin Hopping where perturbations include:
     a) Rigid-body rotation of the whole cluster (preserves internal geometry).
     b) Small random Gaussian displacement (allows structural changes).
   - This hybrid perturbation helps navigate the energy landscape by 
     exploring orientation space (rigid) and shape space (deformation) 
     separately, which is often more efficient than pure random noise.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0
_START_TIME = None
_MAX_THREADS = 4

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return max(0.0, _DEADLINE_SEC - (time.time() - _START_TIME))

def _generate_fcc(n, spacing=0.72):
    """Generate n atoms from an FCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_hcp(n, spacing=0.72):
    """Generate n atoms from an HCP lattice fragment."""
    a = spacing
    c = 1.633 * a
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3) / 2 * j)
            for k in range(-max_r, max_r + 1):
                points.append([x, y, k * c])
                points.append([x + 2*a/3, y + a*np.sqrt(3)/3, (k + 0.5) * c])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_bcc(n, spacing=0.72):
    """Generate n atoms from a BCC lattice fragment."""
    a = spacing
    max_coord = int(np.ceil(n ** (1/3))) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_icosahedral(n, spacing=0.72):
    """
    Generate an icosahedral cluster.
    13-atom icosahedron: 1 center, 12 vertices.
    For larger n, add shells or just take a fragment.
    """
    a = spacing
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    # Scale to desired spacing (distance from center to vertex is ~a)
    verts *= a
    
    points = [np.zeros(3)] # Center
    points.extend(verts.tolist())
    
    if n <= 13:
        # Just take the first n points (center + some vertices)
        return np.array(points[:n])
    
    # For n > 13, we need to add more atoms.
    # Simple strategy: Add a second shell or just use FCC/HCP if n is much larger.
    # For this heuristic, we'll just return the 13-atom core and let the 
    # "embedding" logic in construct() handle the rest, or fallback to FCC.
    # To keep it simple and robust, if n > 13, we just return the 13-atom 
    # icosahedron and the caller will pad it or we use it as a core.
    # However, _generate_* functions are expected to return n atoms.
    # Let's pad with random points close to the surface if n > 13.
    rng = np.random.default_rng(42)
    while len(points) < n:
        # Place new atom on the surface of the icosahedron (random direction)
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        p = dir * a * 1.1 # Slightly outside
        points.append(p)
        
    return np.array(points[:n])

def _generate_core_embedded(n, seed=0):
    """
    Strategy: 
    1. Create a core of size n_core = max(13, n//2) using FCC or Icosahedral.
    2. Embed remaining atoms into interstitial sites (random positions near the core).
    """
    rng = np.random.default_rng(seed)
    
    n_core = max(13, n // 2)
    if n_core > n:
        n_core = n
        
    # Generate core
    # Alternate between FCC and Icosa based on seed
    if seed % 2 == 0:
        core = _generate_fcc(n_core, spacing=0.72)
    else:
        core = _generate_icosahedral(n_core, spacing=0.72)
        
    if core is None:
        core = _generate_fcc(n_core, spacing=0.72)
        
    if core is None:
        return _generate_random(n, seed=seed)

    points = list(core)
    
    # Embed remaining atoms
    n_remaining = n - len(points)
    if n_remaining > 0:
        # Calculate radius of core
        core_arr = np.array(points)
        radius = np.max(np.linalg.norm(core_arr, axis=1))
        
        for _ in range(n_remaining):
            # Random direction
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            # Place at radius + small jitter
            r = radius * (1.0 + rng.uniform(0.1, 0.5))
            p = dir * r
            points.append(p)
            
    return np.array(points[:n])

def _generate_random(n, spacing=0.72, seed=0):
    """Generate a random compact cluster with rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    max_attempts = n * 200
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _rotate(points, angle, axis):
    """Rotate points by angle around a unit vector axis."""
    # Ensure axis is unit length
    axis = axis / np.linalg.norm(axis)
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return points @ R.T

def _worker(args):
    """
    Worker function for parallel execution.
    args: (instance, start_pos, thread_id)
    Returns: (energy, positions)
    """
    instance, start_pos, thread_id = args
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    if _time_left() < 5:
        return (float('inf'), None)
    
    best_energy = float('inf')
    best_pos = None
    
    # 1. Initial Relaxation
    try:
        e, pos = relax(start_pos, potential, kind)
        if np.isfinite(e):
            best_energy = e
            best_pos = pos
    except Exception:
        pass
    
    # 2. Structure-Guided Basin Hopping
    if best_pos is not None and _time_left() > 10:
        rng = np.random.default_rng(thread_id * 1000 + 42)
        step_size_trans = 0.15 # Translation/Deformation step
        step_size_rot = 0.2    # Rotation step (radians)
        current_energy = best_energy
        current_pos = best_pos.copy()
        
        max_hops = int((_time_left() - 5) / 0.1)
        max_hops = min(max_hops, 60)
        
        for _ in range(max_hops):
            if _time_left() < 5:
                break
            
            # Perturbation: Rigid Rotation + Small Random Displacement
            # 1. Rotate the whole cluster
            axis = rng.normal(size=3)
            angle = rng.uniform(-step_size_rot, step_size_rot)
            rotated = _rotate(current_pos, angle, axis)
            
            # 2. Add small random displacement to each atom
            noise = rng.normal(0, step_size_trans, size=(n, 3))
            perturbed = rotated + noise
            
            try:
                e_new, pos_new = relax(perturbed, potential, kind)
                if np.isfinite(e_new):
                    if e_new < current_energy:
                        current_energy = e_new
                        current_pos = pos_new
                        if e_new < best_energy:
                            best_energy = e_new
                            best_pos = pos_new.copy()
                        # Anneal steps down
                        step_size_trans *= 0.95
                        step_size_rot *= 0.95
                    else:
                        # Accept if Boltzmann probability (simplified: accept some bad moves)
                        # Or just reject and increase steps
                        # Let's use a simple Metropolis criterion for better exploration
                        T = 0.5 * (1 - _ / max_hops) # Simple cooling
                        if rng.random() < np.exp((current_energy - e_new) / (T + 1e-6)):
                            current_energy = e_new
                            current_pos = pos_new
                        else:
                            # Increase steps to escape
                            step_size_trans *= 1.05
                            step_size_rot *= 1.05
                    
                    step_size_trans = np.clip(step_size_trans, 0.05, 1.0)
                    step_size_rot = np.clip(step_size_rot, 0.05, 1.0)
            except Exception:
                continue
                
    return (best_energy, best_pos)

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse starting points
    starts = []
    
    # 1. Standard Templates
    fcc = _generate_fcc(n)
    if fcc is not None:
        starts.append(fcc)
        
    hcp = _generate_hcp(n)
    if hcp is not None:
        starts.append(hcp)
        
    bcc = _generate_bcc(n)
    if bcc is not None:
        starts.append(bcc)
        
    # 2. Structure-Guided Starts (Core Embedded)
    # These are the key new starts
    for seed in range(4):
        core_emb = _generate_core_embedded(n, seed=seed)
        starts.append(core_emb)
        
    # 3. Random Starts
    for seed in range(4):
        rand = _generate_random(n, seed=seed)
        starts.append(rand)
        
    # Shuffle starts
    rng_master = np.random.default_rng(123)
    rng_master.shuffle(starts)
    
    # Select 4 starts for 4 threads
    # Prioritize Core Embedded and FCC as they are likely closer to global minima
    selected_starts = []
    
    # Ensure we have at least one Core Embedded and one FCC if possible
    # Re-shuffle logic: just pick the first 4 from the shuffled list, 
    # but we can bias by putting Core Embedded at the front.
    
    # Let's explicitly pick:
    # Thread 0: FCC
    # Thread 1: Core Embedded (seed 0)
    # Thread 2: Core Embedded (seed 1)
    # Thread 3: Random (seed 0)
    
    # This ensures we test the new hypothesis in 2 threads, 
    # while keeping 1 known good (FCC) and 1 diverse (Random).
    
    if fcc is not None:
        selected_starts.append(fcc)
    else:
        selected_starts.append(starts[0])
        
    selected_starts.append(_generate_core_embedded(n, seed=0))
    selected_starts.append(_generate_core_embedded(n, seed=1))
    selected_starts.append(_generate_random(n, seed=0))
    
    # If we need more (we only have 4 slots, so this is fine)
    while len(selected_starts) < _MAX_THREADS:
        selected_starts.append(_generate_random(n, seed=len(selected_starts)+100))
        
    tasks = []
    for i in range(_MAX_THREADS):
        tasks.append((instance, selected_starts[i], i))
        
    best_energy_global = float('inf')
    best_pos_global = None
    
    with ThreadPoolExecutor(max_workers=_MAX_THREADS) as executor:
        futures = {executor.submit(_worker, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            if _time_left() < 2:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy_global:
                    best_energy_global = energy
                    best_pos_global = positions
            except Exception:
                pass
                
    if best_pos_global is None:
        fallback_start = _generate_random(n, seed=999)
        try:
            e, pos = relax(fallback_start, 'sutton_chen', kind)
            best_pos_global = pos
        except Exception:
            best_pos_global = fallback_start
            
    return best_pos_global
# EVOLVE-BLOCK-END
