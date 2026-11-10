import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 315.0  # Safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_icosahedral(n, rng):
    """
    Generate an icosahedral-like cluster.
    The Dzugutov potential favors polytetrahedral order.
    We build concentric icosahedral shells.
    """
    # Icosahedral vertices
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
        [1, t, 0], [-1, -t, 0], [1, -t, 0], [-1, t, 0] # Duplicates for simplicity in generation, will dedupe
    ])
    # Actually, standard icosahedron has 12 vertices.
    # Let's use a cleaner set.
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize to unit radius
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    positions = []
    if n >= 1:
        positions.append([0, 0, 0]) # Center atom
    
    # Shell 1: 12 atoms at radius r1
    # Ideal nearest neighbor distance for Dzugutov is around 1.8-1.9
    # Let's target a nearest neighbor distance of ~1.8
    # For icosahedron, distance from center to vertex is R.
    # Distance between adjacent vertices is ~ 2*R*sin(pi/5) ~ 1.17*R
    # If we want d ~ 1.8, R ~ 1.8/1.17 ~ 1.54
    r1 = 1.5
    if n > 1:
        # Add 12 atoms
        shell1 = v * r1
        for i in range(12):
            if len(positions) < n:
                positions.append(shell1[i])
    
    # Shell 2: 30 atoms at radius r2
    # Vertices of a pentagonal icosahedron? Or just scale up?
    # A second shell of 30 atoms (vertices of a pentakis icositetrahedron or similar)
    # For simplicity, let's use a scaled version of the icosahedron vertices + face centers?
    # Or just a larger icosahedron? No, that's not dense.
    # Let's use a simple heuristic: if n > 13, add a shell of 30 atoms.
    # The 30 vertices are the midpoints of the edges of the icosahedron? No, that's 30 edges.
    # Let's generate points on a sphere with higher density.
    # For n up to 60, we likely need 1 + 12 + 30 + ...
    # Let's just add random points on a sphere of radius r2 if we need more, 
    # but biased towards icosahedral directions.
    
    # Better: Use a known 30-atom shell structure if possible, or just random spherical shell.
    # Given time constraints, random spherical shell with relaxation is robust.
    if n > 13:
        r2 = 2.8
        # Generate 30 points on sphere
        # Use golden spiral or random
        while len(positions) < n:
            # Random point on sphere
            z = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2 * np.pi)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z]) * r2
            positions.append(p)
            
    pos = np.array(positions[:n])
    
    # Add small random noise to break symmetry
    pos += rng.normal(0, 0.1, pos.shape)
    
    return pos

def _generate_fcc(n, rng):
    """Generate FCC-like cluster."""
    # Approximate radius
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    a = 2.2 # Lattice constant
    points = []
    
    # Generate lattice points in a cube and keep those within sphere
    # FCC basis: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    # Scale a so that volume matches n
    
    # Simple approach: iterate over lattice indices
    max_k = int(np.ceil(R / a)) + 2
    
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            for k in range(-max_k, max_k + 1):
                # FCC positions
                for basis in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a + basis[0], j*a + basis[1], k*a + basis[2]])
                    if np.linalg.norm(p) <= R:
                        points.append(p)
                
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
            
    pos = np.array(points[:n])
    if len(pos) < n:
        # Fill remaining with random points
        while len(pos) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                pos = np.vstack([pos, p])
                
    # Add small random noise
    pos += rng.normal(0, 0.1, pos.shape)
    
    return pos

def _generate_random(n, rng):
    """Generate random cluster."""
    R = 2.5 * (n / 4.0) ** (1.0/3.0)
    positions = np.zeros((n, 3))
    count = 0
    while count < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            if count == 0 or np.all(np.linalg.norm(p - positions[:count], axis=1) > 0.5):
                positions[count] = p
                count += 1
    return positions[:n]

def _basin_hopping(n, seed, t_start, t_end, steps, dt, generator_type):
    """
    Perform Basin Hopping.
    """
    rng = np.random.default_rng(seed)
    
    # Select generator
    if generator_type == 'ico':
        current_pos = _generate_icosahedral(n, rng)
    elif generator_type == 'fcc':
        current_pos = _generate_fcc(n, rng)
    else:
        current_pos = _generate_random(n, rng)
        
    # Initial relaxation
    try:
        current_energy, current_pos = relax(current_pos, 'dzugutov', None)
    except Exception:
        return None, float('inf')
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    step_size = 0.5
    
    for i in range(steps):
        if _get_time_remaining() < 5.0:
            break
            
        # Perturb
        # Step size annealing?
        # Keep step size constant or slowly decrease
        new_pos = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        try:
            new_energy, new_pos_relaxed = relax(new_pos, 'dzugutov', None)
        except Exception:
            continue
            
        if not np.isfinite(new_energy):
            continue
            
        # Acceptance criterion: 
        # For global minimization, we can just accept if better (greedy) 
        # or use Metropolis. Given the time limit, greedy is often safer 
        # to ensure we don't wander. However, to escape local minima, 
        # we need to sometimes accept worse moves.
        # Let's use a simple temperature annealing for acceptance.
        # T decreases from t_start to t_end
        t = t_start * (t_end / t_start) ** (i / steps)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / t):
            current_energy = new_energy
            current_pos = new_pos_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos_relaxed.copy()
                # Shrink step size slightly when finding new best
                step_size *= 0.95
        else:
            # Increase step size slightly if stuck?
            step_size *= 1.02
            if step_size > 2.0:
                step_size = 2.0

    return best_pos, best_energy

def _worker(args):
    n, seed, t_start, t_end, steps, dt, gen_type = args
    try:
        pos, energy = _basin_hopping(n, seed, t_start, t_end, steps, dt, gen_type)
        return pos, energy
    except Exception:
        # Fallback
        rng = np.random.default_rng(seed)
        pos = _generate_random(n, rng)
        try:
            e, p = relax(pos, 'dzugutov', None)
            return p, e
        except:
            return pos, float('inf')

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    
    # Define multiple strategies in parallel
    # We have 4 threads.
    # Strategy 1: Icosahedral seed, moderate BH
    # Strategy 2: FCC seed, moderate BH
    # Strategy 3: Random seed, moderate BH
    # Strategy 4: Icosahedral seed, different seed/temp
    
    runs = [
        (n, 1, 5.0, 0.1, 1000, 0.01, 'ico'),
        (n, 2, 5.0, 0.1, 1000, 0.01, 'fcc'),
        (n, 3, 5.0, 0.1, 1000, 0.01, 'random'),
        (n, 4, 10.0, 0.01, 1500, 0.01, 'ico'), # Higher start temp, slower cool
    ]
    
    best_pos = None
    best_energy = float('inf')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, r) for r in runs]
        for f in concurrent.futures.as_completed(futures):
            pos, energy = f.result()
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
                
    # If we have time left, do a quick refinement on the best result
    if best_pos is not None and _get_time_remaining() > 10.0:
        current_pos = best_pos
        current_energy = best_energy
        rng = np.random.default_rng(n + 100)
        
        # Short basin hopping refinement
        step_size = 0.3
        for i in range(50):
            if _get_time_remaining() < 5.0:
                break
            new_pos = current_pos + rng.normal(0, step_size, current_pos.shape)
            try:
                e, p = relax(new_pos, 'dzugutov', None)
                if e < current_energy:
                    current_energy = e
                    current_pos = p
                    step_size *= 0.9
                else:
                    step_size *= 1.05
                    if step_size > 1.0: step_size = 1.0
            except:
                pass
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos

    return best_pos
