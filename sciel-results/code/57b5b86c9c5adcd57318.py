"""Improved program for Dzugutov cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Time management
_START_TIME = time.time()
TIME_LIMIT = 280.0  # Safety margin before 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _get_time_remaining():
    return max(0, TIME_LIMIT - (time.time() - _START_TIME))

def _generate_mackay_icosahedron(n, scale=1.0):
    """
    Generate a Mackay icosahedral cluster with proper shell structure.
    Shells: 1 (center), 12 (1st), 42 (2nd), 120 (3rd), 220 (4th)
    Total magic numbers: 1, 13, 55, 175, 395
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms * scale
    
    positions = [np.zeros(3)]  # Center atom
    
    # First shell: 12 atoms at distance 1*scale
    if n > 1:
        positions.extend(ico_verts.tolist())
    
    if n > 13:
        # Second shell: 42 atoms
        # These are face centers and edge midpoints of the icosahedron
        # Face centers: 20 faces -> 20 points at distance sqrt(3)*scale from center
        # Edge midpoints: 30 edges -> 30 points at distance 0.5*sqrt(phi+1)*scale... 
        # Actually, let's use the known coordinates for the 42-atom shell
        
        # The 42 points consist of:
        # - 20 face centers (at distance sqrt(3) from center for unit icosahedron)
        # - 30 edge midpoints (at distance 0.5*sqrt(2+phi) from center... let me compute)
        
        # For unit icosahedron (vertices at distance 1):
        # Edge midpoint distance from center: half the edge length
        # Edge length = sqrt(2 - 2*dot(v1, v2)) for adjacent vertices
        # Adjacent vertices have dot product = 1/sqrt(1+phi^2) = 1/sqrt(phi+2)
        # Actually, for our normalized icosahedron, edge length = sqrt(2 - 2*cos(angle))
        # The angle between adjacent vertices: cos(theta) = 1/sqrt(1+phi^2) = 1/sqrt(phi+2)
        # phi+2 = (3+sqrt(5))/2 ≈ 2.618, sqrt ≈ 1.618 = phi
        # So cos(theta) = 1/phi ≈ 0.618, theta ≈ 51.8°
        # Edge length = sqrt(2 - 2/phi) = sqrt(2 - 0.2/1)... let me just compute numerically
        
        # Simpler approach: generate 42 points by combining:
        # 20 face centers + 30 edge midpoints of the icosahedron
        
        # Compute face normals (face centers)
        # Icosahedron faces: need to identify triangles
        # This is complex, so let's use a different approach:
        # Place 42 points on a sphere of radius r2 using icosahedral symmetry
        
        # Alternative: use the fact that the 42-shell is a known polyhedron
        # Let's just place them on a sphere at radius ~1.9*scale with good spacing
        
        r2 = 1.9 * scale
        # Generate 42 points with icosahedral symmetry
        # Start with one point and apply icosahedral rotation group (60 elements)
        # But 60 > 42, so we need a subset
        
        # Simpler: use Fibonacci sphere or similar for 42 points, then relax
        # For now, place 42 points uniformly on sphere of radius r2
        golden_angle = np.pi * (3 - np.sqrt(5))
        face_centers = []
        for i in range(42):
            z = 1 - 2 * i / 41
            r = np.sqrt(1 - z*z)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r2
            face_centers.append(p)
        
        positions.extend(face_centers[:min(42, n - 13)])
    
    # If we still need more atoms (n > 55), add third shell
    if n > 55:
        r3 = 2.8 * scale
        golden_angle = np.pi * (3 - np.sqrt(5))
        third_shell = []
        needed = n - len(positions)
        for i in range(needed):
            z = 1 - 2 * i / max(1, needed - 1)
            r = np.sqrt(1 - z*z)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r3
            third_shell.append(p)
        positions.extend(third_shell)
    
    pos_array = np.array(positions[:n])
    
    # Ensure minimum distance of 0.5
    # Quick check and adjustment if needed
    for i in range(len(pos_array)):
        for j in range(i+1, len(pos_array)):
            d = np.linalg.norm(pos_array[i] - pos_array[j])
            if d < 0.5 and d > 1e-10:
                # Nudge apart
                direction = (pos_array[i] - pos_array[j]) / d
                pos_array[i] += 0.01 * direction
                pos_array[j] -= 0.01 * direction
    
    return pos_array

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = 0.8 * n ** (1.0/3.0)
    points = [np.zeros(3)]  # Start with center
    
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        # Sample point in sphere
        r = radius * np.cbrt(rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        # Check minimum distance
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            points.append(p)
    
    # If we couldn't place all, add remaining randomly (relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _generate_bcc_cluster(n, spacing=1.0):
    """Generate a BCC-like cluster as fallback."""
    points = []
    L = int(np.ceil((n / 2.0) ** (1/3))) + 1
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            for z in range(-L, L+1):
                p = np.array([x, y, z]) * spacing
                if np.linalg.norm(p) < 6*spacing:
                    points.append(p)
                pc = (np.array([x, y, z]) + 0.5) * spacing
                if np.linalg.norm(pc) < 6*spacing:
                    points.append(pc)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    if len(points) > n:
        points = sorted(points, key=lambda p: np.linalg.norm(p))[:n]
    else:
        rng = np.random.default_rng(123)
        while len(points) < n:
            p = rng.uniform(-3*spacing, 3*spacing, 3)
            points.append(p)
    
    return np.array(points)

def _relax_with_timeout(positions, potential, parameter, timeout=5.0):
    """Relax with a timeout to prevent hanging."""
    result = {'energy': None, 'pos': None, 'done': False}
    
    def _do_relax():
        try:
            e, p = relax(positions, potential, parameter)
            result['energy'] = e
            result['pos'] = p
        except Exception as ex:
            pass
        result['done'] = True
    
    t = threading.Thread(target=_do_relax)
    t.start()
    t.join(timeout=timeout)
    
    if result['done'] and result['pos'] is not None:
        return result['energy'], result['pos']
    return None, None

def _basin_hopping_worker(initial_pos, potential, parameter, max_iter, time_limit, result_holder, lock):
    """
    Perform basin hopping from a given initial position.
    Updates result_holder with best found energy/position.
    """
    if not _check_time():
        return
    
    # Initial relaxation
    energy, pos = _relax_with_timeout(initial_pos, potential, parameter, timeout=10.0)
    if energy is None:
        return
    
    best_energy = energy
    best_pos = pos.copy()
    current_energy = energy
    current_pos = pos.copy()
    
    rng = np.random.default_rng(hash(id(initial_pos)) % 2**32)
    
    # Adaptive step size
    step_size = 0.6
    
    for i in range(max_iter):
        if not _check_time():
            break
        if time.time() > time_limit:
            break
        
        # Multi-atom perturbation: perturb a random subset of atoms
        n_perturb = max(1, min(5, len(current_pos) // 10))
        indices = rng.choice(len(current_pos), size=n_perturb, replace=False)
        perturbed = current_pos.copy()
        perturbed[indices] += rng.normal(0, step_size, (n_perturb, 3))
        
        # Relax
        new_energy, new_pos = _relax_with_timeout(perturbed, potential, parameter, timeout=8.0)
        if new_energy is None:
            continue
        
        # Metropolis acceptance with cooling temperature
        # Temperature decreases over iterations
        T = 0.5 * (1.0 - i / max(1, max_iter))
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / max(T, 0.01)):
            current_energy = new_energy
            current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
            # Update global best
            with lock:
                if result_holder['energy'] is None or best_energy < result_holder['energy']:
                    result_holder['energy'] = best_energy
                    result_holder['pos'] = best_pos.copy()
        
        # Adaptive step size: increase if stuck, decrease if improving
        if new_energy < current_energy:
            step_size = max(0.3, step_size * 0.95)
        else:
            step_size = min(1.5, step_size * 1.05)
    
    # Final update
    with lock:
        if result_holder['energy'] is None or best_energy < result_holder['energy']:
            result_holder['energy'] = best_energy
            result_holder['pos'] = best_pos.copy()

def _simulated_annealing_worker(initial_pos, potential, parameter, max_iter, time_limit, result_holder, lock):
    """
    Perform simulated annealing from a given initial position.
    Different from basin hopping: uses smaller steps and more iterations.
    """
    if not _check_time():
        return
    
    # Initial relaxation
    energy, pos = _relax_with_timeout(initial_pos, potential, parameter, timeout=10.0)
    if energy is None:
        return
    
    best_energy = energy
    best_pos = pos.copy()
    current_energy = energy
    current_pos = pos.copy()
    
    rng = np.random.default_rng(hash(id(initial_pos) + 1) % 2**32)
    
    # SA parameters
    T_start = 2.0
    T_end = 0.05
    n_atoms = len(current_pos)
    
    for i in range(max_iter):
        if not _check_time():
            break
        if time.time() > time_limit:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (i / max(1, max_iter))
        
        # Single-atom perturbation with small step
        idx = rng.integers(0, n_atoms)
        step = 0.3 * np.sqrt(T)
        perturbed = current_pos.copy()
        perturbed[idx] += rng.normal(0, step, 3)
        
        # Relax
        new_energy, new_pos = _relax_with_timeout(perturbed, potential, parameter, timeout=5.0)
        if new_energy is None:
            continue
        
        # Metropolis acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 0.01)):
            current_energy = new_energy
            current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
            with lock:
                if result_holder['energy'] is None or best_energy < result_holder['energy']:
                    result_holder['energy'] = best_energy
                    result_holder['pos'] = best_pos.copy()
    
    with lock:
        if result_holder['energy'] is None or best_energy < result_holder['energy']:
            result_holder['energy'] = best_energy
            result_holder['pos'] = best_pos.copy()

def construct(instance):
    """
    Strategy:
    1. Generate multiple diverse initial guesses (Mackay icosahedron, random, BCC).
    2. Run parallel basin hopping and simulated annealing on 4 threads.
    3. Return the best result found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", "dzugutov"))
    
    # Time budget
    time_limit = _START_TIME + TIME_LIMIT - 10  # Leave 10s safety margin
    
    # Global best result
    result_holder = {'energy': None, 'pos': None}
    lock = threading.Lock()
    
    # Generate initial guesses
    ico_pos = _generate_mackay_icosahedron(size)
    rand_pos = _generate_random_cluster(size, seed=size)
    bcc_pos = _generate_bcc_cluster(size)
    perturbed_ico = ico_pos + np.random.default_rng(999).normal(0, 0.15, ico_pos.shape)
    
    # Define worker functions
    def worker_bh_ico():
        _basin_hopping_worker(ico_pos, potential, parameter, 80, time_limit, result_holder, lock)
    
    def worker_bh_rand():
        _basin_hopping_worker(rand_pos, potential, parameter, 100, time_limit, result_holder, lock)
    
    def worker_sa_ico():
        _simulated_annealing_worker(ico_pos, potential, parameter, 200, time_limit, result_holder, lock)
    
    def worker_sa_perturbed():
        _simulated_annealing_worker(perturbed_ico, potential, parameter, 200, time_limit, result_holder, lock)
    
    # Run 4 workers in parallel
    threads = [
        threading.Thread(target=worker_bh_ico),
        threading.Thread(target=worker_bh_rand),
        threading.Thread(target=worker_sa_ico),
        threading.Thread(target=worker_sa_perturbed),
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Return best result or fallback
    if result_holder['pos'] is not None:
        return result_holder['pos']
    else:
        # Fallback: return the icosahedral structure
        return ico_pos

# EVOLVE-BLOCK-END
