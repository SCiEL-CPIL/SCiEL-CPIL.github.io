"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Leave 10s safety margin for 300s limit (evaluator kills at 330s, but task says 300s)
# Actually, the prompt says "all calls together within 300 s" and "evaluator KILLS ... after 330s".
# To be safe, aim for 290s.

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _generate_mackay_icosahedron(n, scale=0.9):
    """
    Generates a Mackay icosahedral cluster of n atoms.
    Shells:
    1. Center (1 atom)
    2. 1st Shell (12 atoms) - Icosahedron vertices
    3. 2nd Shell (42 atoms) - 30 Face Centers + 12 Edge Midpoints? 
       Actually, Mackay icosahedron 2nd shell consists of:
       - 12 atoms at edge midpoints (distance from center: 1.5 * R1? No.)
       Let's use geometric construction.
       
       R1 = 1.0 (arbitrary unit)
       R2 = sqrt(1 + 1/phi^2)? No.
       
       Standard Mackay Icosahedron radii (relative to R1=1):
       Shell 1: 1 atom at r=0
       Shell 2: 12 atoms at r=1 (Icosahedron vertices)
       Shell 3: 42 atoms at r=sqrt(1+1/phi^2) ~ 1.272? 
       Wait, the 2nd shell of a Mackay icosahedron is composed of:
       - 12 atoms at the midpoints of the edges of the icosahedron? No, that's 30 edges.
       - 12 atoms at the centers of the faces? No, that's 20 faces.
       
       Let's look up the Mackay icosahedron structure.
       It is a 3D cluster with icosahedral symmetry.
       Magic numbers: 1, 13, 55, 147, 309, 561.
       
       For n=55:
       Center: 1
       Shell 1: 12 (Icosahedron vertices)
       Shell 2: 42
       
       How to generate Shell 2 (42 atoms)?
       The 42 atoms are located at:
       - 30 atoms at the midpoints of the 30 edges of the icosahedron? 
         Distance from center for edge midpoint: 
         If vertices are at distance R, edge length is 2R sin(pi/5)? No, in unit icosahedron, edge length is 2.
         Midpoint of edge connecting (0,1,phi) and (0,-1,phi) is (0,0,phi). Distance = phi = 1.618.
       - 12 atoms at the centers of the 12 vertices? No, those are already there.
       - 12 atoms at the centers of the 12 faces? No, icosahedron has 20 faces.
       
       Actually, the Mackay icosahedron is built by packing spheres.
       Shell 2 atoms are at distance R2 = R1 * sqrt(1 + 1/phi^2) ?
       Let's check the distance.
       If R1 = 1.
       Nearest neighbor distance in Shell 1 is 2 (for unit icosahedron).
       In Mackay, the 2nd shell atoms are tangent to the 1st shell atoms.
       
       Let's use a simpler robust method:
       1. Generate 13-atom icosahedron.
       2. For n > 13, add atoms at the "holes" or expand the shell.
       
       Alternative: Use a known coordinate set for the 55-atom Mackay icosahedron and truncate/expand.
       But we need 38, 45, 50, 55, 57, 60.
       
       Let's generate the 55-atom Mackay icosahedron properly.
       Coordinates for 55-atom Mackay Icosahedron (normalized to min distance ~ 1.8):
       
       We can generate it by:
       1. 12 vertices of icosahedron (Shell 1)
       2. 30 edge midpoints? 
       3. 20 face centers?
       12 + 30 + 20 = 62. Too many.
       
       Correct Mackay 55:
       1 (center)
       12 (vertices)
       42 (second shell)
       
       The 42 atoms in the second shell are:
       - 12 atoms at distance R2 from center, along the directions of the 12 vertices? No.
       - 30 atoms at distance R2 from center, along the directions of the 30 edges?
       - 12 atoms at distance R2 from center, along the directions of the 12 faces of the dual dodecahedron?
       
       Let's use the property that it's a close-packed structure.
       We can generate a dense random cloud on a sphere of radius R2 and relax it, but that's not "exact".
       
       Better approach:
       Generate the 13-atom icosahedron.
       For the next atoms, place them at the centers of the 20 faces of the icosahedron?
       Distance from center to face center:
       In unit icosahedron, face center distance is 1/phi^2 ? No.
       
       Let's just use a robust "shell filling" strategy:
       1. Start with 13-atom icosahedron.
       2. Add atoms one by one at the position that minimizes the energy of the current cluster, 
          constrained to be at a distance > 0.5 from all existing atoms.
          This is slow for n=60.
       
       Let's try a different geometric construction for the 55-atom Mackay icosahedron.
       The 55-atom Mackay icosahedron can be constructed by:
       - 1 central atom
       - 12 atoms in the first shell (icosahedron vertices)
       - 42 atoms in the second shell.
       
       The second shell consists of:
       - 12 atoms at the same polar angles as the first shell vertices but at a larger radius? No.
       
       Actually, there is a known set of coordinates. 
       Since I can't look it up, I will use a "perturbed icosahedron + relaxation" strategy which is more robust.
       
       Strategy:
       1. Generate a 13-atom icosahedron.
       2. For n > 13, add atoms randomly on a sphere of radius R2 = 1.5 * R1, then relax the whole cluster.
       This effectively finds the local minimum which is likely the Mackay structure for these sizes.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale to nearest neighbor distance ~ 1.8
    # Min dist in unit icosahedron is 2.
    # So scale = 1.8 / 2 = 0.9
    verts = verts * 0.9 * scale
    
    points = [np.zeros(3)]  # Central atom
    if n > 1:
        points.extend([v.copy() for v in verts[:min(n-1, 12)]])
    
    # If n > 13, add atoms on a second shell
    if n > 13:
        # Estimate second shell radius
        # In Mackay, R2/R1 = sqrt(1 + 1/phi^2) ? 
        # Let's try R2 = 1.5 * R1 (approx)
        R2 = 0.9 * scale * 1.5
        # We need n - len(points) more atoms
        num_to_add = n - len(points)
        # Use a Fibonacci sphere or random points on sphere
        # Random points on sphere
        for _ in range(num_to_add):
            # Random direction
            z = np.random.uniform(-1, 1)
            theta = np.random.uniform(0, 2 * np.pi)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z]) * R2
            
            # Check min distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                # If invalid, try again with slightly different radius
                # Or just place it and let relax fix it
                points.append(p)
                
    return np.array(points)

def _generate_random_cluster(n):
    """Generate a random cluster with minimum distance 0.5."""
    points = []
    for _ in range(n):
        for _attempt in range(50):
            p = np.random.uniform(-2, 2, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
        else:
            points.append(np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)]))
    return np.array(points)

def _basin_hopping_worker(start_pos, potential, parameter, time_limit, best_energy_holder, lock):
    """
    Worker for Basin Hopping.
    start_pos: initial positions
    potential: 'dzugutov'
    parameter: None or dict
    time_limit: time to run this worker
    best_energy_holder: [best_energy, best_positions]
    lock: threading.Lock
    """
    pos = start_pos.copy()
    e, pos_relaxed = relax(pos, potential, parameter)
    current_energy = e
    
    T = 5.0
    cooling_rate = 0.99
    step_size = 0.3
    
    end_time = time.time() + time_limit
    
    while time.time() < end_time:
        # Perturb
        # Move a random subset of atoms
        n_perturb = max(1, len(pos) // 10)
        indices = np.random.choice(len(pos), n_perturb, replace=False)
        new_pos = pos.copy()
        for i in indices:
            new_pos[i] += np.random.normal(0, step_size, 3)
        
        # Relax
        try:
            new_e, new_pos_relaxed = relax(new_pos, potential, parameter)
        except Exception:
            continue
            
        delta_e = new_e - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            pos = new_pos_relaxed
            current_energy = new_e
            
            # Update global best
            with lock:
                if current_energy < best_energy_holder[0]:
                    best_energy_holder[0] = current_energy
                    best_energy_holder[1] = pos.copy()
                    
            # Adaptive step size: reduce if improving, increase if stuck?
            # Simple: keep step size constant or slightly reduce
            step_size *= 0.999
            if step_size < 0.1:
                step_size = 0.1
                
        T *= cooling_rate
        if T < 0.1:
            T = 5.0 # Reheat

    return current_energy

def construct(instance):
    """
    Strategy:
    1. Generate a high-quality initial guess: Mackay Icosahedron.
    2. Relax it.
    3. Run parallel Basin Hopping from this initial guess (and a few perturbed versions) to ensure we find the global minimum.
    4. Return the best structure found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # 1. Generate Initial Guess
    # Use the Mackay generator
    init_pos = _generate_mackay_icosahedron(size, 1.0)
    
    # Relax the initial guess
    try:
        e_init, pos_init = relax(init_pos, potential, parameter)
    except Exception:
        # Fallback to random
        init_pos = _generate_random_cluster(size)
        e_init, pos_init = relax(init_pos, potential, parameter)
        
    best_energy = e_init
    best_positions = pos_init.copy()
    
    # 2. Parallel Basin Hopping
    # We have 300s total. 
    # Let's allocate 250s for BH.
    # Use 4 threads.
    num_threads = 4
    time_per_thread = 250.0 / num_threads
    
    # Create multiple starting points:
    # 1. The relaxed initial guess
    # 2. Perturbed versions
    start_points = [pos_init.copy()]
    for _ in range(3):
        perturbed = pos_init + np.random.normal(0, 0.2, (size, 3))
        start_points.append(perturbed)
        
    # Ensure we have exactly num_threads start points
    while len(start_points) < num_threads:
        start_points.append(pos_init + np.random.normal(0, 0.5, (size, 3)))
        
    best_energy_holder = [best_energy, best_positions]
    lock = threading.Lock()
    
    # Launch workers
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for sp in start_points:
            # Check time
            if not _check_time():
                break
            futures.append(executor.submit(_basin_hopping_worker, sp, potential, parameter, time_per_thread, best_energy_holder, lock))
            
        # Wait for completion or timeout
        try:
            for future in as_completed(futures, timeout=260):
                future.result()
        except Exception:
            pass
            
    # 3. Final Relaxation of the best found
    try:
        e_final, pos_final = relax(best_energy_holder[1], potential, parameter)
        return pos_final
    except Exception:
        return best_energy_holder[1]

# EVOLVE-BLOCK-END
