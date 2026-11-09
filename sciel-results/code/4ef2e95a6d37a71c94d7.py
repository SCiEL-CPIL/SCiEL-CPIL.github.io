"""
Optimized Morse Cluster Minimizer using Multi-threaded Adaptive Basin Hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import threading
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
TOTAL_TIME_BUDGET = 295.0
start_time_global = time.time()
global_stop_event = threading.Event()

def _time_remaining():
    return max(0.0, TOTAL_TIME_BUDGET - (time.time() - start_time_global))

def generate_fcc_shells(n):
    """Generate n atoms in FCC shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate a large FCC lattice
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 5
    coords = []
    a = np.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                p1 = base
                p2 = base + a * np.array([0, 0.5, 0.5])
                p3 = base + a * np.array([0.5, 0, 0.5])
                p4 = base + a * np.array([0.5, 0.5, 0])
                coords.extend([p1, p2, p3, p4])
                
    coords = np.array(coords)
    com = coords.mean(axis=0)
    coords = coords - com
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    selected = coords[indices[:n]]
    return selected

def generate_hcp_shells(n):
    """Generate n atoms in HCP shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 5
    coords = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3)/2 * j
                z = c * k
                p1 = np.array([x, y, z])
                p2 = p1 + np.array([a/2, a/(2*np.sqrt(3)), c/2])
                coords.append(p1)
                coords.append(p2)
                
    coords = np.array(coords)
    com = coords.mean(axis=0)
    coords = coords - com
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    selected = coords[indices[:n]]
    return selected

def generate_random_sphere(n, seed):
    """Generate n random points in a sphere with min distance constraint."""
    rng = np.random.default_rng(seed)
    radius = (n / (4/3 * np.pi * 0.6**3)) ** (1/3) # Approx radius for close packing
    radius *= 1.2 # Expand slightly
    
    points = []
    max_attempts = 100 * n
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        if all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
        else:
            attempts += 1
            
    if len(points) < n:
        # Fallback: just add more points randomly if we couldn't pack
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def generate_icosahedral_like(n, seed):
    """
    Approximate icosahedral structure.
    For small n, use icosahedron vertices.
    For larger n, add shells.
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (radius 1)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0]) # Normalize to radius 1
    
    if n <= 12:
        return verts[:n]
        
    # Add center
    points = [np.zeros(3)]
    points.extend([v for v in verts])
    
    # Add edge midpoints (30 edges)
    # This is complex to generate generically. 
    # Instead, let's just scale the icosahedron and add random points on the surface or inside.
    # A simpler "icosahedral" approach for larger N is to use a distorted FCC or just rely on the optimizer.
    # Let's generate a dense cluster by adding points in shells.
    
    # Actually, for N > 13, icosahedral structures are metastable. FCC is better.
    # So this initializer is mostly useful for N <= 13 or as a "spherical" start.
    # Let's just return the first n points from a dense spherical packing approximation.
    
    # Re-use random sphere but with a seed biased towards center?
    # Or just return FCC for n > 13? 
    # Let's just use the random sphere generator with a specific seed for variety.
    return generate_random_sphere(n, seed)

def _basin_hop_worker(initial_pos, potential, rho, n, seed, stop_event):
    """
    Run a single basin hopping chain with adaptive step size.
    Returns the best position and energy found.
    """
    rng = np.random.default_rng(seed)
    
    # Initial relaxation
    try:
        e_curr, pos_curr = relax(initial_pos, potential, rho)
    except:
        return None, np.inf
        
    e_best = e_curr
    pos_best = pos_curr.copy()
    
    # Adaptive parameters
    step_size = 0.5
    target_accept = 0.6
    accept_count = 0
    total_count = 0
    no_improve = 0
    max_no_improve = 10
    
    while not stop_event.is_set() and _time_remaining() > 1.0:
        # Perturb
        noise = rng.normal(0, step_size, size=(n, 3))
        pos_new = pos_curr + noise
        
        # Relax
        try:
            e_new, pos_new_relaxed = relax(pos_new, potential, rho)
        except:
            # If relaxation fails, shrink step
            step_size *= 0.9
            continue
            
        total_count += 1
        
        if e_new < e_curr:
            # Accept
            pos_curr = pos_new_relaxed
            e_curr = e_new
            accept_count += 1
            no_improve = 0
            
            if e_new < e_best:
                e_best = e_new
                pos_best = pos_new_relaxed.copy()
        else:
            # Reject (Metropolis accept if very close? Standard BH usually rejects if worse)
            # Let's use strict rejection for efficiency, but allow some uphill moves if stuck?
            # Standard BH: accept if e_new < e_curr.
            no_improve += 1
            
            # If stuck, try a larger jump
            if no_improve > max_no_improve:
                step_size *= 1.2
                no_improve = 0
                # Optionally reset to best? No, keep current to explore.
        
        # Adapt step size based on acceptance rate over last 100 steps
        if total_count % 100 == 0:
            current_accept = accept_count / total_count
            if current_accept > target_accept + 0.1:
                step_size *= 1.1
            elif current_accept < target_accept - 0.1:
                step_size *= 0.9
            accept_count = 0
            total_count = 0
            
        # Clamp step size
        step_size = np.clip(step_size, 0.1, 2.0)

    return pos_best, e_best

def construct(instance):
    global start_time_global, global_stop_event
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Check if we are out of time globally
    if _time_remaining() < 5.0:
        # Emergency return
        positions = np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float)
        try:
            _, positions = relax(positions, potential, rho)
        except:
            pass
        return positions

    # Reset stop event for this instance? 
    # We need to ensure we don't exceed the TOTAL budget.
    # We will run 4 threads concurrently.
    
    # Generate initial structures
    init_structures = []
    
    # 1. FCC
    try:
        init_structures.append(generate_fcc_shells(size))
    except:
        pass
        
    # 2. HCP
    try:
        init_structures.append(generate_hcp_shells(size))
    except:
        pass
        
    # 3. Random Sphere
    try:
        init_structures.append(generate_random_sphere(size, seed=42))
    except:
        pass
        
    # 4. Another Random Sphere with different seed
    try:
        init_structures.append(generate_random_sphere(size, seed=123))
    except:
        pass
        
    # If we have fewer than 4, pad with random
    while len(init_structures) < 4:
        try:
            init_structures.append(generate_random_sphere(size, seed=len(init_structures)*100))
        except:
            init_structures.append(np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float))

    # Execute 4 basin hopping chains in parallel
    # We use a ThreadPoolExecutor
    # Note: GIL might limit CPU parallelism, but relax() likely releases GIL if it's C-based or uses NumPy.
    # If relax is pure Python, threads won't help much, but they still allow interleaving.
    # Given the constraint "using at most 4 threads", this is compliant.
    
    results = []
    lock = threading.Lock()
    
    def run_chain(idx, init_pos):
        pos, e = _basin_hop_worker(init_pos, potential, rho, size, seed=idx*1000+size, stop_event=global_stop_event)
        with lock:
            if pos is not None:
                results.append((e, pos))
                
    # We need to be careful with time. 
    # We will run the threads until the global time limit is close.
    # Since we don't know how long this instance takes relative to others, 
    # we should allocate a fraction of the remaining time.
    # However, the problem says "all six calls together must finish within 300 s".
    # We can't know which instance is which.
    # Strategy: Run the threads for a fixed fraction of the *remaining* time?
    # No, we should just run them until the global stop event is triggered or a local time limit.
    # Let's set a local time limit for this instance to be 45 seconds (1/6 of 270s).
    # If we run faster, great. If slower, we cut off.
    
    local_budget = min(45.0, _time_remaining() * 0.5) # Use at most 45s or half remaining global
    local_stop_event = threading.Event()
    
    # A timer thread to stop the workers
    def timer():
        time.sleep(local_budget)
        local_stop_event.set()
        
    t_timer = threading.Thread(target=timer)
    t_timer.start()
    
    # We need to pass a stop event that is set by either local or global time.
    # Let's create a combined stop event.
    
    def combined_stop_checker():
        while not local_stop_event.is_set():
            if global_stop_event.is_set() or _time_remaining() < 2.0:
                local_stop_event.set()
                break
            time.sleep(0.1)
            
    t_checker = threading.Thread(target=combined_stop_checker)
    t_checker.start()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, init_pos in enumerate(init_structures):
            # We need a wrapper that checks local_stop_event
            def worker_wrapper(init_p, idx):
                return _basin_hop_worker(init_p, potential, rho, size, seed=idx*1000+size, stop_event=local_stop_event)
            futures.append(executor.submit(worker_wrapper, init_pos, i))
            
        for f in concurrent.futures.as_completed(futures):
            try:
                pos, e = f.result()
                if pos is not None:
                    results.append((e, pos))
            except Exception:
                pass
                
    # Stop timer
    local_stop_event.set()
    t_timer.join()
    t_checker.join()
    
    if not results:
        # Fallback
        positions = np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float)
        try:
            _, positions = relax(positions, potential, rho)
        except:
            pass
        return positions
        
    # Find best
    results.sort(key=lambda x: x[0])
    best_pos = results[0][1]
    
    return best_pos

# EVOLVE-BLOCK-END
