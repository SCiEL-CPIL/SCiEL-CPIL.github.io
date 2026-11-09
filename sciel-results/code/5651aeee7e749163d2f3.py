"""
Optimized Morse Cluster Minimizer using Structured Initializers and Basin Hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Global time management
TOTAL_TIME_BUDGET = 295.0
INSTANCE_TIME_BUDGET = TOTAL_TIME_BUDGET / 6.0
start_time_global = time.time()
instance_count = 0

def generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    a = np.sqrt(2.0)
    max_coord = int(np.ceil(n ** (1/3.0)) * 2) + 5
    coords = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                coords.extend([
                    base,
                    base + a * np.array([0, 0.5, 0.5]),
                    base + a * np.array([0.5, 0, 0.5]),
                    base + a * np.array([0.5, 0.5, 0])
                ])
                
    coords = np.array(coords)
    com = coords.mean(axis=0)
    coords = coords - com
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    return coords[indices[:n]]

def generate_hcp_shells(n, spacing=1.0):
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
                coords.extend([p1, p2])
                
    coords = np.array(coords)
    com = coords.mean(axis=0)
    coords = coords - com
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    return coords[indices[:n]]

def generate_icosahedral_shell(n):
    """Generate icosahedral-like structure for small n."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedron vertices (12)
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, t, 1/t], [0, t, -1/t], [0, -t, 1/t], [0, -t, -1/t],
        [t, 1/t, 0], [t, -1/t, 0], [-t, 1/t, 0], [-t, -1/t, 0],
        [1/t, 0, t], [1/t, 0, -t], [-1/t, 0, t], [-1/t, 0, -t]
    ], dtype=float)
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    if n <= 12:
        # Scale to have nearest neighbor distance ~1
        scale = 1.0 / np.linalg.norm(vertices[0] - vertices[1])
        return vertices[:n] * scale
    
    # For larger n, add center and scale
    if n == 13:
        return np.vstack([vertices * 0.5, [[0, 0, 0]]])
    
    # For n > 13, use FCC as fallback
    return generate_fcc_shells(n)

def generate_random_close_pack(n, rng, min_dist=0.8):
    """Generate random close-packed structure."""
    radius = n ** (1.0 / 3.0) * 0.9
    points = []
    max_attempts = 50 * n
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    
    if len(points) < n:
        # Fallback: add remaining points with minimum distance check
        for _ in range(n - len(points)):
            for attempt in range(100):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    if len(points) == 0 or all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
                        break
    
    return np.array(points)

def construct(instance):
    global start_time_global, instance_count
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    instance_count += 1
    elapsed = time.time() - start_time_global
    
    if elapsed > TOTAL_TIME_BUDGET - 5.0:
        positions = np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float)
        try:
            _, positions = relax(positions, potential, rho)
        except:
            pass
        return positions

    remaining_total = TOTAL_TIME_BUDGET - elapsed
    inst_budget = min(INSTANCE_TIME_BUDGET, remaining_total * 0.4)
    
    start_time_inst = time.time()
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse initial structures
    candidates = []
    
    # FCC
    try:
        candidates.append(generate_fcc_shells(size))
    except:
        pass
        
    # HCP
    try:
        candidates.append(generate_hcp_shells(size))
    except:
        pass
        
    # Icosahedral (for smaller n)
    if size <= 20:
        try:
            candidates.append(generate_icosahedral_shell(size))
        except:
            pass
        
    # Random close-packed structures (multiple)
    for _ in range(3):
        try:
            candidates.append(generate_random_close_pack(size, rng))
        except:
            pass
        
    if not candidates:
        candidates.append(np.array([np.array([i, 0, 0]) for i in range(size)], dtype=float))

    # Relax all candidates and track best
    best_energy = np.inf
    best_pos = None
    relaxed_candidates = []
    
    for pos in candidates:
        try:
            e, relaxed_pos = relax(pos, potential, rho)
            relaxed_candidates.append((e, relaxed_pos))
            if e < best_energy:
                best_energy = e
                best_pos = relaxed_pos
        except:
            pass
            
    if best_pos is None:
        return candidates[0]

    # Multi-start basin hopping with adaptive noise
    # Use simulated annealing-like acceptance
    T_initial = 0.5
    T_final = 0.05
    current_energy = best_energy
    current_pos = best_pos.copy()
    
    # Sort relaxed candidates by energy
    relaxed_candidates.sort(key=lambda x: x[0])
    
    # Perform basin hopping from multiple starting points
    total_steps_budget = max(10, int(inst_budget / 0.05))  # Estimate ~50ms per step
    
    steps_per_start = total_steps_budget // max(1, len(relaxed_candidates[:3]))
    
    for start_idx, (e_start, pos_start) in enumerate(relaxed_candidates[:3]):
        if time.time() - start_time_inst >= inst_budget:
            break
        if time.time() - start_time_global > TOTAL_TIME_BUDGET - 2.0:
            break
            
        current_energy = e_start
        current_pos = pos_start.copy()
        
        # Adaptive noise schedule
        noise_scale = 0.6
        no_improve = 0
        max_no_improve = 15
        
        for step in range(steps_per_start):
            if time.time() - start_time_inst >= inst_budget:
                break
            if time.time() - start_time_global > TOTAL_TIME_BUDGET - 2.0:
                break
                
            # Perturb
            noise = rng.normal(0, noise_scale, size=(size, 3))
            perturbed_pos = current_pos + noise
            
            # Relax
            try:
                e_new, pos_new = relax(perturbed_pos, potential, rho)
            except:
                noise_scale *= 0.9
                continue
            
            # Acceptance criterion: always accept if better, else with probability exp(-delta/T)
            if e_new < current_energy:
                current_energy = e_new
                current_pos = pos_new
                no_improve = 0
                noise_scale *= 0.98  # Slowly decrease noise on success
            else:
                # Simulated annealing acceptance
                T = T_initial * (T_final / T_initial) ** (step / max(1, steps_per_start))
                delta = e_new - current_energy
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    current_energy = e_new
                    current_pos = pos_new
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        # Reset to best and increase noise
                        current_pos = best_pos.copy()
                        current_energy = best_energy
                        no_improve = 0
                        noise_scale = min(noise_scale * 1.2, 1.0)
            
            # Update global best
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos

# EVOLVE-BLOCK-END
