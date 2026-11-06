import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
POTENTIAL = 'dzugutov'
# Global time tracking
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_mackay_icosahedron(size):
    """
    Generate Mackay icosahedron seeds for various sizes.
    Uses icosahedral symmetry and defect seeding for sizes near 55.
    """
    ico_verts = _get_icosahedron_vertices()
    phi = (1 + np.sqrt(5)) / 2
    
    # 13-atom Mackay icosahedron: center + 12 icosahedron vertices
    if size == 13:
        center = np.zeros(3)
        return np.vstack([center, ico_verts])
    
    # For sizes 38-60, use a strategy based on the 55-atom Mackay icosahedron
    # with atoms added/removed to reach the target size
    
    # Base 55-atom structure: 1 + 12 + 42
    # Layer 1: 1 center
    # Layer 2: 12 at radius r2
    # Layer 3: 42 atoms (30 edge midpoints + 12 face centers)
    
    r2 = 1.0
    r3 = r2 * np.sqrt(1 + 1/phi**2)
    
    # Generate 55-atom base
    center = np.zeros(3)
    shell2 = ico_verts * r2
    
    # Find edges and faces of icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    
    # 30 edge midpoints
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                mid_norm = np.linalg.norm(mid)
                edge_mids.append(mid / mid_norm * r3)
    
    # 20 face centers
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if abs(d1 - min_dist) < 1e-6 and abs(d2 - min_dist) < 1e-6 and abs(d3 - min_dist) < 1e-6:
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
        fc_norm = np.linalg.norm(fc)
        face_centers.append(fc / fc_norm * r3)
    
    # 55-atom: 1 + 12 + 30 + 12 (select 12 of 20 face centers)
    # For simplicity, use all 42 from shell3 for now, then trim/add as needed
    shell3_all = np.vstack([np.array(edge_mids), np.array(face_centers)])
    
    # Build 55-atom structure
    base55 = np.vstack([center[None], shell2, shell3_all[:42]])
    
    if size == 55:
        return base55
    
    # For other sizes, add or remove atoms
    if size < 55:
        # Remove atoms from the outer shell, keeping symmetry as much as possible
        # Start with center + shell2 (13 atoms), then add from shell3
        needed = size - 13
        if needed <= 0:
            return base55[:size]
        # Select 'needed' atoms from shell3, preferring those with higher coordination
        # For simplicity, just take the first 'needed'
        result = np.vstack([center[None], shell2, shell3_all[:needed]])
        return result
    
    else:
        # size > 55: add atoms on a slightly larger shell
        base = base55.copy()
        extra = size - 55
        # Add atoms on a sphere of radius r4
        r4 = r3 * 1.15
        extra_atoms = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(extra):
            y = 1 - 2 * (i + 0.5) / extra
            radius_at_y = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            extra_atoms.append([x, y, z] * r4)
        extra_atoms = np.array(extra_atoms)
        return np.vstack([base, extra_atoms])

def _generate_seed(size, rng, variant=0):
    """Generate diverse seeds using icosahedral symmetry and defect seeding."""
    if variant == 0:
        # Perfect icosahedral seed
        try:
            seed = _generate_mackay_icosahedron(size)
            if seed.shape[0] == size:
                return seed
        except:
            pass
        # Fallback: random
        radius = 0.8 * size ** (1/3) + 1.0
        points = []
        for _ in range(size):
            for _ in range(200):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
                    break
            if len(points) < size:
                points.append(rng.uniform(-radius, radius, 3))
        return np.array(points)
    
    elif variant == 1:
        # Defect-seeded: icosahedral with one atom moved
        try:
            seed = _generate_mackay_icosahedron(size).copy()
            if seed.shape[0] == size:
                # Move one random outer atom
                idx = rng.integers(13, size)
                seed[idx] += rng.normal(0, 0.3, 3)
                return seed
        except:
            pass
        # Fallback
        radius = 0.8 * size ** (1/3) + 1.0
        return rng.uniform(-radius, radius, (size, 3))
    
    elif variant == 2:
        # Perturbed icosahedral
        try:
            seed = _generate_mackay_icosahedron(size).copy()
            if seed.shape[0] == size:
                seed += rng.normal(0, 0.08, (size, 3))
                return seed
        except:
            pass
        radius = 0.8 * size ** (1/3) + 1.0
        return rng.uniform(-radius, radius, (size, 3))
    
    else:
        # Random compact
        radius = 0.8 * size ** (1/3) + 1.0
        points = []
        for _ in range(size):
            for _ in range(200):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
                    break
            if len(points) < size:
                points.append(rng.uniform(-radius, radius, 3))
        return np.array(points)

def _basin_hopping(positions, rng, time_limit, initial_temp=0.12, initial_step=0.20):
    """Adaptive basin hopping with energy-based acceptance and adaptive step size."""
    current_pos = positions.copy()
    current_energy, current_relaxed = relax(current_pos, POTENTIAL)
    best_energy = current_energy
    best_pos = current_relaxed.copy()
    
    temperature = initial_temp
    step_size = initial_step
    consecutive_rejects = 0
    improvements = 0
    
    while time.time() < time_limit:
        # Adaptive step size: larger when stuck, smaller when improving
        if improvements > 0:
            step_size *= 0.97
            improvements = 0
        elif consecutive_rejects > 10:
            step_size *= 1.15
            consecutive_rejects = 0
        
        step_size = np.clip(step_size, 0.05, 0.40)
        
        perturbation = rng.normal(0, step_size, size=(len(positions), 3))
        new_pos = current_pos + perturbation
        
        try:
            new_energy, new_relaxed = relax(new_pos, POTENTIAL)
        except:
            continue
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_relaxed.copy()
            current_pos = new_relaxed.copy()
            current_energy = new_energy
            consecutive_rejects = 0
            improvements += 1
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0:
                # Accept downhill moves
                current_pos = new_relaxed.copy()
                current_energy = new_energy
                consecutive_rejects = 0
            else:
                # Metropolis acceptance
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_pos = new_relaxed.copy()
                    current_energy = new_energy
                consecutive_rejects += 1
        
        # Adaptive temperature
        if consecutive_rejects > 25:
            temperature *= 1.3
            consecutive_rejects = 0
        else:
            temperature *= 0.96
        
        temperature = max(temperature, 0.005)
    
    return best_pos, best_energy

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 280.0  # Safety margin

    size = instance["n"]
    start_time = time.time()
    
    # Time budget: distribute among instances
    # 6 instances, total 280s -> ~46s each
    instance_deadline = min(start_time + 44.0, _GLOBAL_DEADLINE - 5.0)
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse seeds and relax them
    n_seeds = 5
    seeds = []
    for variant in range(n_seeds):
        seed = _generate_seed(size, rng, variant)
        if seed.shape[0] == size:
            seeds.append(seed)
    
    # Relax each seed and keep the best
    for seed in seeds:
        if time.time() > instance_deadline - 8:
            break
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            continue
    
    if best_positions is None:
        # Fallback
        radius = 0.8 * size ** (1/3) + 1.0
        best_positions = rng.uniform(-radius, radius, (size, 3))
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass
    
    # Run multiple basin hopping trajectories from different starting points
    remaining_time = instance_deadline - time.time()
    if remaining_time > 3:
        # Allocate time to multiple BH trajectories
        n_bh = min(3, max(1, int(remaining_time / 10)))
        time_per_bh = (remaining_time - 2) / n_bh
        
        for i in range(n_bh):
            if time.time() > instance_deadline - 2:
                break
            
            # Start BH from best position or a perturbed version
            if i == 0:
                start_pos = best_positions.copy()
            else:
                start_pos = best_positions + rng.normal(0, 0.15, (size, 3))
            
            bh_deadline = time.time() + time_per_bh
            try:
                bh_pos, bh_energy = _basin_hopping(start_pos, rng, bh_deadline)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
            except:
                continue
    
    return best_positions
