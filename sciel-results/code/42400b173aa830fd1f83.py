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

def _get_dodecahedron_vertices():
    """Return the 20 vertices of a regular dodecahedron (dual of icosahedron), normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_mackay_icosahedron(size):
    """
    Generate Mackay icosahedron seeds for various sizes.
    M(1) = 1, M(2) = 13, M(3) = 55
    For other sizes, we grow from these base structures.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = _get_icosahedron_vertices()
    
    # 13-atom Mackay icosahedron: center + 12 icosahedron vertices
    if size == 13:
        center = np.zeros(3)
        return np.vstack([center, ico_verts])
    
    # 55-atom Mackay icosahedron: 1 + 12 + 42
    if size == 55:
        center = np.zeros(3)
        # Second shell: 12 icosahedron vertices at radius r2
        r2 = 1.0
        shell2 = ico_verts * r2
        
        # Third shell: 42 atoms
        # 30 edge midpoints + 12 dodecahedron vertices (face centers of icosahedron)
        # Find edges: pairs of icosahedron vertices with minimum distance
        dists = np.zeros((12, 12))
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                dists[i, j] = d
                dists[j, i] = d
        
        min_dist = np.min(dists[dists > 0])
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if abs(dists[i, j] - min_dist) < 1e-6:
                    edges.append((i, j))
        
        # 30 edge midpoints
        edge_mids = []
        for i, j in edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2
            edge_mids.append(mid)
        edge_mids = np.array(edge_mids)
        # Scale edge midpoints to same radius as dodecahedron vertices
        edge_mid_norms = np.linalg.norm(edge_mids, axis=1, keepdims=True)
        edge_mids_scaled = edge_mids / edge_mid_norms * r2 * np.sqrt(1 + 1/phi**2)
        
        # 12 dodecahedron vertices (face centers)
        dod_verts = _get_dodecahedron_vertices()
        # The 12 dodecahedron vertices that correspond to icosahedron face centers
        # Actually, we need 12 specific ones. Let's use the 12 vertices of the dual.
        # For now, use all 20 and select 12? No, we need exactly 12.
        # The 12 face centers of the icosahedron form a dodecahedron.
        # Let's compute face centers from icosahedron faces.
        
        # Find faces: triples of vertices forming equilateral triangles
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                    d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                    if abs(d1 - min_dist) < 1e-6 and abs(d2 - min_dist) < 1e-6 and abs(d3 - min_dist) < 1e-6:
                        faces.append((i, j, k))
        
        # There should be 20 faces
        face_centers = []
        for i, j, k in faces:
            fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
            face_centers.append(fc)
        face_centers = np.array(face_centers)
        face_center_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
        face_centers_scaled = face_centers / face_center_norms * r2 * np.sqrt(1 + 1/phi**2)
        
        # We have 30 edge midpoints and 20 face centers = 50, but we need 42.
        # The Mackay icosahedron 3rd shell has 42 atoms: 30 edge midpoints + 12 face centers?
        # Actually, let me reconsider. The 42 atoms are:
        # - 30 at edge midpoints
        # - 12 at the vertices of the dual dodecahedron (which are the face centers of the icosahedron)
        # But there are 20 face centers, not 12.
        
        # Let me just use a different approach: generate the 55-atom structure using known coordinates
        # or use a simpler growth strategy.
        
        # Alternative: use the 13-atom icosahedron and add 42 atoms in a polytetrahedral pattern
        # For now, let's create a seed by placing atoms on a truncated icosahedron-like surface
        
        # Simplified: use icosahedron vertices and add atoms at face centers
        # This gives 1 + 12 + 20 = 33, not 55.
        
        # Let's use a different strategy: start with 13-atom and grow
        # For 55, we can use the known fact that it's a local minimum
        # Let's just create a reasonable seed and let relaxation handle it
        
        # Create 55-atom seed: 1 center + 12 at r=1 + 42 at r~1.05
        r3 = r2 * np.sqrt(1 + 1/phi**2)
        
        # 42 atoms: distribute on sphere of radius r3
        # Use Fibonacci sphere for even distribution
        n_outer = 42
        golden_angle = np.pi * (3 - np.sqrt(5))
        outer_atoms = []
        for i in range(n_outer):
            y = 1 - 2 * i / (n_outer - 1)
            radius_at_y = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            outer_atoms.append([x, y, z] * r3)
        outer_atoms = np.array(outer_atoms)
        
        return np.vstack([center, shell2, outer_atoms])
    
    # For other sizes, grow from 13 or 55
    if size <= 13:
        # Use 13-atom and remove atoms
        base = _generate_mackay_icosahedron(13)
        if size == 13:
            return base
        # Remove outer atoms
        return base[:size]
    
    if size <= 55:
        # Start with 13 and add atoms
        base = _generate_mackay_icosahedron(13)
        current = base.copy()
        r = 1.0
        for _ in range(size - 13):
            # Add atom at random position on surface
            u1 = np.random.uniform(0, 2 * np.pi)
            u2 = np.random.uniform(-1, 1)
            r_new = r * np.sqrt(1 + np.random.uniform(0.1, 0.3))
            x = r_new * np.sqrt(1 - u2**2) * np.cos(u1)
            y = r_new * np.sqrt(1 - u2**2) * np.sin(u1)
            z = r_new * u2
            current = np.vstack([current, [x, y, z]])
        return current
    
    # size > 55: start with 55 and add
    base = _generate_mackay_icosahedron(55)
    current = base.copy()
    r = 1.05
    for _ in range(size - 55):
        u1 = np.random.uniform(0, 2 * np.pi)
        u2 = np.random.uniform(-1, 1)
        r_new = r * np.sqrt(1 + np.random.uniform(0.1, 0.3))
        x = r_new * np.sqrt(1 - u2**2) * np.cos(u1)
        y = r_new * np.sqrt(1 - u2**2) * np.sin(u1)
        z = r_new * u2
        current = np.vstack([current, [x, y, z]])
    return current

def _generate_random_compact(size, rng):
    """Generate a random compact seed."""
    radius = 0.8 * size ** (1/3) + 1.0
    points = []
    for _ in range(size):
        found = False
        for _ in range(1000):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
                found = True
                break
        if not found:
            p = rng.uniform(-radius*1.2, radius*1.2, 3)
            points.append(p)
    return np.array(points)

def _generate_fcc_cluster(size):
    """Generate an FCC-like cluster seed."""
    points = []
    # FCC lattice vectors
    a = 1.1  # lattice constant
    # Generate points in a sphere
    max_coord = int(np.ceil((size / (4/np.pi)) ** (1/3))) + 1
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            for z in range(-max_coord, max_coord+1):
                # FCC positions
                pos1 = np.array([x, y, z]) * a
                pos2 = np.array([x + 0.5, y + 0.5, z]) * a
                pos3 = np.array([x + 0.5, y, z + 0.5]) * a
                pos4 = np.array([x, y + 0.5, z + 0.5]) * a
                for pos in [pos1, pos2, pos3, pos4]:
                    if len(points) < size:
                        if np.linalg.norm(pos) < 2.5 * size**(1/3):
                            # Check minimum distance
                            if all(np.linalg.norm(pos - q) >= 0.5 for q in points):
                                points.append(pos)
    if len(points) < size:
        # Pad with random
        for _ in range(size - len(points)):
            p = np.random.randn(3) * 0.5
            points.append(p)
    return np.array(points[:size])

def _generate_seed(size, rng, seed_type='auto'):
    """Generate diverse seeds for the given size."""
    seeds = []
    
    if seed_type in ['auto', 'mackay']:
        try:
            seed = _generate_mackay_icosahedron(size)
            if seed.shape[0] == size:
                seeds.append(seed)
        except:
            pass
    
    if seed_type in ['auto', 'random']:
        seed = _generate_random_compact(size, rng)
        seeds.append(seed)
    
    if seed_type in ['auto', 'fcc']:
        try:
            seed = _generate_fcc_cluster(size)
            if seed.shape[0] == size:
                seeds.append(seed)
        except:
            pass
    
    return seeds

def _basin_hopping(positions, rng, time_limit, initial_temp=0.15, initial_step=0.25):
    """Adaptive basin hopping with energy-based acceptance."""
    current_pos = positions.copy()
    current_energy, current_relaxed = relax(current_pos, POTENTIAL)
    best_energy = current_energy
    best_pos = current_relaxed.copy()
    
    temperature = initial_temp
    step_size = initial_step
    consecutive_rejects = 0
    
    while time.time() < time_limit:
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
            consecutive_rejects = 0
            # Reduce step size slightly on improvement
            step_size *= 0.98
        else:
            delta_e = new_energy - current_energy
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current_pos = new_relaxed.copy()
                current_energy = new_energy
                consecutive_rejects += 1
            else:
                consecutive_rejects += 1
        
        # Adaptive temperature and step size
        if consecutive_rejects > 20:
            temperature *= 1.2
            step_size *= 1.1
            consecutive_rejects = 0
        else:
            temperature *= 0.97
            step_size *= 0.995
        
        temperature = max(temperature, 0.01)
        step_size = min(step_size, 0.5)
    
    return best_pos, best_energy

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 285.0  # Safety margin

    size = instance["n"]
    start_time = time.time()
    
    # Time budget: distribute among instances
    # 6 instances, total 285s -> ~47s each
    instance_deadline = min(start_time + 45.0, _GLOBAL_DEADLINE - 3.0)
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate multiple diverse seeds
    all_seeds = []
    
    # Mackay icosahedron-based
    try:
        mackay_seed = _generate_mackay_icosahedron(size)
        if mackay_seed.shape[0] == size:
            all_seeds.append(mackay_seed)
    except:
        pass
    
    # Random compact (multiple)
    for i in range(3):
        rng_sub = np.random.default_rng(42 + size + i * 1000)
        all_seeds.append(_generate_random_compact(size, rng_sub))
    
    # FCC-like
    try:
        fcc_seed = _generate_fcc_cluster(size)
        if fcc_seed.shape[0] == size:
            all_seeds.append(fcc_seed)
    except:
        pass
    
    # Perturbed Mackay
    if len(all_seeds) > 0:
        for i in range(2):
            perturbed = all_seeds[0].copy()
            perturbed += rng.normal(0, 0.1, size=(size, 3))
            all_seeds.append(perturbed)
    
    # Relax each seed and keep the best
    for seed in all_seeds:
        if time.time() > instance_deadline - 10:
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
        best_positions = np.random.randn(size, 3) * 0.3
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass
    
    # Basin hopping from the best seed
    remaining_time = instance_deadline - time.time()
    if remaining_time > 5:
        bh_pos, bh_energy = _basin_hopping(best_positions, rng, time.time() + remaining_time - 2)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    return best_positions
