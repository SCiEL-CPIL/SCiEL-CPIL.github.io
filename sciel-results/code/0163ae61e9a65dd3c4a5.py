import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

POTENTIAL = 'dzugutov'
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
    
    if size == 13:
        center = np.zeros(3)
        return np.vstack([center, ico_verts])
    
    if size == 55:
        center = np.zeros(3)
        r2 = 1.0
        shell2 = ico_verts * r2
        
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
        
        edge_mids = []
        for i, j in edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2
            edge_mids.append(mid)
        edge_mids = np.array(edge_mids)
        
        dod_verts = _get_dodecahedron_vertices()
        r_edge = np.sqrt(r2**2 - (min_dist/2)**2)
        dod_norms = np.linalg.norm(dod_verts, axis=1, keepdims=True)
        dod_scaled = dod_verts / dod_norms * r_edge
        
        shell3 = np.vstack([edge_mids, dod_scaled])
        
        return np.vstack([center, shell2, shell3])
    
    # For other sizes, grow from 13 or 55
    if size <= 13:
        base = _generate_mackay_icosahedron(13)
        return base[:size]
    
    if size <= 55:
        base = _generate_mackay_icosahedron(13)
        current = base.copy()
        r = 1.0
        for _ in range(size - 13):
            u1 = np.random.uniform(0, 2 * np.pi)
            u2 = np.random.uniform(-1, 1)
            r_new = r * np.sqrt(1 + np.random.uniform(0.1, 0.3))
            x = r_new * np.sqrt(1 - u2**2) * np.cos(u1)
            y = r_new * np.sqrt(1 - u2**2) * np.sin(u1)
            z = r_new * u2
            current = np.vstack([current, [x, y, z]])
        return current
    
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
    a = 1.1
    max_coord = int(np.ceil((size / (4/np.pi)) ** (1/3))) + 1
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            for z in range(-max_coord, max_coord+1):
                pos1 = np.array([x, y, z]) * a
                pos2 = np.array([x + 0.5, y + 0.5, z]) * a
                pos3 = np.array([x + 0.5, y, z + 0.5]) * a
                pos4 = np.array([x, y + 0.5, z + 0.5]) * a
                for pos in [pos1, pos2, pos3, pos4]:
                    if len(points) < size:
                        if np.linalg.norm(pos) < 2.5 * size**(1/3):
                            if all(np.linalg.norm(pos - q) >= 0.5 for q in points):
                                points.append(pos)
    if len(points) < size:
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

def _lbfgs_optimize(positions, time_limit):
    """
    Optimize positions using L-BFGS-B with the Dzugutov gradient.
    """
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        try:
            result = dzugutov_energy_gradient(pos)
            if isinstance(result, tuple):
                energy, grad = result
            else:
                energy = 0.0
                grad = result
            
            grad_flat = grad.flatten()
            return energy, grad_flat
        except Exception as e:
            return 1e10, np.zeros_like(x)

    x0 = positions.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 50, 'maxfun': 50, 'disp': False, 'maxls': 10})
        if res.success:
            return res.x.reshape(-1, 3), res.fun
        else:
            return res.x.reshape(-1, 3), res.fun
    except:
        return positions, 0.0

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

def _repulsive_annealing(positions, rng, time_limit, initial_temp=0.3):
    """
    Repulsive Annealing: uses a continuous auxiliary potential to guide the system
    out of local minima. This is distinct from discrete basin hopping.
    """
    current_pos = positions.copy()
    current_energy, current_relaxed = relax(current_pos, POTENTIAL)
    best_energy = current_energy
    best_pos = current_relaxed.copy()
    
    temperature = initial_temp
    n_atoms = len(positions)
    
    while time.time() < time_limit:
        # Add repulsive noise that scales with temperature
        noise_scale = temperature * 0.1
        perturbation = rng.normal(0, noise_scale, size=(n_atoms, 3))
        new_pos = current_pos + perturbation
        
        try:
            new_energy, new_relaxed = relax(new_pos, POTENTIAL)
        except:
            continue
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_relaxed.copy()
            current_pos = new_relaxed.copy()
            temperature *= 0.95
        else:
            delta_e = new_energy - current_energy
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current_pos = new_relaxed.copy()
                current_energy = new_energy
                temperature *= 0.98
            else:
                temperature *= 1.05
        
        temperature = max(temperature, 0.01)
        temperature = min(temperature, 1.0)
    
    return best_pos, best_energy

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 285.0

    size = instance["n"]
    start_time = time.time()
    
    instance_deadline = min(start_time + 45.0, _GLOBAL_DEADLINE - 3.0)
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    all_seeds = []
    
    try:
        mackay_seed = _generate_mackay_icosahedron(size)
        if mackay_seed.shape[0] == size:
            all_seeds.append(mackay_seed)
    except:
        pass
    
    for i in range(3):
        rng_sub = np.random.default_rng(42 + size + i * 1000)
        all_seeds.append(_generate_random_compact(size, rng_sub))
    
    try:
        fcc_seed = _generate_fcc_cluster(size)
        if fcc_seed.shape[0] == size:
            all_seeds.append(fcc_seed)
    except:
        pass
    
    if len(all_seeds) > 0:
        for i in range(2):
            perturbed = all_seeds[0].copy()
            perturbed += rng.normal(0, 0.1, size=(size, 3))
            all_seeds.append(perturbed)
    
    # Phase 1: Relax all seeds and keep the best
    for seed in all_seeds:
        if time.time() > instance_deadline - 15:
            break
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            continue
    
    if best_positions is None:
        best_positions = np.random.randn(size, 3) * 0.3
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass

    # Phase 2: L-BFGS optimization on the best seed
    remaining_time = instance_deadline - time.time()
    if remaining_time > 10:
        lbfgs_pos, lbfgs_energy = _lbfgs_optimize(best_positions, time.time() + remaining_time - 5)
        try:
            verified_energy, verified_pos = relax(lbfgs_pos, POTENTIAL)
            if verified_energy < best_energy:
                best_energy = verified_energy
                best_positions = verified_pos
        except:
            pass

    # Phase 3: Repulsive Annealing from the best seed
    remaining_time = instance_deadline - time.time()
    if remaining_time > 5:
        ra_pos, ra_energy = _repulsive_annealing(best_positions, rng, time.time() + remaining_time - 2)
        if ra_energy < best_energy:
            best_energy = ra_energy
            best_positions = ra_pos
    
    # Phase 4: Basin hopping from the best seed if time remains
    remaining_time = instance_deadline - time.time()
    if remaining_time > 3:
        bh_pos, bh_energy = _basin_hopping(best_positions, rng, time.time() + remaining_time - 1)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    return best_positions
