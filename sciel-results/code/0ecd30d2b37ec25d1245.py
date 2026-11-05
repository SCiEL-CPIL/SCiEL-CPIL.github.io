# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit to be safe

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_flat(x, rho):
    """
    Compute Morse energy for a flat array of coordinates.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, _ = morse_energy_gradient(positions, rho)
        return energy
    except Exception:
        return 1e10

def _morse_energy_grad_flat(x, rho):
    """
    Compute Morse energy and gradient for a flat array.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def _min_distance(positions):
    """
    Compute minimum pairwise distance.
    """
    n = len(positions)
    min_d = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    return min_d

def _generate_fcc_cluster(n, seed=None):
    """
    Generate an FCC-like cluster template.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    radius = 1.5 * (n ** (1.0/3.0))
    lattice_points = []
    
    # FCC lattice generation
    for i in range(-int(radius), int(radius)+1):
        for j in range(-int(radius), int(radius)+1):
            for k in range(-int(radius), int(radius)+1):
                # FCC positions in a cubic unit cell of side 1
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(lattice_points[:n]) + jitter
    else:
        remaining = n - len(lattice_points)
        for _ in range(remaining):
            p = rng.uniform(-radius, radius, 3)
            lattice_points.append(p)
        return np.array(lattice_points)

def _generate_bcc_cluster(n, seed=None):
    """
    Generate a BCC-like cluster template.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    radius = 1.5 * (n ** (1.0/3.0))
    lattice_points = []
    
    for i in range(-int(radius), int(radius)+1):
        for j in range(-int(radius), int(radius)+1):
            for k in range(-int(radius), int(radius)+1):
                # BCC positions: (0,0,0) and (0.5,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(lattice_points[:n]) + jitter
    else:
        remaining = n - len(lattice_points)
        for _ in range(remaining):
            p = rng.uniform(-radius, radius, 3)
            lattice_points.append(p)
        return np.array(lattice_points)

def _generate_icosahedral_cluster(n, seed=None):
    """
    Generate an icosahedral cluster template.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    # Start with icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2
    ico_vertices = np.array([
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_vertices /= np.linalg.norm(ico_vertices[0])
    
    # Add center
    points = [np.zeros(3)]
    points.extend(ico_vertices)
    
    # If we need more, add a second shell (subdivided)
    if n > 13:
        # Add points on edges
        edges = [
            (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
            (0,8), (0,9), (0,10), (0,11),
            (1,2), (1,4), (1,5), (1,8), (1,9),
            (2,3), (2,4), (2,6), (2,10), (2,11),
            (3,4), (3,5), (3,6), (3,7),
            (4,5), (4,8), (4,9),
            (5,6), (5,8), (5,9),
            (6,7), (6,10), (6,11),
            (7,8), (7,9),
            (8,9), (8,10), (8,11),
            (10,11)
        ]
        for i, j in edges:
            mid = (ico_vertices[i] + ico_vertices[j]) / 2
            mid *= 1.1  # Push out slightly
            points.append(mid)
        
        # If still need more, add random points near surface
        while len(points) < n:
            base = ico_vertices[np.random.randint(0, 12)]
            p = base * 1.2 + rng.uniform(-0.3, 0.3, 3)
            points.append(p)
    
    points = np.array(points[:n])
    # Scale to have reasonable spacing
    scale = 1.0 / (1.0 + 0.1 * (n ** (1.0/3.0)))
    points *= scale
    jitter = rng.uniform(-0.05, 0.05, (n, 3))
    return points + jitter

def _generate_random_cluster(n, seed=None, min_dist=0.6):
    """
    Generate a random cluster with minimum distance constraint.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _refine_structure(positions, potential, rho, max_iter=50):
    """
    Refine a structure using L-BFGS with the provided gradient.
    """
    n = len(positions)
    x0 = positions.flatten()
    
    try:
        result = minimize(
            _morse_energy_flat,
            x0,
            args=(rho,),
            jac=lambda x: _morse_energy_grad_flat(x, rho)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 10000, 'gtol': 1e-8}
        )
        return result.x.reshape(n, 3), result.fun
    except Exception:
        return positions, _morse_energy_flat(x0, rho)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate diverse initial templates
    starts = []
    
    # FCC template
    try:
        starts.append(_generate_fcc_cluster(n, seed=42))
    except Exception:
        pass
        
    # BCC template
    try:
        starts.append(_generate_bcc_cluster(n, seed=42))
    except Exception:
        pass
        
    # Icosahedral template
    try:
        starts.append(_generate_icosahedral_cluster(n, seed=42))
    except Exception:
        pass
        
    # Random starts
    for seed in [1, 2, 3, 4, 5]:
        try:
            starts.append(_generate_random_cluster(n, seed=seed))
        except Exception:
            pass
            
    if not starts:
        starts.append(_generate_random_cluster(n, seed=0))
        
    # 2. Relax all initial starts
    relaxed_starts = []
    for i, start in enumerate(starts):
        if _get_remaining_time() < 15.0:
            break
        try:
            energy, positions = relax(start, potential, rho)
            min_dist = _min_distance(positions)
            if min_dist >= 0.5:
                relaxed_starts.append((energy, positions))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
            
    if not relaxed_starts:
        if best_positions is None:
            best_positions = starts[0]
            try:
                e, _ = morse_energy_gradient(best_positions, rho)
                best_energy = e
            except:
                best_energy = 1e10
                
    # 3. Basin Hopping from the best relaxed start
    if best_positions is not None and _get_remaining_time() > 20.0:
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        # Adaptive step size
        step_size = 0.4
        temp = 1.0
        cooling_rate = 0.97
        max_bh_steps = 150
        no_improve_count = 0
        best_in_bh = current_energy
        
        for bh_iter in range(max_bh_steps):
            if _get_remaining_time() < 10.0:
                break
                
            # Perturb
            noise = np.random.normal(0, step_size, current_positions.shape)
            new_positions = current_positions + noise
            
            # Relax
            try:
                new_energy, new_relaxed = relax(new_positions, potential, rho)
            except Exception:
                new_energy = float('inf')
                new_relaxed = current_positions
                
            min_dist = _min_distance(new_relaxed)
            if min_dist < 0.5:
                new_energy = float('inf')
                
            # Metropolis acceptance
            if new_energy < current_energy:
                current_energy = new_energy
                current_positions = new_relaxed
                no_improve_count = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_positions
                if current_energy < best_in_bh:
                    best_in_bh = current_energy
            else:
                delta_e = new_energy - current_energy
                if temp > 1e-8:
                    accept_prob = math.exp(-delta_e / temp)
                    if np.random.rand() < accept_prob:
                        current_energy = new_energy
                        current_positions = new_relaxed
                no_improve_count += 1
                
            # Cooling and step size adaptation
            temp *= cooling_rate
            if no_improve_count > 10:
                step_size *= 1.1  # Increase step size to escape local minima
                no_improve_count = 0
            elif new_energy < current_energy:
                step_size *= 0.95  # Decrease step size on success
        
        # 4. If time permits, try BH from second best start
        if len(relaxed_starts) > 1 and _get_remaining_time() > 25.0:
            relaxed_starts.sort(key=lambda x: x[0])
            for energy, positions in relaxed_starts[1:3]:
                if _get_remaining_time() < 15.0:
                    break
                    
                current_energy = energy
                current_positions = positions.copy()
                step_size = 0.4
                temp = 1.0
                no_improve_count = 0
                
                for bh_iter in range(50):
                    if _get_remaining_time() < 10.0:
                        break
                        
                    noise = np.random.normal(0, step_size, current_positions.shape)
                    new_positions = current_positions + noise
                    
                    try:
                        new_energy, new_relaxed = relax(new_positions, potential, rho)
                    except Exception:
                        new_energy = float('inf')
                        new_relaxed = current_positions
                        
                    min_dist = _min_distance(new_relaxed)
                    if min_dist < 0.5:
                        new_energy = float('inf')
                        
                    if new_energy < current_energy:
                        current_energy = new_energy
                        current_positions = new_relaxed
                        no_improve_count = 0
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_positions = current_positions
                    else:
                        delta_e = new_energy - current_energy
                        if temp > 1e-8:
                            accept_prob = math.exp(-delta_e / temp)
                            if np.random.rand() < accept_prob:
                                current_energy = new_energy
                                current_positions = new_relaxed
                        no_improve_count += 1
                        
                    temp *= cooling_rate
                    if no_improve_count > 10:
                        step_size *= 1.1
                        no_improve_count = 0
                    elif new_energy < current_energy:
                        step_size *= 0.95

    # 5. Final polish with L-BFGS
    if best_positions is not None and _get_remaining_time() > 5.0:
        try:
            polished, polished_energy = _refine_structure(best_positions, potential, rho, max_iter=30)
            min_dist = _min_distance(polished)
            if min_dist >= 0.5 and polished_energy < best_energy:
                best_positions = polished
                best_energy = polished_energy
        except Exception:
            pass

    if best_positions is None:
        best_positions = starts[0]
        
    return best_positions
# EVOLVE-BLOCK-END
