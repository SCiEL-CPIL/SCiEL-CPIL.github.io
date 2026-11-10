"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_TOTAL_DEADLINE = 300.0  # 300s total budget


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _TOTAL_DEADLINE - (time.time() - _START_TIME)


def _generate_icosahedral(n, rng):
    """Generate an icosahedral-like cluster for polytetrahedral packing."""
    phi = (1 + np.sqrt(5)) / 2
    # Standard 12 icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Add vertices at shells with increasing radius
    shell_radii = [1.0, 1.8, 2.6, 3.4, 4.2, 5.0]
    idx = 0
    for shell_r in shell_radii:
        if len(points) >= n:
            break
        for _ in range(12):
            if len(points) >= n:
                break
            v = ico_verts[idx % 12]
            # Small random offset for diversity
            noise = rng.normal(0, 0.05, 3)
            p = shell_r * v + noise
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
            idx += 1
    
    # If still need more, add in a denser shell
    while len(points) < n:
        r = 1.0 + 0.5 * len(points) / 12.0
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        xy = np.sqrt(max(0, 1 - z*z))
        p = r * np.array([xy * np.cos(theta), xy * np.sin(theta), z])
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            # Try with small perturbation
            p += rng.normal(0, 0.1, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    return np.array(points[:n])


def _generate_fcc(n, rng):
    """Generate an FCC-like cluster."""
    a = 2.2
    points = []
    max_coord = int((n / 4.0) ** (1.0/3.0) * a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for basis in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    if len(points) >= n:
                        break
                    p = np.array([i*a, j*a, k*a]) + np.array(basis, dtype=float)
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_random(n, rng):
    """Generate a random compact cluster."""
    radius = 1.1 * n ** (1.0/3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 for q in points):
            points.append(p)
        attempts += 1
    
    # Fallback: add points with decreasing spacing requirement
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            # Try closer
            for _ in range(10):
                p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
                    break
    
    return np.array(points[:n])


def _generate_bcc(n, rng):
    """Generate a BCC-like cluster."""
    a = 2.2
    points = []
    max_coord = int((n / 2.0) ** (1.0/3.0) * a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if len(points) >= n:
                    break
                # Corner
                p = np.array([i*a, j*a, k*a], dtype=float)
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
                # Body center
                if len(points) < n:
                    p2 = np.array([i*a, j*a, k*a], dtype=float) + np.array([a/2, a/2, a/2])
                    if all(np.linalg.norm(p2 - q) >= 0.5 for q in points):
                        points.append(p2)
    
    while len(points) < n:
        p = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _local_minimize(positions, potential, max_steps=1000, step_size=0.01):
    """Simple gradient-based local minimization using the provided gradient function."""
    pos = positions.copy()
    try:
        energy, grad = dzugutov_energy_gradient(pos)
    except Exception:
        return positions, float('inf')
    
    for step in range(max_steps):
        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            break
        
        # Take a step
        new_pos = pos - step_size * grad / max(grad_norm, 1e-8)
        
        # Check minimum distance constraint
        valid = True
        for i in range(len(new_pos)):
            for j in range(i+1, len(new_pos)):
                if np.linalg.norm(new_pos[i] - new_pos[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            step_size *= 0.5
            continue
        
        try:
            new_energy, new_grad = dzugutov_energy_gradient(new_pos)
        except Exception:
            step_size *= 0.5
            continue
        
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            grad = new_grad
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size *= 0.7
        
        if step_size < 1e-8:
            break
    
    return pos, energy


def _basin_hopping(positions, potential, rng, max_iterations=100, initial_delta=0.3):
    """Basin hopping with adaptive step size."""
    current_energy, current_pos = relax(positions, potential)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    delta = initial_delta
    for it in range(max_iterations):
        # Random perturbation
        delta_vec = rng.normal(0, delta, positions.shape)
        perturbed = current_pos + delta_vec
        
        # Check minimum distance
        valid = True
        for i in range(len(perturbed)):
            for j in range(i+1, len(perturbed)):
                if np.linalg.norm(perturbed[i] - perturbed[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Try smaller perturbation
            delta *= 0.9
            continue
        
        try:
            new_energy, new_pos = relax(perturbed, potential)
        except Exception:
            delta *= 0.9
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            delta = min(delta * 1.05, 0.5)
        else:
            # Accept with probability
            temp = 0.1
            if rng.random() < np.exp(-(new_energy - current_energy) / temp):
                current_energy = new_energy
                current_pos = new_pos
            else:
                delta *= 0.95
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
        
        # Check time
        if _get_time_remaining() < 5:
            break
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy structures using basin hopping with diverse seeds."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse initial structures
    generators = [
        lambda: _generate_icosahedral(size, rng),
        lambda: _generate_fcc(size, rng),
        lambda: _generate_random(size, rng),
        lambda: _generate_bcc(size, rng),
        lambda: _generate_random(size, rng),
        lambda: _generate_icosahedral(size, rng),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Allocate time per generator based on remaining time
    n_gens = len(geners) if False else len(generators)
    time_per_gen = max(10, _get_time_remaining() / (n_gens + 2))
    
    for gen in generators:
        if _get_time_remaining() < 15:
            break
        
        try:
            positions = gen()
            if positions.shape[0] != size:
                continue
            
            # First, do a local minimization
            try:
                _, pos_min = relax(positions, potential)
            except Exception:
                pos_min = positions
            
            # Then basin hopping from the minimized structure
            max_iter = int(min(80, max(20, _get_time_remaining() / 2.0)))
            energy, pos = _basin_hopping(pos_min, potential, rng, 
                                          max_iterations=max_iter, 
                                          initial_delta=0.25)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            continue
    
    # If we have time, do more basin hopping from the best
    if best_positions is not None and _get_time_remaining() > 30:
        try:
            max_iter = int(min(100, max(30, _get_time_remaining() / 1.5)))
            energy, pos = _basin_hopping(best_positions, potential, rng,
                                         max_iterations=max_iter,
                                         initial_delta=0.15)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            pass
    
    # Final relaxation
    if best_positions is not None:
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback to random
        best_positions = _generate_random(size, rng)
        try:
            _, best_positions = relax(best_positions, potential)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
