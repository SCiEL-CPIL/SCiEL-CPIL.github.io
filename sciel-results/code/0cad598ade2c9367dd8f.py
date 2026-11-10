"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_icosahedral(n, rng):
    """Generate an icosahedral-like cluster for polytetrahedral packing."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0]
    ])
    # Actually let's use the standard 12 icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Add vertices at various shells
    shell_radii = [1.0, 2.0, 2.8, 3.5, 4.2]
    shell_counts = [12, 30, 60, 120, 180]
    
    idx = 0
    for shell_r, shell_n in zip(shell_radii, shell_counts):
        for _ in range(shell_n):
            if len(points) >= n:
                break
            v = ico_verts[idx % 12]
            # Add some randomness
            noise = rng.normal(0, 0.1, 3)
            p = shell_r * v + noise
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
            idx += 1
        if len(points) >= n:
            break
    
    # If we still need more, add random points
    while len(points) < n:
        p = rng.uniform(-5, 5, 3)
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
                    p = np.array([i*a, j*a, k*a]) + np.array(basis)
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    while len(points) < n:
        p = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _generate_random(n, rng):
    """Generate a random compact cluster."""
    radius = 1.1 * n ** (1.0/3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
    return np.array(points)


def _basin_hopping(positions, potential, parameter, rng, max_iterations=50, delta_e=0.1):
    """Simple basin hopping algorithm."""
    current_energy, current_pos = relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for _ in range(max_iterations):
        # Random perturbation
        delta = rng.normal(0, delta_e, positions.shape)
        perturbed = current_pos + delta
        
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
            continue
        
        new_energy, new_pos = relax(perturbed, potential, parameter)
        
        # Accept if lower energy or with Boltzmann probability
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
        elif rng.random() < np.exp(-(new_energy - current_energy) / 0.1):
            current_energy = new_energy
            current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy structures using basin hopping with diverse seeds."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial structures
    generators = [
        lambda: _generate_icosahedral(size, rng),
        lambda: _generate_fcc(size, rng),
        lambda: _generate_random(size, rng),
        lambda: _generate_random(size, rng),
        lambda: _generate_random(size, rng),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Try each generator with basin hopping
    for gen in generators:
        if _get_time_remaining() < 30:
            break
        
        try:
            positions = gen()
            if positions.shape[0] != size:
                continue
            
            energy, pos = _basin_hopping(positions, potential, parameter, rng, 
                                         max_iterations=30, delta_e=0.15)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            continue
    
    # If we have time, do more basin hopping from the best
    if best_positions is not None and _get_time_remaining() > 60:
        try:
            energy, pos = _basin_hopping(best_positions, potential, parameter, rng,
                                         max_iterations=50, delta_e=0.1)
            if energy < best_energy:
                best_energy = energy
                best_positions = pos
        except Exception:
            pass
    
    # Final relaxation
    if best_positions is not None:
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback to random
        best_positions = _generate_random(size, rng)
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
