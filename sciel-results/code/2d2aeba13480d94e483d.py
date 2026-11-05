# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit to be safe

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_gradient(x, rho):
    """
    Wrapper for morse_energy_gradient that reshapes input/output for scipy.
    x: (n*3,) array
    Returns: (energy, gradient) where gradient is (n*3,)
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def _generate_random_cluster(n, seed, radius_factor=1.5):
    """Generate a random cluster with minimum distance constraints."""
    rng = np.random.default_rng(seed)
    radius = radius_factor * (n ** (1.0/3.0))
    
    points = []
    max_attempts = 10000
    attempts = 0
    min_dist = 0.51
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
            
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _generate_crystalline_template(n, seed, lattice_type='fcc'):
    """Generate a crystalline template."""
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    lattice_points = []
    
    r = int(math.ceil(radius)) + 1
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                if lattice_type == 'fcc':
                    basis = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
                elif lattice_type == 'bcc':
                    basis = [(0,0,0), (0.5,0.5,0.5)]
                else:
                    basis = [(0,0,0), (0, 1/math.sqrt(3), 0.5)]
                
                for dx, dy, dz in basis:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        jitter = rng.uniform(-0.01, 0.01, (n, 3))
        return np.array(lattice_points[:n]) + jitter
    else:
        remaining = n - len(lattice_points)
        for _ in range(remaining):
            p = rng.uniform(-radius, radius, 3)
            lattice_points.append(p)
        return np.array(lattice_points)

def _generate_icosahedral_cluster(n, seed):
    """Generate an icosahedral cluster template."""
    rng = np.random.default_rng(seed)
    radius = 1.2 * (n ** (1.0/3.0))
    
    # Icosahedral shells
    shells = []
    # Center
    shells.append(np.array([[0, 0, 0]]))
    
    # First shell: 12 vertices of icosahedron
    phi = (1 + math.sqrt(5)) / 2
    ico_vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]
    for v in ico_vertices:
        v = np.array(v) / np.linalg.norm(v)
        shells.append(v.reshape(1, 3))
    
    # Second shell: 30 points (edges of icosahedron)
    # Simplified: use points on edges
    edge_points = []
    for i in range(12):
        for j in range(i+1, 12):
            v1 = np.array(ico_vertices[i])
            v2 = np.array(ico_vertices[j])
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            d = np.linalg.norm(v1 - v2)
            if d < 1.5:  # Adjacent vertices
                for t in [0.3, 0.7]:
                    p = (1-t)*v1 + t*v2
                    p = p / np.linalg.norm(p)
                    edge_points.append(p)
    
    for p in edge_points:
        shells.append(p.reshape(1, 3))
    
    # Collect all points
    all_points = np.vstack(shells)
    
    # If we need more, add random points in a shell
    if len(all_points) < n:
        remaining = n - len(all_points)
        for _ in range(remaining):
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * 2.0
            all_points = np.vstack([all_points, p.reshape(1, 3)])
    
    # Scale to appropriate radius
    scale = radius / np.max(np.linalg.norm(all_points, axis=1))
    all_points = all_points * scale
    
    # Add small jitter
    jitter = rng.uniform(-0.01, 0.01, (n, 3))
    return all_points[:n] + jitter

def _min_distance_check(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(positions)
    for j in range(n):
        for k in range(j+1, n):
            d = np.linalg.norm(positions[j] - positions[k])
            if d < min_dist:
                return False
    return True

def _basin_hopping_step(positions, rng, step_size, rho):
    """
    Perform a single basin-hopping step:
    1. Perturb positions by random noise
    2. Relax the perturbed structure
    3. Return the relaxed structure and its energy
    """
    n = len(positions)
    perturbed = positions + rng.normal(0, step_size, positions.shape)
    
    try:
        energy, relaxed = relax(perturbed, 'morse', rho)
        
        # Check validity: minimum distance constraint
        if not _min_distance_check(relaxed):
            return float('inf'), positions
        
        return energy, relaxed
    except Exception:
        return float('inf'), positions

def _high_precision_polish(positions, rho):
    """
    Final high-precision polish pass using scipy's L-BFGS-B with tighter tolerances.
    """
    n = len(positions)
    x0 = positions.flatten()
    
    try:
        result = minimize(
            _morse_energy_gradient,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success:
            polished = result.x.reshape(n, 3)
            if _min_distance_check(polished):
                energy = _morse_energy_gradient(result.x, rho)[0]
                return energy, polished
    except Exception:
        pass
    
    return float('inf'), positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy: Multi-start basin-hopping with diverse initializations
    # including icosahedral templates, plus a final high-precision polish pass
    
    # Generate diverse initializations
    initializations = []
    
    # Crystalline templates
    for lattice in ['fcc', 'bcc']:
        for seed in [1, 2, 3]:
            try:
                initializations.append(_generate_crystalline_template(n, seed, lattice))
            except:
                pass
                
    # Icosahedral templates
    for seed in [1, 2, 3]:
        try:
            initializations.append(_generate_icosahedral_cluster(n, seed))
        except:
            pass
            
    # Random clusters with different densities
    for seed in range(6):
        try:
            initializations.append(_generate_random_cluster(n, seed, radius_factor=1.2 + 0.2*seed))
        except:
            pass
            
    # Relax each initialization and keep the best
    for init_pos in initializations:
        if _get_remaining_time() < 15.0:
            break
            
        try:
            energy, positions = relax(init_pos, potential, rho)
            
            if _min_distance_check(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback
        best_positions = _generate_random_cluster(n, 999, radius_factor=2.0)
        best_energy = 1e10
    
    # Basin-Hopping phase
    # Adaptive step size: start large, decrease if no improvement
    rng = np.random.default_rng(42)
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    step_size = 0.5  # Initial step size
    min_step_size = 0.05
    max_consecutive_failures = 10
    consecutive_failures = 0
    total_bh_steps = 0
    max_bh_steps = 600
    
    while _get_remaining_time() > 25.0 and total_bh_steps < max_bh_steps:
        # Perform a BH move
        new_energy, new_positions = _basin_hopping_step(
            current_positions, rng, step_size, rho
        )
        
        total_bh_steps += 1
        
        if new_energy < current_energy:
            # Accept the move
            current_energy = new_energy
            current_positions = new_positions.copy()
            consecutive_failures = 0
            
            # Update best
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_positions.copy()
                
            # Occasionally increase step size to explore
            if rng.random() < 0.1:
                step_size = min(step_size * 1.2, 0.8)
        else:
            # Reject the move
            consecutive_failures += 1
            
            # Decrease step size if stuck
            if consecutive_failures > max_consecutive_failures:
                step_size = max(step_size * 0.8, min_step_size)
                consecutive_failures = 0
                max_consecutive_failures = max(max_consecutive_failures // 2, 5)
    
    # Final high-precision polish pass
    if _get_remaining_time() > 5.0 and best_positions is not None:
        polished_energy, polished_positions = _high_precision_polish(best_positions, rho)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_positions
    
    return best_positions
# EVOLVE-BLOCK-END
