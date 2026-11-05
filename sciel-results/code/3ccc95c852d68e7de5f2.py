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
        min_dist = np.inf
        for j in range(n):
            for k in range(j+1, n):
                d = np.linalg.norm(relaxed[j] - relaxed[k])
                if d < min_dist:
                    min_dist = d
                if d < 0.5:
                    min_dist = 0.0
                    break
            if min_dist < 0.5:
                break
        
        if min_dist >= 0.5:
            return energy, relaxed
        else:
            return float('inf'), positions
    except Exception:
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
    
    # Strategy: Basin-Hopping with adaptive step size
    # Start from multiple diverse initializations, then perform BH moves
    
    # Generate diverse initializations
    initializations = []
    
    # Crystalline templates
    for lattice in ['fcc', 'bcc']:
        for seed in [1, 2, 3]:
            try:
                initializations.append(_generate_crystalline_template(n, seed, lattice))
            except:
                pass
                
    # Random clusters with different densities
    for seed in range(5):
        try:
            initializations.append(_generate_random_cluster(n, seed, radius_factor=1.2 + 0.2*seed))
        except:
            pass
            
    # Relax each initialization and keep the best
    for init_pos in initializations:
        if _get_remaining_time() < 10.0:
            break
            
        try:
            energy, positions = relax(init_pos, potential, rho)
            
            min_dist = np.inf
            for j in range(n):
                for k in range(j+1, n):
                    d = np.linalg.norm(positions[j] - positions[k])
                    if d < min_dist:
                        min_dist = d
                    if d < 0.5:
                        min_dist = 0.0
                        break
                if min_dist < 0.5:
                    break
                    
            if min_dist >= 0.5 and energy < best_energy:
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
    max_bh_steps = 500
    
    while _get_remaining_time() > 15.0 and total_bh_steps < max_bh_steps:
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
    
    return best_positions
# EVOLVE-BLOCK-END
