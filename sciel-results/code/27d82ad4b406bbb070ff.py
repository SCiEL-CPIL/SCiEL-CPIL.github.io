"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_overlap(positions, r):
    """
    Compute the total overlap for a given radius r.
    Overlap is sum of squared penetrations for pairs and boundary.
    """
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Distance to boundary
    radii = np.sqrt(np.sum(positions ** 2, axis=1))
    
    # Pair overlaps
    mask = ~np.eye(N, dtype=bool)
    dists_masked = np.where(mask, dists, 0)
    
    pen_pairs = np.maximum(0, 2 * r - dists_masked)
    overlap_pairs = np.sum(pen_pairs ** 2) / 2.0
    
    # Boundary overlaps
    pen_boundary = np.maximum(0, r + radii - 1)
    overlap_boundary = np.sum(pen_boundary ** 2)
    
    return overlap_pairs + overlap_boundary

def _simulated_annealing(r_target, time_limit, seed=0):
    """
    Run Simulated Annealing to minimize overlap for a fixed target radius r_target.
    Returns the best positions found.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize positions randomly inside the disk of radius (1 - r_target)
    positions = np.random.uniform(-1, 1, (N, 2))
    radii = np.sqrt(np.sum(positions ** 2, axis=1))
    max_rad = 1.0 - r_target
    if max_rad <= 0:
         return positions, float('inf')
    positions *= max_rad / np.maximum(radii, 1e-6)[:, np.newaxis]
    
    current_energy = _compute_overlap(positions, r_target)
    best_energy = current_energy
    best_positions = positions.copy()
    
    # SA Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    alpha = 0.9995 # Cooling rate per step
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < time_limit and T > T_min:
        i = np.random.randint(N)
        
        move_mag = 0.05 * r_target
        move = np.random.normal(0, move_mag, 2)
        
        new_pos = positions[i].copy()
        new_pos += move
        
        old_pos = positions[i].copy()
        positions[i] = new_pos
        new_energy = _compute_overlap(positions, r_target)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
        else:
            positions[i] = old_pos
            
        T *= alpha
        step_count += 1
        
    return best_positions, best_energy

def _find_optimal_radius(total_time):
    """
    Iteratively increase radius and run SA.
    """
    start_time = time.time()
    
    # 1. Find a baseline packing
    # Try r=0.1
    base_time = min(5.0, total_time * 0.1)
    r_base = 0.1
    pos_base, e_base = _simulated_annealing(r_base, base_time, seed=1)
    
    if e_base > 1e-3:
        # If even 0.1 fails, something is wrong, but 0.1 should be easy
        # Just return random
        return 0.1, pos_base
        
    # Calculate actual radius of this packing to set a better starting point
    # We can just use r_base as start, or slightly higher
    r_current = r_base + 0.001
    
    best_r = r_base
    best_positions = pos_base
    
    r_step = 0.002
    sa_time_slice = 2.0
    
    while time.time() - start_time < total_time - 5:
        # Run SA for r_current
        seed = int(time.time() * 1000) % 100000
        positions, energy = _simulated_annealing(r_current, sa_time_slice, seed=seed)
        
        if energy < 1e-4:
            # Success
            best_r = r_current
            best_positions = positions
            r_current += r_step
        else:
            # Failure
            # Try one more time with a different seed to be sure
            seed2 = seed + 1
            positions, energy = _simulated_annealing(r_current, sa_time_slice, seed=seed2)
            if energy < 1e-4:
                best_r = r_current
                best_positions = positions
                r_current += r_step
            else:
                # Failed to pack at r_current.
                # We might be close to the limit.
                # We can try to binary search between best_r and r_current if we have time
                # But simple break is safer for time management.
                break
                
    return best_r, best_positions

def construct():
    """Construct positions for N circles in the unit disk."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_r, best_positions = _find_optimal_radius(deadline - start_time)
    
    # Final check and polish
    # The positions from SA are valid for best_r.
    # The evaluator will compute the true score.
    
    return best_positions
# EVOLVE-BLOCK-END
