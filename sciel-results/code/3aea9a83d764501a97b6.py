"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Distance between all pairs
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Square is [-0.5, 0.5]^2
    wall_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r


def _repulsive_force(centers, exponent=2.0, softness=0.01):
    """
    Compute repulsive forces between circles.
    Uses a soft potential to avoid singularities at zero distance.
    Force is proportional to -dU/dx where U = sum_{i<j} 1/(d_ij^exponent)
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1) + softness ** 2)
    np.fill_diagonal(dists, np.inf)
    
    # Force magnitude: d(1/d^p)/dd = -p/d^(p+1)
    # Force direction: normalized difference
    # F_i = sum_j p * (c_i - c_j) / d_ij^(p+2)
    inv_d_p2 = np.zeros_like(dists)
    mask = dists < np.inf
    inv_d_p2[mask] = 1.0 / (dists[mask] ** (exponent + 2.0))
    
    # Multiply by exponent and sum
    forces = exponent * np.sum(diffs * inv_d_p2[:, :, np.newaxis], axis=1)
    
    return forces


def _wall_force(centers, strength=10.0, margin=0.0):
    """
    Force pushing centers away from walls.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    half = 0.5 - margin
    
    for axis in range(2):
        # Distance to left/bottom wall
        d_left = centers[:, axis] + half
        d_right = half - centers[:, axis]
        
        # Strong repulsion when close to wall
        # Use inverse power law
        f_left = strength / (d_left ** 3 + 1e-6)
        f_right = strength / (d_right ** 3 + 1e-6)
        
        forces[:, axis] += f_left - f_right
    
    return forces


def _langevin_step(centers, velocity, dt, temperature, exponent, damping):
    """
    One step of Langevin dynamics.
    """
    n = len(centers)
    
    # Compute total force
    force = _repulsive_force(centers, exponent=exponent) + _wall_force(centers)
    
    # Langevin equation:
    # m * dv/dt = -gamma * v + F + sqrt(2*gamma*kT) * eta
    # v_new = v + dt * (-gamma/m * v + F/m + sqrt(2*gamma*kT/m) * eta)
    # For simplicity, m=1
    eta = np.random.randn(n, 2)
    noise_strength = np.sqrt(2.0 * damping * temperature / dt)
    
    velocity = velocity * (1.0 - damping * dt) + force * dt + noise_strength * eta
    centers = centers + velocity * dt
    
    return centers, velocity


def _anneal_schedule(t, t_start, t_end, n_steps, initial_temp, final_temp):
    """
    Exponential cooling schedule.
    """
    if t >= t_end:
        return final_temp
    progress = t / max(1, n_steps)
    return initial_temp * np.exp(-10.0 * progress)


def construct():
    """
    Construct centers of 50 equal circles in the unit square.
    Uses Langevin dynamics with annealing to find a good packing.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Parameters
    initial_temp = 0.01
    final_temp = 1e-6
    damping = 0.1
    exponent_start = 2.0
    exponent_end = 6.0
    dt = 0.001
    n_steps = 50000
    
    velocity = np.zeros_like(centers)
    
    step = 0
    while step < n_steps:
        if time.time() > deadline:
            break
        
        # Update temperature and exponent
        progress = step / n_steps
        temperature = initial_temp * np.exp(-10.0 * progress)
        exponent = exponent_start + (exponent_end - exponent_start) * progress
        
        # Langevin step
        centers, velocity = _langevin_step(centers, velocity, dt, temperature, exponent, damping)
        
        # Ensure centers stay in the square (hard constraint)
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        step += 1
        
        # Check improvement every 100 steps
        if step % 100 == 0:
            current_radius = _compute_radius(centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
    
    # Final polish: local optimization using gradient ascent on radius
    # Use a simple iterative method to push circles apart
    if time.time() < deadline - 5.0:
        polish_centers = best_centers.copy()
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            
            # Compute pairwise distances
            diffs = polish_centers[:, np.newaxis, :] - polish_centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Find closest pair
            min_dist = np.min(dists)
            
            # Compute forces to increase minimum distance
            # Push all pairs apart with force proportional to 1/d^2
            forces = np.zeros_like(polish_centers)
            inv_d2 = np.zeros_like(dists)
            mask = dists < np.inf
            inv_d2[mask] = 1.0 / (dists[mask] ** 2)
            
            forces = np.sum(diffs * inv_d2[:, :, np.newaxis], axis=1)
            
            # Also push away from walls
            for axis in range(2):
                d_left = polish_centers[:, axis] + 0.5
                d_right = 0.5 - polish_centers[:, axis]
                f_left = 1.0 / (d_left ** 2 + 1e-6)
                f_right = 1.0 / (d_right ** 2 + 1e-6)
                forces[:, axis] += f_left - f_right
            
            # Update
            step_size = 0.0001
            polish_centers += step_size * forces
            polish_centers[:, 0] = np.clip(polish_centers[:, 0], -0.5, 0.5)
            polish_centers[:, 1] = np.clip(polish_centers[:, 1], -0.5, 0.5)
            
            current_radius = _compute_radius(polish_centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = polish_centers.copy()
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius < best_radius * 0.99:
        # Something went wrong, return grid
        return grid[:N].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
