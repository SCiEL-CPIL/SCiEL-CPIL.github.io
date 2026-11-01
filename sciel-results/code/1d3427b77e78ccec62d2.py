"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

def _force_directed_optimize(c0, time_limit, initial_temp=0.01, damping=0.95):
    """
    Physics-based optimization using Velocity Verlet integration.
    We treat the spheres as particles with repulsive forces (inverse square)
    and attractive forces towards the center (to keep them inside).
    The 'temperature' controls the magnitude of the repulsive force.
    """
    c = c0.copy()
    v = np.zeros_like(c)
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.stack([i_idx, j_idx], axis=1)
    
    best_c = c.copy()
    best_score = _score(best_c)
    
    start_time = time.time()
    
    # Parameters for the force law
    # F_rep = k * (1/d^2) * (r_vec/d)  for d < d_cut
    # F_wall = -k_wall * (x - x_max) if x > x_max - margin
    # We want to maximize min distance.
    # Let's use a potential energy U = -sum(1/d) + sum(wall_violation^2)
    # Actually, for packing, we want to push them apart.
    # Let's use a simple repulsion: F = k / d^2.
    # To prevent explosion, we cap the force.
    
    k_rep = 1.0
    k_wall = 10.0
    margin = 0.01
    max_force = 1.0
    
    # Adaptive step size
    dt = 0.01
    
    while time.time() < time_limit:
        # Check time
        if time.time() > time_limit:
            break
            
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diffs = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        
        # Force magnitude: k / d^2
        # Direction: unit vector from j to i (pushes i away from j)
        unit_vecs = diffs / dists_safe[:, np.newaxis]
        mags = k_rep / (dists_safe ** 2)
        
        # Cap forces to prevent instability
        mags = np.clip(mags, 0, max_force)
        
        f_pairs = mags[:, np.newaxis] * unit_vecs
        
        # Scatter forces
        np.add.at(forces, i_idx, f_pairs)
        np.add.at(forces, j_idx, -f_pairs)
        
        # Wall forces (attractive if inside, repulsive if outside)
        # We want spheres to stay inside [-0.5, 0.5]
        # If a sphere is close to the wall, pull it in.
        # If it's outside, push it in.
        
        # Let's define a soft wall potential.
        # Force towards center if outside, or if we want to fill space, maybe just keep them in.
        # For packing, we usually want them to touch the walls.
        # So, let's apply a force that pushes them towards the boundary if they are too far in?
        # No, that's complex. Let's just constrain them to the box with hard walls 
        # but add a small inward force to prevent sticking to one wall and leaving space elsewhere.
        
        # Actually, for maximizing radius, we want them to spread out.
        # The repulsion handles spreading. The walls handle containment.
        # Let's add a small random jitter to break symmetry if stuck.
        
        # Apply velocity Verlet
        # v_half = v + 0.5 * dt * F
        v_half = v + 0.5 * dt * forces
        
        # x_new = x + dt * v_half
        c_new = c + dt * v_half
        
        # Clip to box
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Compute new forces at x_new
        diffs_new = c_new[i_idx] - c_new[j_idx]
        dists_new = np.linalg.norm(diffs_new, axis=1)
        dists_new_safe = np.where(dists_new < 1e-6, 1e-6, dists_new)
        unit_vecs_new = diffs_new / dists_new_safe[:, np.newaxis]
        mags_new = k_rep / (dists_new_safe ** 2)
        mags_new = np.clip(mags_new, 0, max_force)
        f_pairs_new = mags_new[:, np.newaxis] * unit_vecs_new
        forces_new = np.zeros_like(c_new)
        np.add.at(forces_new, i_idx, f_pairs_new)
        np.add.at(forces_new, j_idx, -f_pairs_new)
        
        # v_new = v_half + 0.5 * dt * F_new
        v_new = v_half + 0.5 * dt * forces_new
        
        # Damping
        v_new *= damping
        
        c = c_new
        v = v_new
        
        # Check score
        curr_score = _score(c)
        if curr_score > best_score:
            best_score = curr_score
            best_c = c.copy()
            
        # Adaptive dt: if score drops significantly, reduce dt
        # Simple heuristic: if best_score hasn't improved in a while, reduce dt
        # Here we just keep it constant for simplicity, damping handles stability
        
    return best_c

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # Start with a good grid initialization
    best_c = _init_grid()
    best_score = _score(best_c)
    
    # Run Force-Directed Optimization from the grid start
    c_fd = _force_directed_optimize(best_c, deadline - 5.0)
    sc_fd = _score(c_fd)
    
    if sc_fd > best_score:
        best_score = sc_fd
        best_c = c_fd
    
    # If we have time, try a random start FD
    if time.time() < deadline - 10.0:
        c_rand = _init_random()
        remaining = deadline - 10.0 - time.time()
        if remaining > 5.0:
            c_rand_fd = _force_directed_optimize(c_rand, remaining)
            sc_rand_fd = _score(c_rand_fd)
            if sc_rand_fd > best_score:
                best_score = sc_rand_fd
                best_c = c_rand_fd
                
    # Final check and return
    return best_c
# EVOLVE-BLOCK-END
