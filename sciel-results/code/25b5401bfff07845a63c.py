# EVOLVE-BLOCK-START
import numpy as np
import time

N = 100

def _compute_radius(c):
    """Compute the largest common radius for given centers."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _generate_bcc_base(n_points=100):
    """
    Generate a base configuration of n_points from a BCC lattice,
    selecting the points closest to the origin.
    """
    # Integer lattice points in [-2, 2]
    grid_int = np.array(np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3))).T.reshape(-1, 3)
    # Half-integer lattice points in [-1.5, 1.5]
    grid_half = np.array(np.meshgrid(np.arange(-1.5, 2, 1), np.arange(-1.5, 2, 1), np.arange(-1.5, 2, 1))).T.reshape(-1, 3)
    
    all_pts = np.vstack([grid_int, grid_half])
    
    # Sort by distance to origin
    dists = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(dists)[:n_points]
    pts = all_pts[idx].copy()
    
    # Center at origin
    pts -= np.mean(pts, axis=0)
    
    # Scale to fit roughly in the cube
    # Find the max extent and scale so it fits with some margin
    max_extent = np.max(np.abs(pts))
    scale = 0.4 / max_extent  # Leave some room for refinement
    pts *= scale
    
    return pts

def _force_directed_refine(c, time_limit=100.0):
    """
    Force-directed refinement: push spheres apart and away from walls.
    Uses actual (non-smooth) radius for acceptance.
    """
    c = np.array(c, dtype=np.float64)
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    step_size = 0.002
    max_iter = 10000
    no_improve_count = 0
    patience = 200
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion - stronger for closer pairs
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random small push for coincident points
                    forces[i] += np.random.randn(3) * 0.001
                    forces[j] -= np.random.randn(3) * 0.001
                else:
                    # Repulsive force: stronger when close
                    # Use 1/dist^3 for stronger short-range repulsion
                    f_mag = 1.0 / (dist * dist * dist)
                    f = d * f_mag
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion - push away from walls
        for i in range(n):
            for k in range(3):
                dist_to_wall = 0.5 - np.abs(c[i, k])
                if dist_to_wall < 0.08:
                    # Stronger repulsion closer to wall
                    wall_force = (0.08 - dist_to_wall) * 50.0
                    forces[i, k] += np.sign(c[i, k]) * wall_force
        
        # Normalize forces per point to prevent any single point from dominating
        norm_f = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_f = np.maximum(norm_f, 1e-10)
        forces = forces / norm_f
        
        # Move
        new_c = c + step_size * forces
        
        # Clamp to cube with small margin
        new_c = np.clip(new_c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Check if improved
        new_r = _compute_radius(new_c)
        
        if new_r > best_r:
            c = new_c
            best_c = c.copy()
            best_r = new_r
            no_improve_count = 0
            # Slowly increase step size on improvement
            step_size = min(step_size * 1.01, 0.01)
        else:
            no_improve_count += 1
            step_size *= 0.98
            if no_improve_count > patience or step_size < 1e-7:
                break
    
    return best_c, best_r

def _annealed_force_refine(c, time_limit=80.0):
    """
    Force-directed refinement with simulated annealing acceptance.
    Allows occasional worse moves to escape local minima.
    """
    c = np.array(c, dtype=np.float64)
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    step_size = 0.003
    temperature = 0.005
    cooling_rate = 0.9995
    max_iter = 15000
    no_improve_count = 0
    patience = 300
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    forces[i] += np.random.randn(3) * 0.001
                    forces[j] -= np.random.randn(3) * 0.001
                else:
                    f_mag = 1.0 / (dist * dist * dist)
                    f = d * f_mag
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion
        for i in range(n):
            for k in range(3):
                dist_to_wall = 0.5 - np.abs(c[i, k])
                if dist_to_wall < 0.08:
                    wall_force = (0.08 - dist_to_wall) * 50.0
                    forces[i, k] += np.sign(c[i, k]) * wall_force
        
        # Normalize forces
        norm_f = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_f = np.maximum(norm_f, 1e-10)
        forces = forces / norm_f
        
        # Add some random perturbation for exploration
        noise = np.random.randn(n, 3) * 0.001
        move = step_size * forces + noise
        
        new_c = c + move
        new_c = np.clip(new_c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(new_c)
        delta_r = new_r - best_r
        
        # Accept if better, or with probability based on temperature
        if delta_r > 0 or (temperature > 1e-8 and np.random.rand() < np.exp(delta_r / temperature)):
            c = new_c
            if new_r > best_r:
                best_c = c.copy()
                best_r = new_r
                no_improve_count = 0
            temperature *= cooling_rate
        else:
            no_improve_count += 1
            temperature *= cooling_rate
        
        if no_improve_count > patience or temperature < 1e-8:
            break
    
    return best_c, best_r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    Uses BCC lattice as initial configuration, then force-directed refinement.
    """
    total_time_limit = 120.0
    start_time = time.time()
    
    # Generate BCC base configuration
    c = _generate_bcc_base(N)
    
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Phase 1: Initial force-directed refinement
    time_for_phase1 = min(40.0, total_time_limit - 80.0)
    if time_for_phase1 > 5.0:
        c_refined, r_refined = _force_directed_refine(c, time_limit=time_for_phase1)
        if r_refined > best_r:
            best_c = c_refined
            best_r = r_refined
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Phase 2: Annealed refinement from best
    if remaining > 30.0:
        time_for_phase2 = min(60.0, remaining - 10.0)
        if time_for_phase2 > 5.0:
            c_annealed, r_annealed = _annealed_force_refine(best_c, time_limit=time_for_phase2)
            if r_annealed > best_r:
                best_c = c_annealed
                best_r = r_annealed
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Phase 3: Final quick refinement if time allows
    if remaining > 15.0:
        time_for_phase3 = min(20.0, remaining - 5.0)
        if time_for_phase3 > 3.0:
            c_final, r_final = _force_directed_refine(best_c, time_limit=time_for_phase3)
            if r_final > best_r:
                best_c = c_final
                best_r = r_final
    
    # Ensure final result is valid
    final_r = _compute_radius(best_c)
    if final_r < 0.01:
        # Fallback: use the BCC base if something went wrong
        best_c = c.copy()
        best_r = _compute_radius(c)
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_c
# EVOLVE-BLOCK-END
