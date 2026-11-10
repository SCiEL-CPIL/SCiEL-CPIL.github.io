"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Uses a simple Molecular Dynamics (MD) simulation to pack N circles.
    We treat circles as particles with repulsive forces when they overlap
    and attractive forces towards the center to keep them inside.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # 1. Initialization: Hexagonal lattice provides a good starting point
    #    that is already relatively dense and valid.
    #    We generate a slightly larger hex grid and select the N closest to center.
    spacing = 0.055  # Slightly larger than expected optimal to ensure initial non-overlap
    # Generate points in a hex grid
    points = []
    k = 15
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    
    # Keep only points inside the unit circle (with a small buffer)
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii < 1.0 - spacing * 0.5
    points = points[mask]
    
    # Sort by distance to origin and take the first N
    # If we have fewer than N, we might need to shrink spacing and retry, 
    # but for N=300 and r~0.05, a grid of size ~15x15 should suffice.
    if len(points) < N:
        # Fallback: random init if grid fails (unlikely)
        theta = np.linspace(0, 2*np.pi, N)
        r = np.sqrt(np.random.rand(N))
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)[:N]
        points = points[indices]
    
    # 2. MD Simulation
    # Parameters
    dt = 0.005
    temperature = 1.0
    cooling = 0.999
    min_temp = 0.001
    
    # Pre-allocate velocity
    v = np.zeros_like(points)
    
    best_r = 0.0
    best_points = points.copy()
    
    while temperature > min_temp and (time.time() - start_time) < time_limit:
        
        # Calculate forces
        # We want to maximize the minimum distance.
        # A common heuristic: Repel all pairs, but weight closer pairs more heavily?
        # Or simply: Repel if distance < 2*current_min_dist? 
        # Better: Use a potential that pushes apart any pair closer than some target.
        # Let's use a simple repulsive force for ALL pairs to maintain uniformity,
        # and a weak centripetal force to keep them in the circle.
        
        # Vectorized pairwise differences
        # To save memory/time, we can do this in chunks or just accept the O(N^2) cost 
        # for N=300 (90k pairs) which is very fast in NumPy.
        
        dx = points[:, 0:1] - points[0:1, 0]
        dy = points[:, 1:2] - points[0:1, 1]
        
        # Wait, broadcasting correctly:
        # points is (N, 2)
        # dx = points[:, 0] - points[0, :] -> (N, N)
        dx = points[:, 0] - points[0, :]
        dy = points[:, 1] - points[1, :]
        
        dist_sq = dx**2 + dy**2
        # Avoid division by zero
        dist_sq[dist_sq == 0] = 1e-12
        dist = np.sqrt(dist_sq)
        
        # Repulsive force: F = 1 / dist^2 (inverse square)
        # We only want repulsion if they are "too close"? 
        # If we repel everything, they will spread out to the boundary.
        # This is good. We want to fill the space uniformly.
        
        force_mag = 1.0 / dist_sq
        # Zero out self-interaction
        np.fill_diagonal(force_mag, 0)
        
        fx = force_mag * dx
        fy = force_mag * dy
        
        # Sum forces for each particle
        Fx = np.sum(fx, axis=0)
        Fy = np.sum(fy, axis=0)
        
        # Centripetal force: pull towards origin to prevent flying off
        # F_center = -k * pos
        Fx -= 0.01 * points[:, 0]
        Fy -= 0.01 * points[:, 1]
        
        # Update velocity (with damping)
        v[:, 0] = 0.9 * v[:, 0] + dt * Fx
        v[:, 1] = 0.9 * v[:, 1] + dt * Fy
        
        # Update position
        points[:, 0] += dt * v[:, 0]
        points[:, 1] += dt * v[:, 1]
        
        # Constraint: Keep inside unit circle
        # If a point goes outside, push it back to the boundary
        r = np.hypot(points[:, 0], points[:, 1])
        outside_mask = r > 1.0
        if np.any(outside_mask):
            # Normalize and set to 1.0 (or slightly less to allow movement)
            # Actually, if we push exactly to 1.0, it might get stuck.
            # Let's project back to r = 1.0 - epsilon
            # But we also need to handle the velocity to prevent jittering.
            # Simple projection:
            points[outside_mask, 0] *= (1.0 - 0.001) / r[outside_mask]
            points[outside_mask, 1] *= (1.0 - 0.001) / r[outside_mask]
            # Damp velocity for outside particles
            v[outside_mask, :] *= 0.5
            
        # Calculate current radius (minimum distance / 2)
        # We can estimate this faster or just compute it periodically.
        # For now, let's compute it every few steps or if we have time.
        # Actually, computing min dist is O(N^2). We just did O(N^2) for forces.
        # Let's reuse dist matrix from force calculation?
        # The dist matrix calculated above is for the PREVIOUS positions.
        # We need current distances.
        
        # Optimization: Only compute score every 10 steps
        if int(1/temperature) % 10 == 0:
             # Recalculate distances for scoring
             dx_s = points[:, 0] - points[0, :]
             dy_s = points[:, 1] - points[1, :]
             dist_s = np.sqrt(dx_s**2 + dy_s**2)
             np.fill_diagonal(dist_s, np.inf)
             min_dist = np.min(dist_s)
             current_r = min_dist / 2.0
             
             # Also check container constraint (should be handled by projection, but verify)
             r_check = np.hypot(points[:, 0], points[:, 1])
             if np.max(r_check) <= 1.0:
                 if current_r > best_r:
                     best_r = current_r
                     best_points = points.copy()
                     
        # Cool down
        temperature *= cooling
        
    # Final check and return
    # Ensure the returned points are valid
    # If best_points is better, use it. Otherwise use current points if valid.
    # Let's just return best_points.
    
    # One final verification of best_points
    # If it's invalid (e.g. due to numerical drift), fall back to current or initial
    # For safety, let's re-calculate the score of best_points
    dx_f = best_points[:, 0] - best_points[0, :]
    dy_f = best_points[:, 1] - best_points[1, :]
    dist_f = np.sqrt(dx_f**2 + dy_f**2)
    np.fill_diagonal(dist_f, np.inf)
    r_f = np.min(dist_f) / 2.0
    r_cont = 1.0 - np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
    final_r = min(r_f, r_cont)
    
    if final_r < 0.05: # If something went wrong badly
        return points # Return the last state, hoping it's better than garbage
    
    return best_points
# EVOLVE-BLOCK-END
