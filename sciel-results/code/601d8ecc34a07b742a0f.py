"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^3."""
    # Distance to nearest face
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    
    # Minimum pairwise distance
    n = len(centres)
    if n < 2:
        return min_face
    
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)

def _repulsive_force(centres, sigma=0.05, wall_margin=0.05):
    """
    Compute a repulsive force for each sphere.
    Spheres repel each other strongly when close.
    Spheres are pushed away from walls.
    """
    n = len(centres)
    forces = np.zeros_like(centres)
    
    # Pairwise repulsion
    # diff[i, j] = centres[i] - centres[j]
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Force magnitude ~ 1 / (dist^2 + sigma^2)
    # This is smooth and avoids division by zero
    force_mag = 1.0 / (dists_sq + sigma**2)
    
    # Zero out diagonal forces
    np.fill_diagonal(force_mag, 0.0)
    
    # Sum forces from all other spheres
    # forces[i] = sum_j (diff[i,j] * force_mag[i,j])
    forces += np.sum(diff * force_mag[:, :, np.newaxis], axis=1)
    
    # Wall repulsion
    # Push away from walls if close
    # x in [-0.5, 0.5]
    # dist to wall: 0.5 - |x|
    # Force should be strong if dist is small
    
    # Let's use a potential-like force
    # F_wall = -dV/dx
    # If we want to maximize min(dist_to_wall, min_pair/2),
    # a simple heuristic is to push away from walls with a force proportional to (1/dist)^2
    
    # Vectorized wall force
    # For each coordinate k:
    # if c_ik > 0: dist = 0.5 - c_ik, force_x = -1/(dist+eps)^2
    # if c_ik < 0: dist = 0.5 + c_ik, force_x = +1/(dist+eps)^2
    
    abs_c = np.abs(centres)
    dist_to_wall = 0.5 - abs_c
    # Add small epsilon to avoid infinity
    dist_to_wall_safe = np.maximum(dist_to_wall, 1e-6)
    
    # Force magnitude
    wall_force_mag = 1.0 / (dist_to_wall_safe**2)
    
    # Direction: away from wall
    # If c > 0, push negative (towards center) -> -sign(c)
    # If c < 0, push positive (towards center) -> -sign(c)
    # Wait, if c > 0, wall is at 0.5. To move away from wall, move towards 0. So force is negative.
    # sign(c) is positive. So -sign(c) is correct.
    # If c < 0, wall is at -0.5. To move away, move towards 0. So force is positive.
    # sign(c) is negative. So -sign(c) is positive. Correct.
    
    # However, we only want this force to be significant near the walls.
    # The 1/d^2 term handles that naturally.
    
    forces -= wall_force_mag * np.sign(centres)
    
    # Add a tiny centering force to prevent drift? 
    # The wall force should be enough to keep them in.
    # But if they are far from walls, force is ~0. 
    # The repulsive force might push them towards corners? 
    # Actually, repulsion pushes them apart. In a cube, the densest packing is near the center.
    # So they will naturally cluster in the center if repulsion is the only force? 
    # No, repulsion pushes them apart until they hit walls or each other.
    # The equilibrium is where repulsion balances wall pressure.
    
    return forces

def _optimize(centres, time_limit=120.0):
    """
    Optimize centres using gradient ascent on a proxy for radius.
    We maximize the minimum of (pair_dist/2, wall_dist).
    This is non-smooth. 
    Using a repulsive force field approximates the gradient of the sum of pair distances 
    or similar, which often correlates with maximizing the minimum distance.
    
    We will use a simple gradient ascent with adaptive step size.
    """
    start_time = time.time()
    current_centres = centres.copy()
    n = len(current_centres)
    
    # Initial step size
    step_size = 0.01
    best_radius = _compute_radius(current_centres)
    best_centres = current_centres.copy()
    
    # Track history of radius for adaptation
    radius_history = [best_radius]
    
    max_iter = 5000
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        forces = _repulsive_force(current_centres)
        
        # Normalize forces to avoid huge steps
        force_norms = np.linalg.norm(forces, axis=1)
        max_norm = np.max(force_norms)
        if max_norm > 1e-9:
            forces = forces / max_norm
        
        # Update
        new_centres = current_centres + step_size * forces
        
        # Project back to cube
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centres)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centres = new_centres.copy()
            # Increase step size slightly if improving
            step_size *= 1.05
        else:
            # Decrease step size
            step_size *= 0.8
            
        current_centres = new_centres
        radius_history.append(new_radius)
        
        # Convergence check
        if len(radius_history) > 100:
            if np.std(radius_history[-100:]) < 1e-10:
                break
                
    return best_centres, best_radius

def _generate_bcc_initial():
    """
    Generate a BCC-like initial configuration.
    Divide cube into 2x2x2 sub-cubes? No, 50 spheres.
    A simple cubic grid of 4x4x4 = 64 is too many.
    3x3x3 = 27 is too few.
    
    Let's use a 4x4x4 grid and randomly remove 14 to get 50?
    Or start with a 4x4x4 grid and optimize?
    The BCC lattice is denser.
    
    Let's try a hybrid:
    1. Create a 4x4x4 grid.
    2. Add BCC centers (center of each sub-cube).
    3. Total points = 64 (grid) + 8 (bcc) = 72? No.
    
    Let's stick to a perturbed grid which is a robust starting point.
    Or, use a "random repulsive" initialization which is often better for hard spheres.
    """
    np.random.seed(42)
    
    # Strategy: Random points, then a few steps of repulsion to spread them out
    centres = np.random.rand(N, 3) - 0.5
    
    # Quick pre-optimization to spread them
    for _ in range(50):
        forces = _repulsive_force(centres)
        force_norms = np.linalg.norm(forces, axis=1)
        max_norm = np.max(force_norms)
        if max_norm > 1e-9:
            forces = forces / max_norm
        centres += 0.05 * forces
        centres = np.clip(centres, -0.49, 0.49)
        
    return centres

def construct():
    """Construct 50 sphere centres in [-0.5, 0.5]^3 to maximize minimum radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centres = None
    best_radius = 0.0
    
    # Multiple restarts with different seeds
    # The vectorized optimization is fast, so we can afford more restarts
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial configuration
        if i == 0:
            # First restart: BCC-like / Grid
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            # Take first 50
            initial = grid[:N].copy()
            np.random.seed(i)
            initial += np.random.randn(N, 3) * 0.02
            initial = np.clip(initial, -0.49, 0.49)
        else:
            # Subsequent restarts: Random repulsive
            np.random.seed(100 + i)
            initial = _generate_bcc_initial()
        
        # Optimize
        time_remaining = deadline - time.time() - 5.0 # Leave 5s for final check
        if time_remaining < 1.0:
            break
            
        opt_centres, opt_radius = _optimize(initial, time_limit=time_remaining)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centres = opt_centres
            
    # Final refinement on the best found
    if best_centres is not None and time.time() < deadline - 5.0:
        time_remaining = deadline - time.time() - 2.0
        refined_centres, refined_radius = _optimize(best_centres, time_limit=time_remaining)
        if refined_radius > best_radius:
            best_radius = refined_radius
            best_centres = refined_centres
            
    return best_centres
# EVOLVE-BLOCK-END
