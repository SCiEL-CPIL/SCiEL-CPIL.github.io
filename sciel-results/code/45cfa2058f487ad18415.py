"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Check bounds
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
        
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    r = min(min_pair / 2.0, min_wall)
    return r

def _project(c):
    """Project points inside the cube."""
    return np.clip(c, -0.5, 0.5)

def _repulsion_step(c, r_target, k=1.0, alpha=0.1):
    """
    Perform one step of repulsion-based optimization.
    We treat the spheres as charged particles repelling each other.
    We also add repulsion from the walls to keep them centered if possible,
    but primarily we want to maximize the minimum distance.
    
    Strategy: 
    1. Compute forces on each sphere due to others (inverse square).
    2. Compute forces from walls (pushing inwards if close to wall).
    3. Update positions with a small step size.
    4. Project back into cube.
    """
    n = len(c)
    forces = np.zeros_like(c)
    
    # Pairwise repulsion
    # We want to push spheres apart.
    # F_ij = k * (c_i - c_j) / |c_i - c_j|^3
    # Using a cutoff or just all pairs since N=50 is small.
    for i in range(n):
        diff = c[i] - c # shape (n, 3)
        dists = np.linalg.norm(diff, axis=1)
        dists[i] = np.inf # avoid self
        # Avoid division by zero for coincident points (add small epsilon)
        dists[dists < 1e-8] = 1e-8
        
        # Force magnitude: k / dist^2, direction: normalized diff
        # F_i = sum_j k * (c_i - c_j) / |c_i - c_j|^3
        inv_d3 = 1.0 / (dists**3)
        inv_d3[i] = 0.0
        forces[i] = np.sum(diff * inv_d3[:, np.newaxis], axis=0) * k
        
    # Wall repulsion
    # If a sphere is close to a wall, push it away from the wall.
    # Wall at x=0.5: if c[i,0] > 0.5 - r_target, push in -x direction.
    # Wall at x=-0.5: if c[i,0] < -0.5 + r_target, push in +x direction.
    # We can model this as a potential well or just simple clamping + force.
    # Let's use a simple force that increases linearly as it approaches the boundary.
    # Actually, for packing, we want to maximize the distance to the wall.
    # So if dist_to_wall < r_target, we are "overlapping" the virtual wall.
    # We push it back.
    
    for i in range(n):
        for axis in range(3):
            pos = c[i, axis]
            # Distance to + wall
            d_pos = 0.5 - pos
            # Distance to - wall
            d_neg = 0.5 + pos
            
            # If close to + wall (d_pos < r_target), push towards -x (negative direction)
            if d_pos < r_target:
                # Force magnitude proportional to penetration
                pen = r_target - d_pos
                forces[i, axis] -= pen * 10.0 # Strong repulsion from wall
                
            # If close to - wall (d_neg < r_target), push towards +x (positive direction)
            if d_neg < r_target:
                pen = r_target - d_neg
                forces[i, axis] += pen * 10.0

    # Update positions
    # Normalize forces to avoid huge steps
    force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
    force_mags[force_mags == 0] = 1.0
    normalized_forces = forces / force_mags
    
    # Adaptive step size
    step_size = alpha * 0.05
    
    c_new = c + normalized_forces * step_size
    c_new = _project(c_new)
    
    return c_new

def _local_optimize(c, r_target, time_limit):
    """
    Iteratively apply repulsion steps, increasing r_target if we make progress.
    """
    t_start = time.time()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    # We try to push r_target up.
    # Start with a target slightly higher than current best.
    target = current_r * 1.01 + 0.001
    
    # If we can achieve 'target', we increase it.
    # If we fail to improve for many steps, we might lower it or just keep current best.
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - t_start < time_limit:
        # Perform a batch of repulsion steps
        for _ in range(10):
            c = _repulsion_step(c, target, k=1.0, alpha=0.1)
            
        current_r = _compute_radius(c)
        
        if current_r > best_r + 1e-6:
            best_r = current_r
            best_c = c.copy()
            no_improve_count = 0
            # Increase target
            target = current_r * 1.005 + 0.0001
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Stuck. Try a small random perturbation to escape local minimum
                c = best_c + np.random.normal(0, 0.01, size=best_c.shape)
                c = _project(c)
                no_improve_count = 0
                # Reset target to current best
                target = best_r * 1.01 + 0.001
                
    return best_c

def construct():
    t0 = time.time()
    deadline = t0 + 145.0 # Safety margin
    
    np.random.seed(42)
    random.seed(42)
    
    best_global_c = None
    best_global_r = 0.0
    
    # Multi-restart strategy
    # Different initial configurations:
    # 1. Grid based
    # 2. Random
    # 3. FCC-like
    
    num_restarts = 5
    time_per_restart = (deadline - t0 - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate initial config
        if restart == 0:
            # Grid 4x4x4, pick 50
            pts = []
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        p = np.array([i/3.0 - 0.5, j/3.0 - 0.5, k/3.0 - 0.5])
                        pts.append(p)
            pts = np.array(pts)
            # Shuffle and take 50
            idx = np.random.permutation(len(pts))[:N]
            c = pts[idx].copy()
        elif restart == 1:
            # Random
            c = np.random.uniform(-0.45, 0.45, size=(N, 3))
        elif restart == 2:
            # FCC-like
            a = 0.25
            offsets = np.array([
                [0, 0, 0], [0, 0.5*a, 0.5*a], [0.5*a, 0, 0.5*a], [0.5*a, 0.5*a, 0]
            ])
            pts = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        for off in offsets:
                            p = np.array([i, j, k]) * a + off
                            if np.all(np.abs(p) <= 0.49):
                                pts.append(p)
            pts = np.array(pts)
            if len(pts) >= N:
                idx = np.random.permutation(len(pts))[:N]
                c = pts[idx].copy()
            else:
                c = np.random.uniform(-0.45, 0.45, size=(N, 3))
        elif restart == 3:
            # Body centered cubic
            a = 0.25
            pts = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        p = np.array([i, j, k]) * a
                        pts.append(p)
                        p2 = p + np.array([0.125, 0.125, 0.125])
                        if np.all(np.abs(p2) <= 0.49):
                            pts.append(p2)
            pts = np.array(pts)
            if len(pts) >= N:
                idx = np.random.permutation(len(pts))[:N]
                c = pts[idx].copy()
            else:
                c = np.random.uniform(-0.45, 0.45, size=(N, 3))
        else:
            # Random with larger spread
            c = np.random.uniform(-0.49, 0.49, size=(N, 3))
            
        c = _project(c)
        
        # Optimize
        remaining_time = deadline - time.time() - 2.0
        if remaining_time < 1.0:
            continue
            
        optimized_c = _local_optimize(c, _compute_radius(c), remaining_time)
        
        r_opt = _compute_radius(optimized_c)
        
        if r_opt > best_global_r:
            best_global_r = r_opt
            best_global_c = optimized_c.copy()
            
    if best_global_c is None:
        # Fallback: random
        best_global_c = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
    return best_global_c
# EVOLVE-BLOCK-END
