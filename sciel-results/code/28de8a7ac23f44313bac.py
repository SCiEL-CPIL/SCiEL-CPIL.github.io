import numpy as np
import time

N = 300

def _init_hex(n):
    """Generate hexagonal lattice points inside unit disk."""
    # Spacing needs to be small enough to fit n points
    # Area of unit disk is pi. Area per circle ~ pi/n. 
    # But we are packing circles, so density is ~ 0.9.
    # Let's just generate a fine grid and pick n closest to center.
    spacing = 0.05
    k = int(1.5 / spacing)
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                pts.append([x, y])
    pts = np.array(pts)
    # Sort by distance to origin
    r = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return pts[idx]

def _compute_forces_hertz(pts, r):
    """
    Compute Hertzian contact forces.
    Force magnitude is proportional to overlap^4.
    Returns force array (N, 2).
    """
    n = len(pts)
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq)
    
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-8, 1e-8, dist)
    
    # Overlap
    overlap = 2.0 * r - dist
    # Only repel if overlapping
    overlap = np.where(overlap > 0, overlap, 0.0)
    
    # Hertzian force: F = k * overlap^4
    # We use a spring constant k that is large enough to separate circles quickly
    # but small enough to avoid numerical instability in Verlet.
    k_hertz = 1000.0 
    force_mag = k_hertz * (overlap ** 4)
    
    # Direction
    dir_x = dx / dist_safe
    dir_y = dy / dist_safe
    
    # Force on i from j is in direction i - j (repulsive)
    # Note: dx = x_i - x_j. So direction is correct for repulsion on i.
    force_x = np.sum(force_mag * dir_x, axis=1)
    force_y = np.sum(force_mag * dir_y, axis=1)
    
    # Boundary forces
    # Treat boundary as a wall at radius 1-r.
    # If point is outside, apply Hertzian repulsion from the wall.
    r_pts = np.hypot(pts[:, 0], pts[:, 1])
    # Effective distance from boundary wall
    # Wall is at R_wall = 1.0 - r
    # Overlap with wall = r_pts - R_wall
    wall_overlap = r_pts - (1.0 - r)
    wall_overlap = np.where(wall_overlap > 0, wall_overlap, 0.0)
    
    force_wall_mag = k_hertz * (wall_overlap ** 4)
    
    # Direction is inward (towards origin)
    dir_bx = np.where(r_pts > 1e-8, -pts[:, 0] / np.where(r_pts > 1e-8, r_pts, 1.0), 0.0)
    dir_by = np.where(r_pts > 1e-8, -pts[:, 1] / np.where(r_pts > 1e-8, r_pts, 1.0), 0.0)
    
    force_x += force_wall_mag * dir_bx
    force_y += force_wall_mag * dir_by
    
    return np.column_stack((force_x, force_y))

def _verlet_step(pts, prev_pts, r, dt, damping=0.99):
    """
    Perform one Verlet integration step.
    pts: current positions
    prev_pts: previous positions
    r: target radius
    dt: time step
    damping: velocity damping factor
    """
    forces = _compute_forces_hertz(pts, r)
    
    # Verlet update: x_new = x + (x - x_prev) * damping + a * dt^2
    # a = F / m, assuming m=1
    acc = forces
    
    # Calculate new positions
    vel = (pts - prev_pts) * damping
    pts_new = pts + vel + acc * (dt * dt)
    
    # Project back inside unit circle strictly to prevent escape
    r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
    max_r = 1.0 - 1e-9
    mask = r_new > max_r
    if np.any(mask):
        pts_new[mask, 0] *= max_r / r_new[mask]
        pts_new[mask, 1] *= max_r / r_new[mask]
        
    return pts_new

def _compute_score(pts):
    """Compute the actual packing radius."""
    n = len(pts)
    if n < 2:
        return 0.0
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    
    return min(min_dist / 2.0, min_boundary)

def _soft_min_grad(pts, r, beta=100.0):
    """
    Compute gradient of the soft-min radius objective.
    Objective: max min(dist/2, 1-|c|)
    Soft-min approximation: -1/beta * log(sum(exp(-beta * x_i)))
    Gradient of soft-min w.r.t x_i is proportional to exp(-beta * x_i) / sum(exp(-beta * x_j))
    """
    n = len(pts)
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Pairwise distances to center (for boundary)
    r_pts = np.hypot(pts[:, 0], pts[:, 1])
    bound_dist = 1.0 - r_pts
    
    # We want to maximize the minimum of:
    # 1. dist_ij / 2
    # 2. 1 - |c_i|
    
    # Let's construct a vector of all constraints.
    # To save memory, we process in chunks or just use the min directly for a simpler gradient.
    # For a smooth gradient, we need the weights.
    
    # Simplified approach: Gradient of min is the gradient of the active constraint.
    # If we use soft-min, we compute weights for all pairs and boundary.
    # This is expensive for N=300 (45k pairs).
    # Let's use a simpler heuristic: 
    # Find the closest pair and the closest boundary point.
    # Move points away from their closest neighbor and towards center if close to boundary.
    
    # Find closest neighbor for each point
    np.fill_diagonal(dist, np.inf)
    min_dists = np.min(dist, axis=1)
    
    # Boundary distance
    min_bound = np.min(bound_dist)
    
    # Determine which constraint is active globally
    # If min_dists/2 < min_bound, the bottleneck is internal.
    # Else, boundary is bottleneck.
    
    # For a local polish, we can just use the gradient of the sum of squared overlaps 
    # or a simple repulsion based on the current min distance.
    
    # Let's implement a simple gradient ascent on the min distance using the "closest pair" logic.
    # For each point i, find its closest neighbor j.
    # Force i away from j.
    # Also, if i is closer to boundary than to any neighbor (relative to r), push in.
    
    forces = np.zeros_like(pts)
    
    # Internal repulsion based on closest neighbors
    # We need the index of the closest neighbor
    idx_closest = np.argmin(dist, axis=1)
    
    for i in range(n):
        j = idx_closest[i]
        d = dist[i, j]
        if d < 2.0 * r:
            # Repel
            dx_ij = pts[i, 0] - pts[j, 0]
            dy_ij = pts[i, 1] - pts[j, 1]
            norm = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
            if norm > 1e-8:
                # Force magnitude proportional to overlap
                overlap = 2.0 * r - d
                forces[i, 0] += overlap * dx_ij / norm
                forces[i, 1] += overlap * dy_ij / norm
        else:
            # If not overlapping, we might still want to push them apart slightly to maximize min dist?
            # No, only if they are the global minimum.
            pass
            
    # Boundary attraction/repulsion
    for i in range(n):
        r_i = r_pts[i]
        if r_i > 1.0 - r:
            # Push in
            overlap = r_i - (1.0 - r)
            forces[i, 0] -= overlap * pts[i, 0] / r_i
            forces[i, 1] -= overlap * pts[i, 1] / r_i
        elif r_i > 0.9:
            # Slight pull in if close to boundary, to keep them safe?
            # Actually, we want to maximize the minimum. If boundary is the min, we need to move them in.
            # If internal is the min, we don't care about boundary as much unless it becomes the new min.
            pass

    return forces

def construct():
    start_time = time.time()
    time_limit = 145.0 # 5s safety margin
    
    # Initialize
    pts = _init_hex(N)
    current_r = _compute_score(pts)
    
    # Start r slightly below current
    r = max(0.01, current_r * 0.9)
    
    best_pts = pts.copy()
    best_score = current_r
    
    # Simulation Parameters
    dt = 0.005
    damping = 0.995
    steps = 5000
    
    prev_pts = pts.copy()
    
    for i in range(steps):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Adaptive r increase
        # Increase r slowly. If the system is stable, we can increase faster.
        # For simplicity, linear increase with a cap.
        r = min(0.053, current_r + i * 0.00002)
        
        pts_new = _verlet_step(pts, prev_pts, r, dt, damping)
        
        prev_pts = pts
        pts = pts_new
        
        # Check score periodically
        if i % 100 == 0:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
            # Update current_r for the next step's r target
            current_r = score
            
    # Final polish phase
    # Use the best configuration found
    pts = best_pts.copy()
    prev_pts = pts.copy()
    r = best_score
    
    # Run a short Verlet simulation with the final r to relax any residual stresses
    # and then a gradient-based polish.
    
    # 1. Relax with Verlet
    for i in range(1000):
        if time.time() - start_time > time_limit - 5:
            break
        pts_new = _verlet_step(pts, prev_pts, r, dt, damping=0.99)
        prev_pts = pts
        pts = pts_new
        
        if i % 50 == 0:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
            r = score # Track the improving radius
            
    # 2. Gradient polish
    # Simple steepest ascent on the min distance
    # We'll do 200 steps
    lr = 0.01
    for i in range(200):
        if time.time() - start_time > time_limit - 1:
            break
        
        forces = _soft_min_grad(pts, r)
        
        # Normalize forces to prevent large jumps
        norm_f = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_f = np.where(norm_f > 1e-8, norm_f, 1.0)
        forces = forces / norm_f
        
        pts += lr * forces
        
        # Project inside
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        mask = r_pts > 1.0
        if np.any(mask):
            pts[mask, 0] *= 1.0 / r_pts[mask]
            pts[mask, 1] *= 1.0 / r_pts[mask]
            
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        else:
            # If score decreased, revert or reduce lr?
            # For simplicity, just keep the best found so far.
            # Reverting might be better to avoid drifting into bad local minima.
            # pts = best_pts.copy() 
            pass
        
        # Decay learning rate
        lr *= 0.99

    return best_pts
