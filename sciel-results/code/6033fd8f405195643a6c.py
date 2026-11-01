import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """
    Calculate the score for a given set of positions.
    Score is the largest common radius r such that circles are disjoint and inside.
    """
    if len(positions) != N:
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Check container boundary
    dists_origin = np.linalg.norm(positions, axis=1)
    r_container = 1.0 - np.max(dists_origin)
    
    # Check inter-circle distances
    # Efficiently compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    r = min(r_container, r_pair)
    if r < 0:
        return 0.0
    return r

def _init_rings(n):
    """
    Initialize points in concentric rings.
    """
    points = np.zeros((n, 2))
    if n == 0:
        return points
    
    points[0] = [0.0, 0.0]
    remaining = n - 1
    radius = 0.5
    ring_size = 6
    
    idx = 1
    while remaining > 0:
        # Number of points in this ring
        count = min(remaining, ring_size)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        points[idx:idx+count] = np.column_stack((xs, ys))
        idx += count
        remaining -= count
        ring_size += 6
        radius += 0.25 # Heuristic expansion
        
        # If we ran out of space, shrink radius and try to fit more? 
        # For N=50, this simple heuristic is usually enough for a good start.
        
    return points

def _dynamics_step(positions, velocities, r_target, wall_points, dt, damping, k_rep, k_wall):
    """
    Perform one step of rigid-body dynamics with wall particles.
    """
    forces = np.zeros_like(positions)
    n = len(positions)
    
    # 1. Inter-circle forces
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)
            min_dist = 2.0 * r_target
            if dist < min_dist and dist > 1e-9:
                # Repulsive force
                overlap = min_dist - dist
                force_mag = k_rep * overlap
                force_vec = (diff / dist) * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
    
    # 2. Wall forces
    # For each circle, find the nearest wall point
    # Wall points are on the unit circle
    for i in range(n):
        p = positions[i]
        # Find nearest wall point
        # Efficient: angle of p
        angle = np.arctan2(p[1], p[0])
        # Wall points are evenly spaced
        n_wall = len(wall_points)
        # Find index of nearest wall point
        # Wall points are at angles 2*pi*k/n_wall
        k = int(np.round(angle / (2 * np.pi) * n_wall)) % n_wall
        w = wall_points[k]
        
        diff = p - w
        dist = np.linalg.norm(diff)
        min_dist = r_target
        if dist < min_dist:
            overlap = min_dist - dist
            force_mag = k_wall * overlap
            if dist > 1e-9:
                force_vec = (diff / dist) * force_mag
            else:
                # If exactly at wall point (center at origin?), push out? 
                # But center at origin is fine. 
                # If p is very close to w, and w is on boundary, p is outside? 
                # Wait, w is on unit circle. If p is close to w, p is near boundary.
                # Force should push p INWARDS (towards origin) if it's penetrating?
                # No, if p is at w, it's ON the boundary. 
                # The condition dist < r_target means the circle extends OUTSIDE the unit circle?
                # No. w is on unit circle. dist(p, w) < r means the circle of radius r centered at p 
                # covers the point w. Since w is on the boundary, this means the circle crosses the boundary.
                # So we need to push p AWAY from w, i.e., INWARDS.
                # Vector p - w points from wall to center. 
                # If p is outside the "forbidden zone" (dist > r), no force.
                # If p is inside (dist < r), force is repulsive from w.
                # So force direction is p - w.
                # This pushes p away from w. Since w is on the outer boundary, this pushes p INWARDS.
                # Correct.
                force_vec = np.zeros(2) # If dist is 0, direction is undefined, but unlikely
            forces[i] += force_vec
    
    # Update velocities and positions
    velocities = velocities * damping + forces * dt
    positions = positions + velocities * dt
    
    return positions, velocities

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Wall particles
    n_wall = 720
    angles = np.linspace(0, 2 * np.pi, n_wall, endpoint=False)
    wall_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Initial guess for radius
    # Known best is ~0.1258
    r_current = 0.120
    
    # Initialize
    positions = _init_rings(N)
    velocities = np.zeros_like(positions)
    
    best_score = _score(positions)
    best_positions = positions.copy()
    
    # Dynamics parameters
    dt = 0.01
    damping = 0.9
    k_rep = 100.0
    k_wall = 100.0
    
    # Annealing schedule
    # Start with high repulsion, low damping? 
    # Or start with low repulsion to settle, then increase?
    # Let's try: Start with r_current, run dynamics to minimize overlap.
    
    # We will iterate: 
    # 1. Run dynamics for r_current to find a valid config.
    # 2. Check actual max radius.
    # 3. If actual > r_current, increase r_current and repeat.
    # 4. If stuck, try to polish.
    
    max_iter = 10000
    iter_count = 0
    
    while time.time() < deadline - 20.0: # Leave 20s for polish
        # Run dynamics for a fixed number of steps or until convergence
        # We'll run for 100 steps
        for _ in range(100):
            positions, velocities = _dynamics_step(positions, velocities, r_current, wall_points, dt, damping, k_rep, k_wall)
            iter_count += 1
            
            # Optional: Check score periodically
            if iter_count % 500 == 0:
                s = _score(positions)
                if s > best_score:
                    best_score = s
                    best_positions = positions.copy()
        
        # After 100 steps, check if we have a valid configuration for r_current
        s = _score(positions)
        if s > best_score:
            best_score = s
            best_positions = positions.copy()
        
        # If the current configuration supports a radius larger than r_current, increase it
        if s > r_current + 1e-5:
            # We can increase r_current
            # How much?
            # Be conservative
            r_current = min(s, r_current + 0.002)
            # Reset velocities to help settle with new constraint
            velocities = np.zeros_like(positions)
        else:
            # Stuck. Try to perturb or reduce r? 
            # Or just stop increasing and move to polish.
            # If s is close to r_current, we might be at the limit for this topology.
            # Let's try a small random perturbation to escape local minima
            if np.random.random() < 0.1:
                positions += np.random.normal(0, 0.01, positions.shape)
                velocities = np.zeros_like(positions)
            else:
                # If we are not improving and not stuck in a deep local min (score is reasonable), 
                # we can break and polish.
                if s > 0.12:
                    break
                # Otherwise, keep trying
                r_current = max(0.1, r_current - 0.001)
                velocities = np.zeros_like(positions)
        
        if time.time() > deadline - 20.0:
            break

    # Final Polish using SLSQP
    # Objective: Maximize r
    # Constraints: d_ij >= 2r, |c_i| <= 1-r
    # This is a non-linear programming problem.
    # We can maximize r by minimizing -r subject to constraints.
    
    def objective(x):
        # x is flattened positions
        pos = x.reshape(-1, 2)
        # Calculate current max feasible radius for this pos
        # But we want to optimize pos AND r? 
        # SLSQP can handle multiple variables.
        # Let's optimize pos for a FIXED r? 
        # No, we want to maximize r.
        # Let's define the problem as:
        # Maximize r
        # Subject to:
        # d_ij(pos) - 2r >= 0
        # 1 - |c_i| - r >= 0
        
        # This is a bit tricky with standard minimizers.
        # Alternative: Minimize the "violation" for a target r, and binary search r.
        # But we already did dynamics.
        
        # Let's try to optimize positions to maximize the minimum clearance.
        # Objective: Minimize -min_clearance(pos)
        # min_clearance(pos) = min( min_ij d_ij/2, min_i (1-|c_i|) )
        # This is non-smooth.
        
        # Smooth surrogate:
        # Minimize sum(exp(-beta * d_ij)) + sum(exp(-beta * (1-|c_i|)))
        # This pushes all distances and margins to be large.
        
        pos = x.reshape(-1, 2)
        
        # Inter-circle
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0) # Ignore self
        
        # Boundary
        dists_origin = np.linalg.norm(pos, axis=1)
        margins = 1.0 - dists_origin
        
        beta = 50.0
        # Use log-sum-exp for stability
        # Sum exp(-beta * dists)
        # We want to MINIMIZE this sum, which maximizes the distances.
        
        # To avoid overflow, shift by max
        # But dists are small (~0.2), beta*dists ~ 10. exp(-10) is small.
        
        val_inter = np.sum(np.exp(-beta * dists))
        val_boundary = np.sum(np.exp(-beta * margins))
        
        return val_inter + val_boundary

    # Initial guess
    x0 = best_positions.flatten()
    
    # Bounds: -1 to 1
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Run minimize
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < objective(x0):
            pos_res = res.x.reshape(-1, 2)
            s_res = _score(pos_res)
            if s_res > best_score:
                best_score = s_res
                best_positions = pos_res
    except Exception:
        pass
    
    # Final check
    final_score = _score(best_positions)
    if final_score > best_score:
        best_score = final_score
    
    return best_positions

# EVOLVE-BLOCK-END
