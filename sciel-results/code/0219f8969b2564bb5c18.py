import numpy as np
import time

N = 300

def construct():
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing for 300 circles in unit circle
    # Area ~ pi, circle area ~ pi r^2. 300 * pi r^2 ~ pi -> r ~ 1/sqrt(300) ~ 0.057
    # Spacing ~ 2r ~ 0.114
    # Let's generate a grid and filter
    r_init = 0.05
    spacing = 2.0 * r_init
    points = []
    # Generate points in a square covering the unit circle
    k = int(1.0 / spacing) + 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    # Keep only points inside the circle (with some margin for radius)
    dists = np.hypot(points[:, 0], points[:, 1])
    points = points[dists <= 1.0 - r_init]
    
    # If we have more than N, select the N closest to the center to form a compact cluster
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        # Fallback: Random init if lattice is too sparse (unlikely with this spacing)
        points = np.random.uniform(-0.9, 0.9, (N, 2))
    
    # Ensure we have exactly N points
    if len(points) < N:
        # Pad with random points near center if necessary
        extra = N - len(points)
        new_pts = np.random.uniform(-0.5, 0.5, (extra, 2))
        points = np.vstack((points, new_pts))
    else:
        points = points[:N]
        
    # Shuffle to break symmetry? No, hex is symmetric. 
    # Let's add a tiny bit of noise to break perfect symmetry and help escape local minima
    points += np.random.uniform(-1e-5, 1e-5, points.shape)

    best_r = 0.0
    current_r = 0.01
    
    # Optimization Loop
    # We will iteratively try to increase current_r.
    # For each r, we relax the system until it's valid or we run out of steps/time.
    
    step_size = 0.01
    max_relax_steps = 500
    
    while time.time() < deadline:
        # Check if current configuration is valid for current_r
        # Calculate min distance between pairs
        # This is O(N^2). N=300 -> 90k pairs. Fast enough in NumPy.
        
        # Vectorized distance calculation
        # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # np.fill_diagonal(dists, np.inf)
        # min_dist = np.min(dists)
        
        # More memory efficient for 300x300:
        # Actually 300x300x2 is 180KB. 300x300 is 90KB. Very small.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distance
        radii = np.hypot(points[:, 0], points[:, 1])
        min_wall_dist = np.min(1.0 - radii)
        
        current_min_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_min_r > best_r:
            best_r = current_min_r
            
        # If valid for current_r, try to increase it
        if current_min_r >= current_r:
            # Valid. Increase target radius
            current_r *= 1.01
            step_size *= 0.95  # Slow down as we get closer to optimum
        else:
            # Invalid. Apply forces to fix violations
            
            # Calculate forces
            # Pair forces
            # We want to push apart if dist < 2 * current_r
            # Force direction: diff / dist
            # Magnitude: proportional to violation
            
            # Avoid division by zero
            safe_dists = np.where(dists < 1e-8, 1e-8, dists)
            
            # Only consider pairs that are too close
            violation = np.maximum(0, 2.0 * current_r - dists)
            
            # Force vector for each pair (i,j)
            # F_ij = violation * (diff / dist)
            # Sum forces on i from all j
            
            # This is tricky to vectorize efficiently without huge memory.
            # 300x300x2 is fine.
            unit_vec = diff / safe_dists[:, :, np.newaxis]
            
            # Force contribution from pair (i,j) on particle i
            # F_i += violation_ij * unit_vec_ij
            forces = np.sum(violation[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Wall forces
            # If radius > 1 - current_r, push in
            wall_violation = np.maximum(0, radii - (1.0 - current_r))
            # Direction: -c_i / |c_i|
            safe_radii = np.where(radii < 1e-8, 1e-8, radii)
            wall_forces = -wall_violation[:, np.newaxis] * (points / safe_radii[:, np.newaxis])
            
            total_forces = forces + wall_forces
            
            # Update positions
            points += step_size * total_forces
            
            # Keep points inside unit circle (hard constraint to prevent explosion)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                points[mask] /= radii[mask, np.newaxis]
                
            # If stuck (small movement), add noise (Simulated Annealing style)
            # Simple heuristic: if best_r hasn't improved in a while, add noise
            # For simplicity, let's just rely on the force relaxation.
            
            # Reduce step size slightly if we are oscillating? 
            # The step_size *= 0.95 above handles gradual convergence.
            # If we are invalid, we keep step size to allow movement.
            
            # Check time
            if time.time() > deadline:
                break

    # Final validation and cleanup
    # Ensure no points are outside
    radii = np.hypot(points[:, 0], points[:, 1])
    points = points[radii <= 1.0]
    
    # If we lost points due to clipping (shouldn't happen with hard constraint), pad
    if len(points) < N:
        # This is a failure case, but better than invalid shape
        # Fill with center points? No, they will overlap.
        # Just return what we have? No, shape must be (N,2).
        # Pad with random points near center, hoping they fit in gaps?
        # Unlikely to improve score if we are in this state.
        # Let's just repeat the best point? No, coincident centers invalid.
        # Let's just return the points we have, padded with zeros?
        # The task says "wrong shape ... makes candidate invalid".
        # So we must return (N,2).
        # If we lost points, we are in trouble.
        # But the hard constraint `points[mask] /= ...` prevents going outside.
        # So len(points) should be N.
        
        # Just in case:
        if len(points) < N:
             extra = N - len(points)
             new_pts = np.zeros((extra, 2))
             points = np.vstack((points, new_pts))

    return points
