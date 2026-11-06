"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initialization: Hexagonal lattice
    # We want to pack N=300 circles in a unit circle.
    # Area of unit circle is pi. Area of one circle is pi*r^2.
    # 300 * pi * r^2 < pi => r < 1/sqrt(300) ~ 0.0577
    # Hex packing density is ~0.9069.
    # So r_eff ~ 0.0577 * sqrt(0.9069) ~ 0.0549
    # Let's start with a slightly smaller radius to ensure valid placement, e.g., r=0.05
    # Spacing d = 2*r.
    
    def generate_hex_lattice(r):
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        # Range of indices to cover the unit circle
        # Max extent is 1.0
        k = int(1.0 / d) + 2
        
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure the circle fits inside the unit circle
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        
        # If we have more than N points, select N points that are "best" distributed
        # Sorting by distance from center and taking the first N is a common heuristic
        # but might lead to a "clumped" center.
        # A better heuristic for initial state is to just take a random subset or
        # the ones closest to the center to ensure they are well within bounds.
        # Given the lattice structure, taking the central ones is safe.
        
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            # If not enough points, we need to adjust r down or just return what we have
            # but construct() must return N points.
            # The loop in construct() will handle adjusting r.
            return points

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        r = min( min_{i<j} dist(i,j)/2, min_i (1 - dist(i, origin)) )
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Distance to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Pairwise distances
        # Using a loop or vectorized calculation.
        # Vectorized:
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # This is O(N^2) memory and compute. N=300 -> 90k pairs. Very fast.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    # Find a good initial radius
    # Start with a guess and binary search or just step down until valid
    r_guess = 0.055
    points = None
    
    # Try to find a valid configuration with r_guess
    # If hex lattice doesn't yield N points, reduce r
    # Actually, generate_hex_lattice returns points that fit in the circle for radius r.
    # If it returns < N points, we need a smaller r (larger spacing? No, smaller r means larger spacing? No.)
    # Wait.
    # If r is large, spacing d=2r is large. Fewer points fit in the unit circle.
    # If r is small, spacing d=2r is small. More points fit.
    # So if we have < N points, we need a SMALLER r to get MORE points.
    # If we have >= N points, we have a valid configuration for radius r.
    # We want to MAXIMIZE r.
    
    # So, let's start with a small r that definitely fits N points, then try to increase?
    # Or start with a large r, if it fails (not enough points), decrease r until we get N points.
    # Then we have a valid packing with radius r.
    # Then we optimize to increase r.
    
    # Let's find the largest r for which the hex lattice yields at least N points.
    # This is a monotonic property: smaller r -> more points.
    # So we can binary search for the threshold r_hex.
    
    r_low = 0.01
    r_high = 0.08
    
    # Ensure r_low gives enough points
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    # Binary search for max r in hex lattice
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    # r_low is the largest radius for which hex lattice works directly.
    points = generate_hex_lattice(r_low)
    
    # If we still don't have N points (shouldn't happen), fallback to random
    if len(points) < N:
         # This is a safety net
         points = np.random.rand(N, 2) * 2 - 1
         # Scale to be inside
         dists = np.hypot(points[:,0], points[:,1])
         points = points / np.maximum(dists, 1e-6)[:, np.newaxis] * (dists / np.maximum(dists, 1e-6)) # This is wrong
         # Just random inside unit circle
         angles = np.random.rand(N) * 2 * np.pi
         radii = np.sqrt(np.random.rand(N))
         points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
         # Scale down to ensure valid
         # Just use a very small radius
         points = points * 0.5

    # Ensure we have exactly N points
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    # 2. Optimization
    # We will use a simple local search / gradient ascent hybrid.
    # Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    # This is a non-smooth function.
    # We can use a "relaxation" approach:
    # In each step, calculate forces.
    # - If a pair is too close (dist < 2r), push them apart.
    # - If a point is too close to boundary (dist > 1-r), push it in.
    # - If a point is "free", maybe nudge it randomly or towards center to allow others to expand?
    # Actually, to maximize the MINIMUM distance, we should push ALL pairs apart that are close to the minimum,
    # and push boundary-violating points in.
    
    # Let's use a simulated annealing style approach or just a greedy hill climbing with random restarts.
    # Given the time limit, a few thousand iterations of a simple local search should be sufficient.
    
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # Precompute some constants
    current_r = best_r
    
    # Optimization loop
    # We will try to increase the radius.
    # Strategy:
    # 1. Calculate current r.
    # 2. Identify "active" constraints: pairs with dist < 2*(r + delta) and points with 1-dist < (r + delta).
    # 3. Apply forces to resolve these constraints.
    # 4. Update positions.
    # 5. Check if new r > old r. If so, keep.
    
    # A simpler, more robust approach for this specific problem:
    # Use a force-based relaxation where we try to expand the "Voronoi" cells or simply
    # push overlapping circles apart and boundary-violating circles in.
    
    # Let's implement a simple iterative improvement:
    # 1. Compute current r.
    # 2. For each point, calculate the "limiting" factor.
    # 3. Move points in the direction that increases the minimum distance.
    
    # Alternative: Random perturbations + acceptance if r increases.
    # This is robust but might be slow.
    
    # Let's try a combination:
    # 1. Start with hex lattice.
    # 2. Iterate:
    #    a. Compute r.
    #    b. Find the pair (i,j) with minimum distance d_ij.
    #    c. Find the point k with minimum boundary distance d_k.
    #    d. If d_ij/2 < d_k, try to increase d_ij by moving i and j apart.
    #       Else, try to increase d_k by moving k towards center.
    #    e. Also, occasionally do random moves to escape local minima.
    
    # This is a bit complex. Let's stick to the force-based relaxation which is standard for circle packing.
    
    forces = np.zeros_like(points)
    step_size = 0.01
    
    for it in range(10000):
        if time.time() > deadline:
            break
            
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        bound_dists = 1.0 - radii
        
        current_r = min(np.min(dists) / 2.0, np.min(bound_dists))
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        # If we are close to the theoretical max or time is up, stop
        # Theoretical max is ~0.0577. Best known is 0.0531.
        # If best_r > 0.0535, we are doing well.
        
        # Calculate forces
        # We want to maximize the minimum distance.
        # So we push apart pairs that are close to the minimum, and pull in points close to the boundary.
        
        # Define a threshold for "active" constraints
        # Let's use a soft threshold.
        # If dist_ij < 2 * current_r * 1.1, apply repulsive force.
        # If bound_dist_k < current_r * 1.1, apply attractive force to center.
        
        # This is tricky because current_r changes.
        # Let's use a fixed small threshold relative to the current minimum.
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        # Active pairs: dist < min_pair_dist * 1.05
        # Active points: bound_dist < min_bound_dist * 1.05
        
        # Actually, a better heuristic is to apply forces to ALL pairs, with strength decaying with distance.
        # Force ~ 1/d^2 or similar.
        # But we must respect the boundary.
        
        # Let's use a simpler "push apart" scheme.
        # For each pair (i,j), if dist < 2 * target_r, push apart.
        # What is target_r? We want to increase r.
        # Let's set target_r = current_r * 1.001 (small increase).
        # Or just use the current minimums.
        
        # Let's try to move points to increase the minimum distance.
        # Gradient of min distance is not well-defined.
        
        # Let's use the "Repulsion" method:
        # F_ij = (1/d^2) * (p_i - p_j) / d  for d < 2r_target
        # F_boundary = -p_k / |p_k|  for |p_k| > 1 - r_target
        
        # Let's pick r_target = best_r * 1.01
        # This encourages expansion.
        
        r_target = best_r * 1.005
        
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        # Only consider pairs with dist < 2 * r_target * 1.5 to save time
        # N=300, N^2/2 = 45k pairs. Vectorized is fast enough.
        
        # Re-use dists and diff
        # Mask for active pairs
        mask = dists < 2.0 * r_target * 1.2
        
        # Apply forces
        # F_i = sum_{j != i, mask[i,j]} (p_i - p_j) / dists[i,j]^2
        
        # Vectorized force calculation is a bit tricky with the mask.
        # Let's do it in a loop for clarity and safety, or use scipy?
        # Loop over N=300 is fast.
        
        for i in range(N):
            # Boundary force
            if radii[i] > 1.0 - r_target:
                # Push towards center
                # Strength proportional to violation
                violation = radii[i] - (1.0 - r_target)
                if violation > 0:
                    forces[i] -= (points[i] / radii[i]) * violation * 10.0
            
            # Pairwise forces
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2.0 * r_target * 1.2:
                    # Repulsion
                    # Force magnitude ~ 1/d^2
                    # Direction: away from each other
                    diff_ij = points[i] - points[j]
                    # Normalize and scale
                    if d > 1e-6:
                        force_mag = 1.0 / (d * d)
                        # Scale down to avoid explosion
                        force_vec = diff_ij * force_mag * 0.01
                        forces[i] += force_vec
                        forces[j] -= force_vec
                        # Wait, if d < 2*r_target, they are "overlapping" relative to target.
                        # We want to push them apart.
                        # The force should be stronger if they are closer.
        
        # Update points
        points += forces * step_size
        
        # Ensure points stay inside the unit circle (hard constraint)
        # If a point goes outside, project it back
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis]
        
        # Decay step size
        step_size *= 0.999
        
        # Occasionally, do a random restart or perturbation to escape local minima
        if it % 100 == 0:
            # Check if we are stuck
            # If best_r hasn't improved in last 100 iters, perturb
            pass # Simplified for now

    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy() # Already a copy

    return best_points
# EVOLVE-BLOCK-END
