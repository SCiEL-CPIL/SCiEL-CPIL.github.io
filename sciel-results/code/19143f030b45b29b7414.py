import numpy as np
import time

N = 50

def compute_radius(points):
    """
    Computes the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary: min(1 - |c_i|)
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Minimum distance between centers
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # The radius is limited by half the minimum distance between centers
    # and the minimum distance to the boundary
    r = min(min_dist / 2.0, min_boundary)
    
    return r if r > 0 else 0.0

def initial_hexagonal():
    """
    Generates an initial set of points using a hexagonal grid pattern.
    This is a much better structural prior than random or simple rings
    for maximizing minimum distances in a confined space.
    """
    points = np.zeros((N, 2))
    
    # Generate a grid of hexagonal points
    # We want to cover the unit circle area roughly.
    # Area of unit circle is pi. 50 circles.
    # Approx radius of small circles ~ sqrt(pi/50) ~ 0.25 (diameter 0.5).
    # Hex grid spacing s ~ 0.5.
    
    # Generate points in a hex grid
    a = 0.55 # spacing
    h = a * np.sqrt(3) / 2
    
    candidates = []
    # Range covering the unit circle
    range_x = int(1.0 / a) + 2
    range_y = int(1.0 / h) + 2
    
    for i in range(-range_x, range_x + 1):
        for j in range(-range_y, range_y + 1):
            x = i * a + (0.5 * a if j % 2 != 0 else 0)
            y = j * h
            candidates.append([x, y])
            
    candidates = np.array(candidates)
    
    # Keep only points inside the unit circle (with a margin)
    norms = np.linalg.norm(candidates, axis=1)
    candidates = candidates[norms < 0.95]
    
    # If we have too many, pick a random subset? 
    # Or just take the first N? 
    # Better: Shuffle and take N to avoid bias towards origin?
    # Actually, for packing, we want them spread out.
    # Let's just take the first N from the list, but the list is ordered by grid.
    # To ensure good coverage, let's randomize the order of valid candidates 
    # and pick N that are "well spread"? 
    # Simpler: Just take N random ones from the valid candidates.
    if len(candidates) >= N:
        indices = np.random.choice(len(candidates), N, replace=False)
        points = candidates[indices]
    else:
        # Fallback if not enough candidates (should not happen with a=0.55)
        # Fill remaining with random points
        points = candidates
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.9, 0.9, (remaining, 2))
        points = np.vstack([points, random_pts])
        
    return points

def energy(points, target_r):
    """
    Calculate the energy of the configuration.
    Energy is the sum of overlaps.
    We want to minimize this.
    Overlap between i, j: max(0, 2*target_r - dist_ij)
    Boundary violation: max(0, |p_i| + target_r - 1)
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise overlaps
    # We sum the excess overlap
    overlaps = np.maximum(0, 2 * target_r - dists)
    # Sum over unique pairs (upper triangle) to avoid double counting
    # Or just sum all and divide by 2?
    # Let's sum all and divide by 2 for simplicity, or use triu.
    # Using triu indices for speed? No, vectorized sum is fine.
    pair_energy = 0.5 * np.sum(overlaps)
    
    # Boundary violations
    norms = np.linalg.norm(points, axis=1)
    bound_violations = np.maximum(0, norms + target_r - 1.0)
    bound_energy = np.sum(bound_violations)
    
    return pair_energy + bound_energy

def anneal(points, deadline):
    """
    Simulated Annealing loop.
    """
    current_pts = points.copy()
    current_r = compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Initial temperature
    T = 0.1
    T_min = 1e-5
    alpha = 0.999 # Cooling rate
    
    # Target radius for energy calculation
    # We want to push for a radius larger than current best
    target_r = max(current_r * 1.05, 0.12)
    
    steps = 0
    while time.time() < deadline:
        # Generate a move: pick a random point and move it randomly
        i = np.random.randint(N)
        # Gaussian step
        step_size = 0.05 * T + 0.001
        move = np.random.normal(0, step_size, 2)
        
        new_pts = current_pts.copy()
        new_pts[i] += move
        
        # Check if new point is way outside
        if np.linalg.norm(new_pts[i]) > 1.5:
            continue
            
        # Calculate energy change
        # For speed, we can approximate delta energy by only checking neighbors
        # But for N=50, full energy calc is ~50^2 = 2500 ops. 
        # 150s / 2500 ops is huge. We can do many steps.
        # Let's optimize energy calc to be faster if possible, but N=50 is small.
        
        E_old = energy(current_pts, target_r)
        E_new = energy(new_pts, target_r)
        
        delta_E = E_new - E_old
        
        # Acceptance probability
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            current_pts = new_pts
            
            # Update best
            # Checking radius every step is expensive (sqrt of 2500 pairs)
            # Let's check every 10 steps
            if steps % 10 == 0:
                r = compute_radius(current_pts)
                if r > best_r:
                    best_r = r
                    best_pts = current_pts.copy()
                    # If we found a better radius, we can increase target_r slightly?
                    # Or keep target_r fixed to push harder?
                    # Keeping it fixed is safer for annealing stability.
                    # But if we are stuck, maybe increase target to force expansion?
                    # Let's stick to fixed target_r for now.
        
        # Cool down
        T *= alpha
        
        # If T is too low, restart with a higher T if stuck?
        # Simple restart strategy:
        if T < T_min:
            # If we haven't improved in a while, reset T
            # Or just break and let the outer loop handle restarts
            T = T_min
            
        steps += 1
        
        # Periodic check for time
        if steps % 1000 == 0:
             pass # Just a marker
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Strategy:
    # 1. Initialize with hexagonal grid.
    # 2. Run Simulated Annealing to optimize.
    # 3. If time permits, restart from best found with perturbation.
    
    pts = initial_hexagonal()
    current_r = compute_radius(pts)
    
    if current_r > best_overall_r:
        best_overall_r = current_r
        best_overall_points = pts.copy()
        
    # Run Annealing
    # We will run annealing in chunks to allow for restarts if stuck
    # Or just one long run?
    # One long run with adaptive temperature is better.
    
    # Let's do a few iterations of "Anneal then Polish"
    # Actually, SA itself is the polish.
    
    # To improve the chance of escaping local minima, we can run SA multiple times
    # with different initial conditions or restarts.
    
    # Run 1: From hexagonal
    pts, r = anneal(pts, deadline - 10) # Leave 10s for safety
    if r > best_overall_r:
        best_overall_r = r
        best_overall_points = pts.copy()
        
    # Run 2: From a different initial guess (e.g. rings) if time permits
    if time.time() < deadline - 20:
        # Generate ring initial guess
        points_rings = np.zeros((N, 2))
        points_rings[0] = [0, 0]
        count = 1
        rings = [
            (0.25, 6),
            (0.50, 12),
            (0.75, 18),
            (0.95, 14)
        ]
        for radius, capacity in rings:
            for k in range(capacity):
                if count >= N:
                    break
                angle = 2 * np.pi * k / capacity
                points_rings[count] = [radius * np.cos(angle), radius * np.sin(angle)]
                count += 1
            if count >= N:
                break
        
        r_rings = compute_radius(points_rings)
        if r_rings > best_overall_r:
             best_overall_r = r_rings
             best_overall_points = points_rings.copy()
             
        # Anneal from rings
        pts, r = anneal(points_rings, deadline - 5)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_points = pts.copy()
            
    # Run 3: From best found, with perturbation
    if time.time() < deadline - 5 and best_overall_points is not None:
        perturbed = best_overall_points + np.random.normal(0, 0.01, (N, 2))
        # Project back to circle
        norms = np.linalg.norm(perturbed, axis=1)
        out_mask = norms > 0.98
        if np.any(out_mask):
            perturbed[out_mask] = perturbed[out_mask] / norms[out_mask, np.newaxis] * 0.98
            
        pts, r = anneal(perturbed, deadline - 2)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_points = pts.copy()
            
    return best_overall_points if best_overall_points is not None else initial_hexagonal()

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
