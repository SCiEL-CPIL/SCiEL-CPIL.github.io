"""Rewritten program: structural change to force-directed packing with multi-start annealing."""
import numpy as np
import time

N = 300
DEADLINE_SECONDS = 115.0  # safety margin before 120s kill


def _score(points):
    """Compute the maximum common radius for the given centres."""
    n = len(points)
    if n == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    # Pairwise distances
    # Use a chunked approach to avoid huge memory usage if N were larger, but 300 is small
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r = min(min_pair_dist / 2.0, np.min(dist_boundary))
    return r


def _force_step(points, dt, temperature):
    """
    Perform a single force-directed step.
    Forces:
    1. Repulsion between all pairs if distance < 2*r_target (or just always repel with inverse square)
    2. Attraction to center to keep them inside (soft boundary)
    3. Hard constraint projection to unit circle
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Repulsion: We want to maximize the minimum distance.
    # A common heuristic is to apply a force proportional to 1/d^2 or similar.
    # To make it effective for packing, we focus on close pairs.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)  # Avoid div by zero
    
    # We use a soft repulsion. If dist is small, force is large.
    # Let's use a force that pushes apart if dist < 2.0 (arbitrary scale, but relative matters)
    # Actually, for packing, we want to maximize the minimum distance.
    # Gradient ascent on min distance is hard.
    # Instead, use a potential energy U = sum 1/d^2. Minimizing U increases distances.
    # Force = -grad U = sum 2 * diff / d^4
    
    # Mask to ignore very large distances to save computation? No, 300x300 is fine.
    # But 1/d^4 can explode if d is tiny.
    # Let's use a safer repulsion: F = diff / (d^2 + eps)
    # Or better, just push apart with a constant force if overlapping?
    
    # Let's try: Force on i from j is (p_i - p_j) / (d_ij^2) * strength
    # This is a standard Coulomb-like repulsion.
    
    # Compute forces vectorized
    # dists is (N, N)
    # We need to sum over j for each i
    # F_i = sum_j (p_i - p_j) / d_ij^2
    # diff is (N, N, 2). diff[i,j] = p_i - p_j
    # We want to divide each vector by d_ij^2
    inv_d2 = 1.0 / (dists**2 + 1e-8)
    inv_d2[np.diag_indices_from(inv_d2)] = 0.0
    
    # Force contribution from repulsion
    repulsive_force = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Attraction to center to prevent drift and help packing near boundary?
    # Actually, for packing in a circle, we want to fill the space.
    # A slight attraction to the center might help cluster, but we want uniform distribution.
    # Let's add a weak attraction to the center to keep them from flying apart, 
    # but the repulsion should dominate for local packing.
    # Alternatively, just let repulsion push them out, and boundary constraint keeps them in.
    # If we only have repulsion, they will expand until they hit the boundary.
    # This is good.
    
    # Add a small random noise for annealing
    noise = np.random.normal(0, temperature, size=points.shape)
    
    # Update positions
    points = points + dt * (repulsive_force + noise)
    
    # Constrain to unit circle
    radii = np.linalg.norm(points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        points[mask] = points[mask] / radii[mask][:, np.newaxis]
    
    return points


def _construct_single_start(seed):
    """Construct a packing starting from a specific seed."""
    np.random.seed(seed)
    
    # Initialize with a Fibonacci sphere/disk distribution
    # This provides a good initial spread.
    points = np.zeros((N, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N)
        theta = i * golden_angle
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
    
    # Annealing schedule
    start_temp = 0.1
    end_temp = 0.001
    steps = 2000
    dt = 0.01
    
    current_points = points.copy()
    best_points = current_points.copy()
    best_score = _score(best_points)
    
    for step in range(steps):
        # Linear cooling
        t = step / steps
        temp = start_temp * (1 - t) + end_temp * t
        
        current_points = _force_step(current_points, dt, temp)
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
    return best_points, best_score


def construct():
    start_time = time.time()
    
    # Try multiple seeds
    # We have ~115 seconds.
    # Each start takes ~0.1-0.2s for 2000 steps?
    # Let's estimate. 300x300 matrix ops are fast.
    # 2000 steps * ~10us/step = 20ms.
    # So we can do ~100 starts in 2 seconds.
    # Let's do 50 starts first, then refine the best one.
    
    best_points = None
    best_score = 0.0
    
    num_starts = 10
    for i in range(num_starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 10:
            break
        
        pts, score = _construct_single_start(i)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Refine the best point with more steps and lower temperature
    if best_points is not None:
        np.random.seed(42) # Deterministic refinement
        points = best_points.copy()
        start_temp = 0.01
        end_temp = 0.0001
        steps = 5000
        dt = 0.005
        
        for step in range(steps):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE_SECONDS - 2:
                break
            
            t = step / steps
            temp = start_temp * (1 - t) + end_temp * t
            points = _force_step(points, dt, temp)
            
            # Check score periodically to save time? No, 300 points is fast.
            current_score = _score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        
        # Final check
        final_score = _score(best_points)
        if final_score > best_score:
            best_score = final_score

    return best_points


# Quick self-test if run directly
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Shape: {pts.shape}")
