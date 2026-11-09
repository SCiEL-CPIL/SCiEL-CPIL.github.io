# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Initialize with a geometrically informed configuration (Icosahedron + 2 points).
    2. Perform a local search (gradient-free, coordinate-wise or small step) to maximize (d_min/d_max)^2.
    3. Use a deadline guard to ensure termination before timeout.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    deadline = start_time + 550  # 550s limit, leaving 50s margin for safety
    
    # --- Initialization: Geometric Construction ---
    # The optimal packing for 12 points in 3D is an icosahedron.
    # We start with an icosahedron and add 2 points, likely on the axis or symmetrically.
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (normalized to unit sphere initially)
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize icosahedron to have max distance 1 (or some reference)
    # Actually, let's just normalize to unit radius first, then scale later if needed.
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, np.newaxis]
    
    # Add 2 points. A good guess for the 13th and 14th points in 3D packing
    # often involves breaking the symmetry slightly or placing them along high-symmetry axes.
    # Let's try placing them along the Z-axis, slightly outside or inside?
    # Or maybe just perturb the icosahedron.
    # A known good configuration for 14 points might be related to the icosahedron
    # but with 2 vertices moved.
    
    # Let's start with the icosahedron points plus 2 points at (0,0,1) and (0,0,-1)?
    # Wait, (0,0,1) is close to (0,1,phi) normalized.
    # Let's just take the first 12 as icosahedron and initialize the last 2 randomly
    # but close to the surface, then let the optimizer fix them.
    # Better: Initialize all 14 on the sphere, then let them move in 3D.
    
    # To be safe and fast, we'll use a deterministic local search.
    # Objective: Maximize (d_min / d_max)^2.
    # This is equivalent to maximizing d_min / d_max.
    
    points = np.zeros((n, d))
    points[:12] = ico_verts
    
    # Initialize last 2 points
    # Let's try placing them at the poles of the icosahedron?
    # The icosahedron has 5-fold symmetry axes.
    # Let's just place them at (0,0,1) and (0,0,-1) and let the search move them.
    points[12] = [0, 0, 1]
    points[13] = [0, 0, -1]
    
    # Normalize the entire set to have unit max distance?
    # The metric is scale-invariant. We can fix the scale to reduce search space.
    # Let's fix d_max = 1 by scaling.
    def normalize(pts):
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max == 0:
            return pts
        return pts / d_max
    
    points = normalize(points)
    
    best_points = points.copy()
    best_score = 0.0
    
    def get_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio # (d_min/d_max)^2
    
    best_score = get_score(best_points)
    
    # --- Local Search ---
    # We will use a simple iterative improvement:
    # Pick a random point, try moving it in small random directions, keep if better.
    # This is effectively a random hill climbing.
    
    rng = np.random.default_rng(42)
    
    # Step size schedule
    step_size = 0.1
    min_step = 1e-6
    
    # Time check helper
    def time_exceeded():
        return time.time() > deadline
    
    # Iteration loop
    while not time_exceeded():
        # Pick a random point to move
        idx = rng.integers(0, n)
        
        # Generate a random direction
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        
        # Try a few step sizes
        found_improvement = False
        for _ in range(10):
            new_point = best_points[idx].copy()
            # Random step magnitude
            magnitude = step_size * rng.uniform(0.1, 1.0)
            new_point += magnitude * direction
            
            candidate = best_points.copy()
            candidate[idx] = new_point
            
            # Optional: Re-center? Translation invariance means we can fix the center of mass to 0.
            # This reduces the search space by 3 dimensions.
            candidate -= np.mean(candidate, axis=0)
            
            # Optional: Re-normalize? Scale invariance.
            # We can fix the max distance to 1.
            candidate = normalize(candidate)
            
            new_score = get_score(candidate)
            
            if new_score > best_score:
                best_score = new_score
                best_points = candidate
                found_improvement = True
                # Shrink step size slightly on success? Or keep?
                # Let's keep step size constant for a while, then decay.
                break
        
        if not found_improvement:
            # Decay step size
            step_size *= 0.95
            if step_size < min_step:
                # Try a different random restart?
                # Or just stop if we are stuck.
                # Given the time limit, we might just break if step is very small.
                if step_size < 1e-8:
                    break
    
    # Final check: ensure finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to icosahedron + 2 points
        best_points = points
        
    return best_points

# EVOLVE-BLOCK-END
