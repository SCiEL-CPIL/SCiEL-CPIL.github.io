"""Improved packing for 100 equal spheres in a unit cube."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a packing of N spheres in a unit cube using a greedy 
    placement strategy followed by local search refinement.
    """
    # Use a fixed seed for reproducibility and stability
    np.random.seed(42)
    
    # Step 1: Greedy Construction
    # Place spheres one by one at the location that maximizes the minimum distance
    # to existing spheres and the walls.
    
    centers = []
    M = 5000 # Number of random candidates to test for each new sphere
    
    for i in range(N):
        # Generate M random points in the cube [-0.5, 0.5]^3
        candidates = np.random.uniform(-0.5, 0.5, (M, 3))
        
        # Calculate distance to walls for each candidate
        # Distance to nearest wall is min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
        wall_clearance = np.min(0.5 - np.abs(candidates), axis=1)
        
        if i == 0:
            # For the first sphere, the center is optimal
            centers.append(np.array([0.0, 0.0, 0.0]))
            continue
            
        # Calculate distances to existing centers
        existing = np.array(centers)
        # candidates: (M, 3), existing: (i, 3)
        # dists: (M, i)
        diffs = candidates[:, np.newaxis, :] - existing[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Minimum distance to any existing sphere for each candidate
        min_dist_to_existing = np.min(dists, axis=1)
        
        # The effective radius for this candidate is min(min_dist_to_existing / 2, wall_clearance)
        supported_radius = np.minimum(min_dist_to_existing / 2.0, wall_clearance)
        
        # Select the candidate with the maximum supported radius
        best_idx = np.argmax(supported_radius)
        centers.append(candidates[best_idx])
        
    centers = np.array(centers)
    
    # Step 2: Local Search Refinement
    # Try to increase the minimum radius by moving spheres.
    
    def get_radius(pts):
        # pts: (N, 3)
        # Check validity (should be guaranteed by construction, but safe to check)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return -1.0
        # Distances to walls
        wall_clearances = np.min(0.5 - np.abs(pts), axis=1)
        min_wall = np.min(wall_clearances)
        
        # Pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        return min(min_pair / 2.0, min_wall)
    
    best_centers = centers.copy()
    best_radius = get_radius(best_centers)
    
    # Local search parameters
    T = 5000 # Iterations
    step_size = 0.05
    min_step_size = 0.001
    
    for t in range(T):
        # Anneal step size
        step = step_size * (min_step_size / step_size) ** (t / T)
        
        # Pick a random sphere to move
        idx = np.random.randint(N)
        
        # Generate a random perturbation
        perturbation = np.random.randn(3) * step
        new_center = best_centers[idx].copy()
        new_center += perturbation
        
        # Clip to cube
        new_center = np.clip(new_center, -0.4999, 0.4999)
        
        # Evaluate new configuration
        temp_centers = best_centers.copy()
        temp_centers[idx] = new_center
        
        new_radius = get_radius(temp_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = temp_centers
    
    return best_centers
# EVOLVE-BLOCK-END
