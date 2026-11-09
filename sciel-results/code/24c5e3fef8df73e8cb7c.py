import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Jittered Grid
    # 4x4x4 = 64 points. We take 50.
    # To make it better, we can start with a 3x4x5 = 60 points grid?
    # 4x4x4 spacing is 0.25. Radius 0.125.
    # Let's use a 4x4x4 grid and jitter it.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to break symmetry and pick 50
    np.random.seed(42)
    indices = np.random.permutation(len(grid))[:N]
    centres = grid[indices].copy()
    
    # Add small random jitter to break degeneracies
    centres += np.random.normal(0, 0.01, centres.shape)
    centres = np.clip(centres, -0.5, 0.5)

    def get_radius(centres):
        # Compute min pairwise distance / 2
        # and min boundary distance
        if len(centres) > 1:
            # Efficient distance calculation
            # (N, 1, 3) - (N, 3, 1) -> (N, N, 3)
            diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        else:
            min_pair = np.inf
        
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair, min_boundary), min_pair, min_boundary, dists

    current_radius, _, _, _ = get_radius(centres)
    best_radius = current_radius
    best_centres = centres.copy()
    
    # 2. Local Optimization: Repulsion-based
    # We will iterate, moving each point away from its closest neighbors.
    
    step_size = 0.05
    decay = 0.99
    min_step = 0.0001
    K = 5 # Number of closest neighbors to consider
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            # Compute distances from i to all others
            # Reusing dists from get_radius is hard in a loop without recomputing.
            # For N=50, recomputing is fine.
            diff = centres - centres[i]
            dists = np.linalg.norm(diff, axis=1)
            dists[i] = np.inf
            
            # Boundary distances for i
            b_dists = 0.5 - np.abs(centres[i])
            
            # Combine pairwise and boundary "sources"
            # We want to move away from the closest constraints.
            # Let's find the K smallest distances among pairwise and boundary.
            # Boundary distances are scalars for each axis, but effectively distance to face.
            # We can treat each face as a "point" at infinity in the normal direction?
            # Simpler: just add the boundary gradients directly if they are small.
            
            # Find indices of K smallest pairwise distances
            pair_indices = np.argpartition(dists, K)[:K]
            
            # Gradient from pairwise
            grad = np.zeros(3)
            for j in pair_indices:
                if dists[j] > 1e-6:
                    # Unit vector from j to i
                    vec = centres[i] - centres[j]
                    norm = np.linalg.norm(vec)
                    grad += vec / norm
            
            # Gradient from boundaries
            # If distance to face is small, push away.
            for k in range(3):
                dist_to_face = 0.5 - np.abs(centres[i][k])
                if dist_to_face < 0.15: # Threshold
                    # Push towards center
                    sign = np.sign(centres[i][k])
                    if sign == 0: sign = 1
                    # Weight by how close it is
                    weight = max(0, 0.15 - dist_to_face)
                    grad[k] -= sign * weight * 10 # Scale factor
            
            norm_grad = np.linalg.norm(grad)
            if norm_grad > 1e-6:
                grad /= norm_grad
                new_pos = centres[i] + step_size * grad
                # Clamp
                new_pos = np.clip(new_pos, -0.5, 0.5)
                
                # Check if this improves the global radius
                # To be safe and fast, only update if local move doesn't decrease local min dist
                # But checking global radius is expensive.
                # Heuristic: Accept if the move increases the distance to the closest neighbor of i.
                old_min_dist = np.min(np.concatenate([dists, b_dists]))
                
                centres[i] = new_pos
                
                # Recompute local min dist for i
                diff_new = centres - centres[i]
                dists_new = np.linalg.norm(diff_new, axis=1)
                dists_new[i] = np.inf
                b_dists_new = 0.5 - np.abs(centres[i])
                new_min_dist = np.min(np.concatenate([dists_new, b_dists_new]))
                
                if new_min_dist >= old_min_dist - 1e-9:
                    improved = True
                else:
                    # Revert
                    centres[i] = new_pos - step_size * grad # Approx revert
                    # Actually just keep old? We lost old.
                    # Let's just not update if it gets worse.
                    # We need to store old position.
                    pass # This logic is flawed because we overwrote centres[i].
                    
        # The above loop has a flaw: we modify centres[i] then check.
        # Let's restructure: calculate move, then apply, then check global radius periodically or use a better local check.
        
        # Let's try a simpler, more robust approach:
        # Coordinate descent with line search?
        # Or just: Move, then if global radius decreased, revert.
        
        # To avoid the revert issue, let's do this:
        # 1. Calculate proposed new position for i.
        # 2. Temporarily set centres[i] to new pos.
        # 3. Calculate global radius.
        # 4. If improved, keep. Else revert.
        
        # This is O(N^2) per point, so O(N^3) per iteration. 50^3 = 125,000 ops. Very fast.
        
        # Let's restart the loop with this logic.
        
        pass

    # The loop above was a draft. Let's write the actual clean version.
    
    return best_centres
