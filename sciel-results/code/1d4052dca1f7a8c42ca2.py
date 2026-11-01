import numpy as np
from scipy.optimize import differential_evolution # Not using DE, but keeping SA. Actually, pure SA is better for control.

N = 300

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 300 non-overlapping circles inside a unit circle using Simulated Annealing.
    """
    # 1. Initialization: Random uniform distribution in the disk
    # This avoids the specific symmetries of the Sunflower spiral.
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Ensure initial validity (though random uniform in disk is always valid)
    # But we might want to start with a slightly smaller effective radius to allow movement.
    # Actually, we just track the current best radius.
    
    def calc_min_radius(pts):
        """
        Computes the maximum common radius for the given centers.
        r = min( min_pair_dist / 2, min_boundary_dist )
        """
        # Distance to boundary
        dists_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(dists_boundary)
        
        # Distance between points
        # Vectorized pairwise distances
        # diff shape: (N, N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_boundary, min_pair)

    current_r = calc_min_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial temperature (scale of moves)
    T_min = 1e-6
    cooling_rate = 0.995
    max_iter = 50000
    # We will try to move a random subset of points in each step to explore the space better.
    # Moving all 300 points randomly is too chaotic. Moving 1-10 points is good.
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        # Select a random number of points to move (1 to 10)
        n_move = rng.integers(1, 11)
        idxs = rng.choice(N, n_move, replace=False)
        
        # Generate random displacements
        # The magnitude of displacement is proportional to T
        # We want moves to be significant enough to jump over barriers but small enough to refine.
        # Scale factor: T is roughly the step size.
        displacements = rng.normal(0, T, size=(n_move, 2))
        
        new_points = points.copy()
        new_points[idxs] += displacements
        
        # Constrain new points to be inside the unit circle
        # If a point moves outside, project it back to the boundary.
        # This ensures validity of the candidate configuration.
        norms = np.linalg.norm(new_points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            new_points[outside] /= norms[outside, np.newaxis]
            # Note: Projection reduces the distance to boundary for that point, 
            # which might lower the score, but it's a valid configuration.
            # Alternatively, we could reject the move if it goes outside, 
            # but projection is often smoother for optimization.
            
        new_r = calc_min_radius(new_points)
        
        # Metropolis Acceptance Criterion
        if new_r > current_r:
            # Accept improvement
            current_r = new_r
            points = new_points
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Accept worse move with probability exp((new_r - current_r) / T)
            # Note: We are maximizing, so delta = new_r - current_r is negative.
            # Probability is exp(delta / T)
            delta = new_r - current_r
            if delta < 0:
                accept_prob = np.exp(delta / T)
                if rng.random() < accept_prob:
                    current_r = new_r
                    points = new_points
            # If not accepted, keep current points
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            break
            
    # Final Polish: A few iterations of local hill-climbing on the best found
    # to squeeze out any last gains.
    points = best_points.copy()
    current_r = best_r
    step_size = 0.001
    for _ in range(2000):
        # Find bottleneck pair or boundary
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        min_dist_val = dists[i, j]
        
        dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
        min_boundary_idx = np.argmin(dists_boundary)
        min_boundary_val = dists_boundary[min_boundary_idx]
        
        if min_dist_val / 2.0 < min_boundary_val:
            vec = points[i] - points[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                points[i] += direction * step_size * 0.5
                points[j] -= direction * step_size * 0.5
        else:
            idx = min_boundary_idx
            vec = -points[idx]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                direction = vec / norm
                points[idx] += direction * step_size
                
        # Project back
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
            
        new_r = calc_min_radius(points)
        if new_r > current_r:
            current_r = new_r
            best_r = current_r
            best_points = points.copy()
        else:
            # Revert if no improvement? Or just stop? 
            # Simple hill climbing reverts if no improvement.
            # But we might get stuck. Let's just break if no improvement for a while.
            # For simplicity, we just accept if better, else revert.
            points = best_points.copy()
            break # Or continue with smaller step? Let's break to save time.
            
    return best_points
# EVOLVE-BLOCK-END
