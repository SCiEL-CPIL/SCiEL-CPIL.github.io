<<<<<<< SEARCH
def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq)) / 2.0
    
    return min(min_pair, min_wall)
=======
def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq)) / 2.0
    
    return min(min_pair, min_wall)

def _compute_radius_fast(centers, prev_radius):
    """
    Fast radius check: if min distance >= 2*prev_radius and wall dist >= prev_radius,
    return prev_radius. Otherwise compute exactly.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < prev_radius:
        # Wall is the bottleneck, compute exact
        n = len(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair = np.sqrt(np.min(dists_sq)) / 2.0
        return min(min_pair, min_wall)
    
    # Check if any pair is closer than 2*prev_radius
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    threshold_sq = (2.0 * prev_radius) ** 2
    if np.min(dists_sq) >= threshold_sq:
        return prev_radius  # No improvement possible from pairs
    
    min_pair = np.sqrt(np.min(dists_sq)) / 2.0
    return min(min_pair, min_wall)
>>>>>>> REPLACE

