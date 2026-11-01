def _gradient(centers):
    N = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    abs_centers = np.abs(centers)
    max_coord = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_coord
    
    min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
    min_wall_dist = np.min(wall_dists)
    
    # Mask for active pairs
    eps = 1e-4
    active_mask = dist < (min_pair_dist + eps)
    np.fill_diagonal(active_mask, False)
    
    grad = np.zeros_like(centers)
    
    if np.any(active_mask):
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-9, dist, 1.0)
        # v / d
        unit_vecs = diff / safe_dist[..., np.newaxis]
        # Mask out inactive
        unit_vecs *= active_mask[..., np.newaxis]
        
        # Sum over j for each i
        # grad[i] += sum_j 0.5 * unit_vecs[i, j]
        grad += 0.5 * np.sum(unit_vecs, axis=1)
        
        # Note: unit_vecs[i, j] = (c_i - c_j) / d
        # So sum_j unit_vecs[i, j] is the force on i.
        # This is correct.
        
    # Wall gradient
    active_wall_mask = wall_dists < (min_wall_dist + eps)
    if np.any(active_wall_mask):
        # Find limiting coordinate for each active wall point
        # k = argmax(abs_centers[i])
        limiting_coords = np.argmax(abs_centers, axis=1)
        
        # Create a zero matrix and place -sign at limiting coords
        wall_grad = np.zeros_like(centers)
        for i in range(N):
            if active_wall_mask[i]:
                k = limiting_coords[i]
                wall_grad[i, k] -= np.sign(centers[i, k])
                
        grad += wall_grad
        
    return -grad
