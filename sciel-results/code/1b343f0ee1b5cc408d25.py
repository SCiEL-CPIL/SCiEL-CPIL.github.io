    def force_step(points, lr, repulsion_strength=1.0, boundary_strength=10.0):
        forces = np.zeros_like(points)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Cap distance to avoid huge forces
        dists = np.maximum(dists, 0.05)
        
        inv_d2 = 1.0 / (dists * dists)
        np.fill_diagonal(inv_d2, 0.0)
        
        # Force vector
        forces += np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Boundary force
        margin = 0.45
        
        mask_r = points[:, 0] > margin
        forces[mask_r, 0] -= (points[mask_r, 0] - margin) * boundary_strength
        
        mask_l = points[:, 0] < -margin
        forces[mask_l, 0] += (-margin - points[mask_l, 0]) * boundary_strength
        
        mask_t = points[:, 1] > margin
        forces[mask_t, 1] -= (points[mask_t, 1] - margin) * boundary_strength
        
        mask_b = points[:, 1] < -margin
        forces[mask_b, 1] += (-margin - points[mask_b, 1]) * boundary_strength
        
        new_points = points + forces * lr
        new_points = np.clip(new_points, -0.5, 0.5)
        
        return new_points
