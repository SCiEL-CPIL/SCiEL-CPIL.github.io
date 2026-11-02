import numpy as np

def construct():
    N = 32
    
    def calc_score(centers):
        """Vectorized score calculation: r_i = min(d_wall_i, 0.5 * min_j d_ij)"""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_nn_dist = np.min(dists, axis=1)
        
        dx = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
        dy = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        d_wall = np.minimum(dx, dy)
        
        radii = np.minimum(d_wall, 0.5 * min_nn_dist)
        
        return np.sum(radii), radii

    def compute_gradients(centers, radii):
        """
        Compute subgradients for the sum of radii.
        For each circle i, find the constraints (walls or neighbors) that define its radius.
        The gradient of the radius with respect to position is the average of the unit vectors
        pointing away from the active constraints.
        """
        grads = np.zeros_like(centers)
        
        # 1. Wall constraints
        # dx = min(x, 1-x). If x < 0.5, active wall is left (grad +1 in x). If x > 0.5, active wall is right (grad -1 in x).
        # We need to check if the wall constraint is actually the limiting one (i.e., r_i == d_wall_i).
        
        dx = centers[:, 0]
        dy = centers[:, 1]
        
        d_left = dx
        d_right = 1.0 - dx
        d_bottom = dy
        d_top = 1.0 - dy
        
        d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_nn_dist = np.min(dists, axis=1)
        
        # Active constraints logic:
        # r_i is min(d_wall, 0.5 * min_nn_dist)
        # Case 1: d_wall < 0.5 * min_nn_dist. Then r_i = d_wall.
        #         The gradient is determined by which wall is closest.
        # Case 2: 0.5 * min_nn_dist <= d_wall. Then r_i = 0.5 * min_nn_dist.
        #         The gradient is determined by the closest neighbor(s).
        
        for i in range(N):
            r_i = radii[i]
            active_vecs = []
            
            # Check walls
            # Tolerance to handle floating point
            tol = 1e-9
            
            if abs(r_i - d_wall[i]) < tol:
                # Wall is limiting
                # Determine which wall(s)
                if abs(d_left[i] - r_i) < tol:
                    active_vecs.append([1.0, 0.0]) # Push right
                if abs(d_right[i] - r_i) < tol:
                    active_vecs.append([-1.0, 0.0]) # Push left
                if abs(d_bottom[i] - r_i) < tol:
                    active_vecs.append([0.0, 1.0]) # Push up
                if abs(d_top[i] - r_i) < tol:
                    active_vecs.append([0.0, -1.0]) # Push down
            else:
                # Neighbor is limiting
                # Find all neighbors j such that 0.5 * dist(i,j) == r_i
                for j in range(N):
                    if i == j: continue
                    if abs(0.5 * dists[i, j] - r_i) < tol:
                        # Gradient of 0.5 * dist(i,j) wrt pos_i is 0.5 * (pos_i - pos_j) / dist(i,j)
                        # This is a unit vector from j to i.
                        vec = centers[i] - centers[j]
                        dist = dists[i, j]
                        if dist > 0:
                            active_vecs.append(vec / dist)
            
            if active_vecs:
                # Sum the unit vectors (subgradient)
                grad_i = np.sum(active_vecs, axis=0)
                # Normalize? The magnitude of the subgradient varies. 
                # For gradient ascent, we usually normalize to unit step or use a fixed step size.
                # Normalizing ensures consistent step length regardless of number of contacts.
                norm = np.linalg.norm(grad_i)
                if norm > 0:
                    grads[i] = grad_i / norm
            else:
                # Should not happen if r_i > 0
                grads[i] = [0, 0]
                
        return grads

    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Try multiple initial configurations
    for trial in range(10):
        if trial == 0:
            # 8x4 Grid
            cols, rows = 8, 4
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        elif trial == 1:
            # 4x8 Grid
            cols, rows = 4, 8
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        elif trial == 2:
            # Hexagonal-ish
            cols, rows = 6, 6
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                offset = 0.5 / cols if r % 2 == 1 else 0.0
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5 + offset) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        elif trial == 3:
            # 5x7 (35 slots, take 32)
            cols, rows = 5, 7
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < N:
                        centers[idx, 0] = (c + 0.5) / cols
                        centers[idx, 1] = (r + 0.5) / rows
                        idx += 1
        else:
            # Random
            centers = rng.uniform(0.1, 0.9, size=(N, 2))
        
        # Jitter
        centers = centers + rng.uniform(-0.02, 0.02, size=(N, 2))
        centers = np.clip(centers, 0.05, 0.95)
        
        # Gradient Ascent
        current_centers = centers.copy()
        steps = 5000
        step_size_start = 0.05
        step_size_end = 0.001
        
        for step in range(steps):
            score, radii = calc_score(current_centers)
            
            if score > best_score:
                best_score = score
                best_centers = current_centers.copy()
                best_radii = radii.copy()
                
            grads = compute_gradients(current_centers, radii)
            
            # Anneal step size
            frac = step / steps
            step_size = step_size_start * (1 - frac) + step_size_end * frac
            
            # Update
            current_centers = current_centers + step_size * grads
            current_centers = np.clip(current_centers, 0.0, 1.0)
            
        # Final check after loop
        score, radii = calc_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            best_radii = radii.copy()

    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i, 0], best_centers[i, 1], best_radii[i]]
        
    return result
