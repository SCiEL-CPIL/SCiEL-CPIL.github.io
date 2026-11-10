import numpy as np

def construct():
    N = 300
    
    # 1. Initialize with a hexagonal lattice
    s = 0.08 # Smaller spacing to have more candidates
    k = int(1.0 / s) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    points = np.array(rows)
    radii = np.hypot(points[:, 0], points[:, 1])
    # Keep points well inside
    points = points[radii < 0.95]
    
    # Select N points using greedy farthest-point sampling
    if len(points) > N:
        # Start with the point closest to the origin? Or just random? 
        # Closest to origin is good for symmetry.
        dists = np.linalg.norm(points, axis=1)
        current_idx = [np.argmin(dists)]
        current_pts = points[current_idx]
        
        for _ in range(N - 1):
            # Calculate min distance from each candidate to current set
            # diff: (M, 2) - (K, 2) -> (M, K, 2)
            diff = points[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dists_to_set = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists_to_set, axis=1)
            # Pick the candidate with the max min_dist
            next_idx = np.argmax(min_dists)
            current_idx.append(next_idx)
            current_pts = np.vstack([current_pts, points[next_idx]])
        
        points = current_pts
    elif len(points) < N:
        # Pad with random points if lattice is too sparse
        while len(points) < N:
            p = np.random.uniform(-0.9, 0.9, 2)
            if np.linalg.norm(p) < 0.95:
                points = np.vstack([points, p])

    # 2. Optimization
    r = 0.02
    max_iter = 5000
    step = 0.02
    noise_level = 0.0
    
    for it in range(max_iter):
        # Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e10) # Avoid self-interaction issues in min
        
        # Force calculation
        force = np.zeros_like(points)
        
        # Repulsion
        mask = dist < 2 * r
        np.fill_diagonal(mask, False)
        
        # coeff[i,j] = (2r - dist[i,j]) / dist[i,j] if mask else 0
        coeff = np.zeros_like(dist)
        coeff[mask] = (2 * r - dist[mask]) / dist[mask]
        
        force += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary
        radii = np.linalg.norm(points, axis=1)
        boundary_mask = radii > (1 - r)
        if np.any(boundary_mask):
            # Push inwards
            # Force = -k * (radii - (1-r)) * unit_vector
            violation = radii[boundary_mask] - (1 - r)
            unit_vec = points[boundary_mask] / radii[boundary_mask, np.newaxis]
            force[boundary_mask] -= unit_vec * violation[:, np.newaxis] * 5.0
            
        # Update
        points += step * force
        
        # Add noise to escape local minima
        if noise_level > 0:
            points += np.random.normal(0, noise_level, points.shape)
            
        # Project to boundary (hard constraint)
        radii = np.linalg.norm(points, axis=1)
        too_far = radii > (1 - r)
        if np.any(too_far):
            points[too_far] = points[too_far] * ((1 - r) / radii[too_far])[:, np.newaxis]
            
        # Check for radius increase
        min_dist = np.min(dist) # dist has 1e10 on diagonal, so min is off-diagonal
        max_rad = np.max(radii)
        
        if min_dist > 2 * r + 1e-5 and max_rad < 1 - r - 1e-5:
            r += 0.001
            step *= 0.99 # Reduce step as we converge
            noise_level = 0.0
        else:
            # If stuck, maybe increase noise or reduce step
            step *= 0.999
            if it % 100 == 0:
                noise_level = 0.001 * r # Small noise
            else:
                noise_level *= 0.95
                
        if r > 0.055: # Stop if we hit a high value
             break

    return points
