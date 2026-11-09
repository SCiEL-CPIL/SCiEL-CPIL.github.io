"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centres):
    """Compute the maximum radius for given centres."""
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centres)
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    if len(centres) > 1:
        # Compute all pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
    
    return min(min_boundary, min_pair / 2.0)

def _local_optimize(centres, time_limit=100.0, max_iter=5000):
    """Local optimization using gradient ascent on the minimum distance."""
    start_time = time.time()
    centres = centres.copy()
    current_radius = _compute_radius(centres)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum pairwise distance and which pair
        min_idx = np.argmin(dists)
        i, j = np.unravel_index(min_idx, dists.shape)
        min_pair_dist = dists[i, j]
        
        # Boundary distances for each point
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        # Determine the limiting constraint
        if min_pair_dist / 2.0 < min_boundary:
            # Limiting constraint is the pair (i, j)
            # Move both points apart
            direction = centres[i] - centres[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm
            
            # Move i in +direction, j in -direction
            step = 0.001
            new_i = centres[i] + step * direction
            new_j = centres[j] - step * direction
            
            # Clamp to bounds
            new_i = np.clip(new_i, -0.5, 0.5)
            new_j = np.clip(new_j, -0.5, 0.5)
            
            # Test if this improves
            old_i, old_j = centres[i].copy(), centres[j].copy()
            centres[i] = new_i
            centres[j] = new_j
            new_radius = _compute_radius(centres)
            
            if new_radius > current_radius + 1e-12:
                current_radius = new_radius
            else:
                centres[i] = old_i
                centres[j] = old_j
        else:
            # Limiting constraint is boundary
            # Find which point has the minimum boundary distance
            min_boundary_idx = np.argmin(boundary_dists)
            i = min_boundary_idx
            min_bdist = boundary_dists[i]
            
            # Find which axis is limiting
            axis_dists = 0.5 - np.abs(centres[i])
            axis = np.argmin(axis_dists)
            
            # Move toward center along that axis
            sign = np.sign(centres[i][axis])
            if sign == 0:
                sign = 1
            step = 0.001
            new_pos = centres[i].copy()
            new_pos[axis] += -sign * step
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            old_pos = centres[i].copy()
            centres[i] = new_pos
            new_radius = _compute_radius(centres)
            
            if new_radius > current_radius + 1e-12:
                current_radius = new_radius
            else:
                centres[i] = old_pos
    
    return centres, current_radius

def _structured_initial():
    """Create a structured initial configuration based on FCC lattice."""
    # Use a subset of FCC lattice points
    # FCC unit cell has points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Scale to fit in [-0.5, 0.5]^3
    
    points = []
    # Generate FCC points in a 3x3x3 supercell
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                # FCC basis
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (ix + dx) / 3.0 - 0.5
                    y = (iy + dy) / 3.0 - 0.5
                    z = (iz + dz) / 3.0 - 0.5
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
    
    points = np.array(points)
    
    # If we have more than N points, select the best N
    if len(points) > N:
        # Compute pairwise distances and select points that maximize minimum distance
        # Use a greedy selection
        selected = [0]
        for _ in range(N - 1):
            best_idx = -1
            best_min_dist = -1
            for idx in range(len(points)):
                if idx in selected:
                    continue
                # Distance to all selected points
                dists = np.linalg.norm(points[idx] - points[selected], axis=1)
                min_dist = np.min(dists)
                # Also consider boundary
                bdist = np.min(0.5 - np.abs(points[idx]))
                effective = min(min_dist / 2.0, bdist)
                if effective > best_min_dist:
                    best_min_dist = effective
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
        points = points[selected]
    
    # If we have fewer than N points, add random points
    if len(points) < N:
        np.random.seed(42)
        for _ in range(N - len(points)):
            new_point = np.random.uniform(-0.5, 0.5, size=3)
            points = np.vstack([points, new_point])
    
    return points[:N]

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius.
    Uses structured initial configuration followed by local optimization.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin below 150s kill
    
    # Start with structured initial configuration
    centres = _structured_initial()
    best_centres = centres.copy()
    best_radius = _compute_radius(best_centres)
    
    # Local optimization
    centres, radius = _local_optimize(centres, time_limit=time_limit - 20.0)
    if radius > best_radius:
        best_centres = centres
        best_radius = radius
    
    # Try a few more restarts from perturbed configurations if time permits
    np.random.seed(123)
    for restart in range(3):
        if time.time() - start_time > time_limit - 30.0:
            break
        
        # Perturb the best configuration
        perturbed = best_centres + np.random.normal(0, 0.01, size=(N, 3))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Local optimize from perturbed
        centres, radius = _local_optimize(perturbed, time_limit=time_limit - (time.time() - start_time) - 10.0)
        if radius > best_radius:
            best_centres = centres
            best_radius = radius
    
    # Final check
    if time.time() - start_time > time_limit - 5.0:
        return best_centres
    
    # One more refinement pass
    centres, radius = _local_optimize(best_centres, time_limit=time_limit - (time.time() - start_time) - 2.0)
    if radius > best_radius:
        best_centres = centres
        best_radius = radius
    
    return best_centres
# EVOLVE-BLOCK-END
