"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 20):
        # For hexagonal packing, alternating rows have offset
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing s
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                r = s / 2.0
                
                if r <= best_r:
                    continue
                
                # Create the grid
                centres = []
                y_start = -0.5 + s * np.sqrt(3) / 2
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_start = -0.5 + s
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _local_optimization(centres, max_iters=5000, initial_step=0.01):
    """
    Parallel local optimization using repulsive forces from nearest neighbors.
    This is much faster and effective than single-point moves.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    step_size = initial_step
    wall_margin = 0.02
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute pairwise distances
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance
        min_dist = np.min(dists)
        
        # Check walls
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        min_wall = np.min(wall_dists)
        
        r = min(min_dist / 2.0, min_wall)
        
        if r > best_r:
            best_r = r
            best_centres = best_centres.copy()
        
        # Compute forces
        forces = np.zeros_like(best_centres)
        
        # Identify pairs that are "too close"
        # Use a threshold to limit calculations, though N=200 is small
        # Threshold: 1.5 times the minimum distance
        threshold = min_dist * 1.5
        mask = dists < threshold
        
        # For each point i, sum up vectors to j where mask[i,j] is True, weighted by 1/d^2
        for i in range(N):
            neigh_idx = np.where(mask[i])[0]
            if len(neigh_idx) == 0:
                continue
            
            # Vectors from i to j
            vecs = best_centres[neigh_idx] - best_centres[i]
            # Distances
            d_vecs = dists[i, neigh_idx]
            
            # Avoid division by zero
            weights = 1.0 / (d_vecs ** 2)
            
            # Sum of weighted vectors
            force_i = np.sum(vecs * weights[:, np.newaxis], axis=0)
            forces[i] += force_i
        
        # Wall forces: push points away from walls if they are too close
        for i in range(N):
            x, y = best_centres[i]
            if x < -0.5 + wall_margin:
                forces[i, 0] += (-0.5 + wall_margin - x) * 10.0
            elif x > 0.5 - wall_margin:
                forces[i, 0] -= (x - (0.5 - wall_margin)) * 10.0
            if y < -0.5 + wall_margin:
                forces[i, 1] += (-0.5 + wall_margin - y) * 10.0
            elif y > 0.5 - wall_margin:
                forces[i, 1] -= (y - (0.5 - wall_margin)) * 10.0

        # Normalize forces to unit vectors to use step_size consistently
        norms = np.linalg.norm(forces, axis=1)
        norms[norms == 0] = 1.0
        directions = forces / norms[:, np.newaxis]
        
        # Update
        best_centres += step_size * directions
        
        # Project back to square (hard constraint)
        best_centres = np.clip(best_centres, -0.5, 0.5)
        
        # Decay step size
        step_size *= 0.999
        if step_size < 1e-7:
            break
            
    return best_centres, _compute_radius(best_centres)


def construct():
    start_time = time.time()
    
    # Start with hexagonal packing as initial configuration
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # Local optimization (parallel force-directed)
    if time.time() - start_time < TIME_LIMIT - 10:
        centres, current_r = _local_optimization(centres, max_iters=2000, initial_step=0.01)
        
    # Second pass with smaller steps to polish
    if time.time() - start_time < TIME_LIMIT - 5:
        centres, current_r = _local_optimization(centres, max_iters=2000, initial_step=0.001)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
