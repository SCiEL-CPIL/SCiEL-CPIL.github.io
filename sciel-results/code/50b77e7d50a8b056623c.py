"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial import KDTree

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
    # Minimum pairwise distance using cdist
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _hexagonal_init():
    """Create a hexagonal packing by selecting N closest points to center from a hex lattice."""
    # Generate a large hexagonal lattice
    i_max = 25
    j_max = 25
    
    points = []
    for j in range(-j_max, j_max + 1):
        for i in range(-i_max, i_max + 1):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2.0
            points.append((x, y))
    
    points = np.array(points)
    
    # Select N points closest to center
    dists_to_center = np.sum(points**2, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    selected_points = points[idx]
    
    # Scale to fit in [-0.5, 0.5] with margin for radius
    max_extent = max(np.max(np.abs(selected_points[:, 0])), 
                     np.max(np.abs(selected_points[:, 1])))
    
    # s * (max_extent + 0.5) <= 0.5  =>  s <= 0.5 / (max_extent + 0.5)
    s = 0.5 / (max_extent + 0.5)
    
    centers = selected_points * s
    
    return centers


def _local_optimization(centres, deadline, max_iters=5000):
    """Local optimization using push-away strategy for the most constrained configuration."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    if best_r <= 0:
        return best_centres, best_r
    
    # Initial step size based on current radius
    step_size = max(0.001, best_r * 0.2)
    no_improve_count = 0
    stall_limit = 50
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute wall distances
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        # Find nearest neighbor for each point using KDTree
        tree = KDTree(best_centres)
        dists, indices = tree.query(best_centres, k=2)
        min_pair_dists = dists[:, 1]
        nearest_indices = indices[:, 1]
        
        # The limiting constraint is min(min_pair_dist/2, wall_dist)
        pair_constraints = min_pair_dists / 2.0
        constraints = np.minimum(pair_constraints, wall_dists)
        
        # Find the most constrained point
        idx = np.argmin(constraints)
        
        # Determine if the limit is pair or wall
        if pair_constraints[idx] <= wall_dists[idx]:
            # Push away from nearest neighbor
            neighbor_idx = nearest_indices[idx]
            direction = best_centres[idx] - best_centres[neighbor_idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm
            
            # Try moving both points apart
            for scale in [1.0, 0.5, 0.25, 2.0, 0.1]:
                if time.time() > deadline:
                    break
                
                delta = step_size * scale * direction
                new_centres = best_centres.copy()
                new_centres[idx] += delta
                new_centres[neighbor_idx] -= delta
                
                # Check bounds
                if np.any(new_centres < -0.5 + 1e-10) or np.any(new_centres > 0.5 - 1e-10):
                    continue
                
                # Quick check: distances from moved points to all others
                diffs_i = new_centres[idx] - new_centres
                diffs_i[idx] = np.inf
                diffs_i[neighbor_idx] = np.inf
                min_near_i = np.min(np.linalg.norm(diffs_i, axis=1))
                
                diffs_j = new_centres[neighbor_idx] - new_centres
                diffs_j[idx] = np.inf
                diffs_j[neighbor_idx] = np.inf
                min_near_j = np.min(np.linalg.norm(diffs_j, axis=1))
                
                min_near = min(min_near_i, min_near_j)
                
                new_wall_dists_i = np.min([
                    0.5 - new_centres[idx, 0],
                    0.5 + new_centres[idx, 0],
                    0.5 - new_centres[idx, 1],
                    0.5 + new_centres[idx, 1]
                ])
                new_wall_dists_j = np.min([
                    0.5 - new_centres[neighbor_idx, 0],
                    0.5 + new_centres[neighbor_idx, 0],
                    0.5 - new_centres[neighbor_idx, 1],
                    0.5 + new_centres[neighbor_idx, 1]
                ])
                new_wall = min(new_wall_dists_i, new_wall_dists_j)
                
                new_r_est = min(min_near / 2.0, new_wall)
                if new_r_est <= best_r + 1e-12:
                    continue
                
                # Full radius computation
                new_r = _compute_radius(new_centres)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_centres
                    no_improve_count = 0
                    break
            else:
                # No improvement with pair move, try wall move
                pass
        
        else:
            # Push away from the closest wall
            dx = np.array([
                0.5 - best_centres[idx, 0],
                0.5 + best_centres[idx, 0],
                0.5 - best_centres[idx, 1],
                0.5 + best_centres[idx, 1]
            ])
            wall_idx = np.argmin(dx)
            direction = np.zeros(2)
            if wall_idx == 0:
                direction[0] = -1.0
            elif wall_idx == 1:
                direction[0] = 1.0
            elif wall_idx == 2:
                direction[1] = -1.0
            else:
                direction[1] = 1.0
            
            for scale in [1.0, 0.5, 0.25, 2.0, 0.1]:
                if time.time() > deadline:
                    break
                
                delta = step_size * scale * direction
                new_centres = best_centres.copy()
                new_centres[idx] += delta
                
                # Check bounds
                if np.any(new_centres < -0.5 + 1e-10) or np.any(new_centres > 0.5 - 1e-10):
                    continue
                
                # Quick estimate
                diffs = new_centres[idx] - new_centres
                diffs[idx] = np.inf
                min_near_dist = np.min(np.linalg.norm(diffs, axis=1))
                
                new_wall_dists = np.min([
                    0.5 - new_centres[idx, 0],
                    0.5 + new_centres[idx, 0],
                    0.5 - new_centres[idx, 1],
                    0.5 + new_centres[idx, 1]
                ])
                
                new_r_est = min(min_near_dist / 2.0, new_wall_dists)
                if new_r_est <= best_r + 1e-12:
                    continue
                
                new_r = _compute_radius(new_centres)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_centres
                    no_improve_count = 0
                    break
        
        if no_improve_count == 0:
            # Improved, keep step size or slightly reduce
            step_size *= 0.95
        else:
            no_improve_count += 1
            step_size *= 0.9
            if step_size < 1e-8 or no_improve_count > stall_limit:
                # Try a random restart perturbation
                if no_improve_count > stall_limit * 2:
                    break
                # Small random jitter to escape local optimum
                jitter = np.random.uniform(-step_size * 0.5, step_size * 0.5, size=best_centres.shape)
                new_centres = best_centres + jitter
                new_centres = np.clip(new_centres, -0.499, 0.499)
                new_r = _compute_radius(new_centres)
                if new_r >= best_r - 1e-10:
                    best_centres = new_centres
                    best_r = new_r
                no_improve_count = 0
                step_size = max(0.001, best_r * 0.1)
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # Leave 5s safety margin
    
    best_centres = None
    best_r = -1
    
    # Try multiple random seeds for diversity
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Start with hexagonal packing as initial configuration
        centres = _hexagonal_init()
        
        # Add small random perturbation for diversity (except first seed)
        if seed != 42:
            perturbation = np.random.uniform(-0.003, 0.003, size=centres.shape)
            centres += perturbation
            # Clip to bounds with small margin
            centres = np.clip(centres, -0.499, 0.499)
        
        current_r = _compute_radius(centres)
        
        # Local optimization
        centres, current_r = _local_optimization(centres, deadline)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
        
        # If we're already above the best known, we can stop early
        if best_r > 0.0366:
            break
    
    # Final check
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > 0:
            return best_centres
    
    # Fallback to a valid grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
