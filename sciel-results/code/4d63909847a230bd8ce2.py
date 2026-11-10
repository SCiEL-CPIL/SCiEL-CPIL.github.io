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


def _local_optimization(centres, max_iters=300, initial_step=0.01):
    """Local optimization using gradient-free method with constraint handling."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    step_size = initial_step
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate constraints
        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)
        
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)
        
        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        idx = np.argmin(constraints)
        
        # Try perturbations on the most constrained point
        improved = False
        n_tries = 20
        for _ in range(n_tries):
            if time.time() > deadline:
                break
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            delta = step_size * np.array([np.cos(angle), np.sin(angle)])
            new_centres = best_centres.copy()
            new_centres[idx] += delta
            
            # Check bounds
            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue
            
            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centres = new_centres
                improved = True
                step_size *= 0.95
                break
        
        if not improved:
            step_size *= 0.85
            if step_size < 1e-7:
                break
    
    return best_centres, best_r


def _simulated_annealing(centres, max_iters=1000, t_start=0.01, t_end=1e-6):
    """Simulated annealing optimization."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    current_centres = centres.copy()
    current_r = best_r
    
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Random perturbation
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, t)
        delta = magnitude * np.array([np.cos(angle), np.sin(angle)])
        
        new_centres = current_centres.copy()
        new_centres[idx] += delta
        
        # Check bounds
        if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
            continue
        
        new_r = _compute_radius(new_centres)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centres = new_centres
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centres = current_centres.copy()
        else:
            # Accept with probability exp((new_r - current_r) / t)
            delta_r = new_r - current_r
            if np.random.random() < np.exp(delta_r / max(t, 1e-12)):
                current_centres = new_centres
                current_r = new_r
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    
    # Start with hexagonal packing as initial configuration
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # Local optimization first
    if time.time() - start_time < TIME_LIMIT - 15:
        centres, current_r = _local_optimization(centres)
    
    # Simulated annealing for further improvement
    if time.time() - start_time < TIME_LIMIT - 5:
        centres, current_r = _simulated_annealing(centres)
    
    # Final local optimization
    if time.time() - start_time < TIME_LIMIT - 3:
        centres, current_r = _local_optimization(centres, max_iters=50, initial_step=0.005)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return centres
