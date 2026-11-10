import numpy as np
import time

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Boundary constraint
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def generate_hexagonal_initial(N, square_half=0.5):
    """Generate a hexagonal packing initialization."""
    max_coord = 0.5
    spacing = 0.15
    
    points = []
    row = 0
    while True:
        y = -max_coord + row * spacing * np.sqrt(3) / 2
        if y > max_coord:
            break
        x_offset = (row % 2) * spacing / 2
        col = 0
        while True:
            x = -max_coord + col * spacing + x_offset
            if x > max_coord:
                break
            points.append([x, y])
            col += 1
        row += 1
        if len(points) > 200:
            break
            
    points = np.array(points)
    
    if len(points) > N:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        points = points[indices]
    elif len(points) < N:
        return generate_hexagonal_initial(N, square_half, spacing * 0.9)
        
    center = np.mean(points, axis=0)
    points = points - center
    
    max_abs = np.max(np.abs(points))
    if max_abs > 0.48:
        points = points * (0.48 / max_abs)
        
    return points

def polish(positions, time_limit=5.0):
    """
    A simple local search polish to refine the best position found.
    Identifies the most constrained particles and nudges them.
    """
    start_time = time.time()
    best_r = compute_radius(positions)
    best_pos = positions.copy()
    
    while time.time() - start_time < time_limit:
        r = compute_radius(positions)
        if r > best_r:
            best_r = r
            best_pos = positions.copy()
            
        # Determine limiting factor
        min_bound = 0.5 - np.max(np.abs(positions))
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Find indices of limiting constraints
        idx_bound = np.argmin(0.5 - np.max(np.abs(positions), axis=1))
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Nudge
        if min_bound < min_dist / 2.0:
            p = positions[idx_bound]
            direction = -p
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[idx_bound] += direction * 0.001
        else:
            p_i = positions[i]
            p_j = positions[j]
            direction = p_i - p_j
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[i] += direction * 0.001
                positions[j] -= direction * 0.001
                
        positions = np.clip(positions, -0.5, 0.5)

    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    positions = generate_hexagonal_initial(N)
    
    current_radius = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.999
    base_step = 0.05
    
    iter_count = 0
    no_improve_count = 0
    restart_interval = 5000
    
    np.random.seed(42)
    
    while time.time() < deadline:
        iter_count += 1
        
        step_size = base_step * (0.5 + T / 0.05)
        
        idx = np.random.randint(0, N)
        
        delta = np.random.normal(0, step_size, 2)
        old_pos = positions[idx].copy()
        new_pos = old_pos + delta
        
        if np.any(np.abs(new_pos) > 0.5):
            continue
            
        positions[idx] = new_pos
        new_radius = compute_radius(positions)
        
        delta_energy = current_radius - new_radius
        
        if delta_energy < 0:
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_positions = positions.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            if np.random.rand() < np.exp(-delta_energy / T):
                current_radius = new_radius
                no_improve_count += 1
            else:
                positions[idx] = old_pos
                no_improve_count += 1
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        if no_improve_count > restart_interval:
            positions = best_positions + np.random.normal(0, 0.02, (N, 2))
            positions = np.clip(positions, -0.49, 0.49)
            current_radius = compute_radius(positions)
            T = 0.02
            no_improve_count = 0
            iter_count = 0
            
    if time.time() < deadline - 10:
        best_positions = polish(best_positions, time_limit=deadline - time.time() - 2)
        
    return best_positions
