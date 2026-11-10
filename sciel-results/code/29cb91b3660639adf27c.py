import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Boundary constraint
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint using cdist (faster)
    dists = cdist(positions, positions)
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
    Iterative local search that identifies and resolves the most constraining
    pairs/boundaries, nudging particles to increase the minimum radius.
    """
    start_time = time.time()
    best_r = compute_radius(positions)
    best_pos = positions.copy()
    
    # Track convergence
    prev_r = best_r
    stall_count = 0
    
    while time.time() - start_time < time_limit:
        r = compute_radius(positions)
        if r > best_r:
            best_r = r
            best_pos = positions.copy()
            stall_count = 0
        else:
            if r <= prev_r + 1e-10:
                stall_count += 1
            prev_r = r
        
        if stall_count > 100:
            break
        
        # Determine limiting factor
        min_bound = 0.5 - np.max(np.abs(positions))
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Find indices of limiting constraints
        boundary_margins = 0.5 - np.max(np.abs(positions), axis=1)
        idx_bound = np.argmin(boundary_margins)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Nudge with adaptive step size
        step = 0.0005 + 0.001 * (best_r / 0.072)
        
        if min_bound < min_dist / 2.0:
            p = positions[idx_bound]
            direction = -p
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[idx_bound] += direction * step
        else:
            p_i = positions[i]
            p_j = positions[j]
            direction = p_i - p_j
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[i] += direction * step
                positions[j] -= direction * step
                
        positions = np.clip(positions, -0.5, 0.5)

    return best_pos

def local_refinement(positions, time_limit=10.0):
    """
    Higher-precision refinement using iterative constraint satisfaction.
    Focuses on the most constrained particles and uses smaller steps.
    """
    start_time = time.time()
    best_r = compute_radius(positions)
    best_pos = positions.copy()
    
    step_sizes = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_step_idx = 0
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        r = compute_radius(positions)
        if r > best_r + 1e-12:
            best_r = r
            best_pos = positions.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 200:
            if current_step_idx < len(step_sizes) - 1:
                current_step_idx += 1
                no_improve = 0
            else:
                break
        
        step = step_sizes[current_step_idx]
        
        # Find all near-minimum constraints and resolve them
        min_bound = 0.5 - np.max(np.abs(positions))
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Handle boundary constraints for particles close to boundary
        boundary_margins = 0.5 - np.max(np.abs(positions), axis=1)
        close_to_boundary = boundary_margins < min_bound + 0.005
        for idx in np.where(close_to_boundary)[0]:
            p = positions[idx]
            direction = -p
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[idx] += direction * step * 0.5
        
        # Handle pair constraints
        # Find all pairs with distance close to minimum
        close_pairs = dists < min_dist + 0.005
        close_pairs[~np.triu(close_pairs, 1)] = False  # upper triangle only
        close_idx = np.where(close_pairs)
        
        for i, j in zip(close_idx[0], close_idx[1]):
            p_i = positions[i]
            p_j = positions[j]
            direction = p_i - p_j
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
                positions[i] += direction * step * 0.5
                positions[j] -= direction * step * 0.5
                
        positions = np.clip(positions, -0.4999, 0.4999)

    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit

    positions = generate_hexagonal_initial(N)
    
    current_radius = compute_radius(positions)
    best_radius = current_radius
    best_positions = positions.copy()
    
    T = 0.03
    T_min = 1e-7
    cooling_rate = 0.9995
    base_step = 0.04
    
    iter_count = 0
    no_improve_count = 0
    restart_interval = 3000
    
    np.random.seed(42)
    
    # Phase 1: Simulated annealing
    while time.time() < deadline - 30.0:  # Reserve 30s for polish
        iter_count += 1
        
        step_size = base_step * (0.3 + T / 0.03)
        
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
            positions = best_positions + np.random.normal(0, 0.015, (N, 2))
            positions = np.clip(positions, -0.495, 0.495)
            current_radius = compute_radius(positions)
            T = 0.015
            no_improve_count = 0
            iter_count = 0
            
    # Phase 2: Polish
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        best_positions = polish(best_positions, time_limit=min(remaining * 0.4, 15.0))
    
    # Phase 3: High-precision local refinement
    remaining = deadline - time.time() - 2.0
    if remaining > 3.0:
        best_positions = local_refinement(best_positions, time_limit=min(remaining * 0.6, 15.0))
    
    return best_positions
