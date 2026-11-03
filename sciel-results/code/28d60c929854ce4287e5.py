"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def _get_radius(centers):
    """Compute the maximum radius for given centers."""
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    n = len(centers)
    if n < 2:
        return max(0.0, min_wall)
        
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    
    r = min(min_wall, min_pair / 2.0)
    return max(0.0, r)

def _generate_hex_initial():
    """Generate a hexagonal grid initialization."""
    best_centers = None
    best_r = -1.0
    
    for d in np.linspace(0.065, 0.085, 40):
        pts = []
        row = 0
        y = -0.5 + d * math.sqrt(3) / 2.0
        while y <= 0.5 - d * math.sqrt(3) / 2.0 + 1e-9:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + d / 2.0 + x_offset
            while x <= 0.5 - d / 2.0 + 1e-9:
                pts.append([x, y])
                x += d
            y += d * math.sqrt(3) / 2.0
            row += 1
        
        if len(pts) < N:
            continue
            
        pts = np.array(pts)
        
        # Center the grid
        center_x = (np.min(pts[:, 0]) + np.max(pts[:, 0])) / 2.0
        center_y = (np.min(pts[:, 1]) + np.max(pts[:, 1])) / 2.0
        pts_centered = pts - np.array([center_x, center_y])
        
        if len(pts_centered) >= N:
            # Select N points. Prefer those closer to center to maintain symmetry? 
            # Or just take first N. Let's take a subset that is roughly uniform.
            # Actually, for hex grid, taking first N (row by row) is usually fine.
            # But to be safe against edge effects, let's try to pick the N points 
            # with the largest minimum distance to boundary? That's expensive.
            # Let's just take the first N for now, or random.
            # Random is better to avoid bias.
            idx = np.random.choice(len(pts_centered), N, replace=False)
            init_pts = pts_centered[idx]
            
            r = _get_radius(init_pts)
            if r > best_r:
                best_r = r
                best_centers = init_pts.copy()
        
    if best_centers is None:
        # Fallback: Square grid
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

def _bottleneck_info(centers):
    """
    Identify the pair of centers (or center-wall) that limits the radius.
    Returns:
    - r: current radius
    - type: 'pair' or 'wall'
    - i, j: indices (if pair) or i (if wall) and axis (0 or 1)
    """
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    min_wall_idx = np.argmin(wall_dists)
    min_wall_axis = int(np.argmin(np.abs(centers[min_wall_idx])))
    
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    min_pair_idx = np.argmin(dist_matrix)
    i, j = np.unravel_index(min_pair_idx, dist_matrix.shape)
    
    if min_wall < min_pair / 2.0:
        return min_wall, 'wall', min_wall_idx, min_wall_axis
    else:
        return min_pair / 2.0, 'pair', i, j

def _sa_local_search(centers, max_time=120.0, initial_temp=0.01):
    """
    Simulated Annealing local search on the exact non-smooth objective.
    Moves are small random perturbations.
    Accept if improvement, or with probability exp(-delta/E).
    """
    centers = centers.copy()
    current_r = _get_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    temp = initial_temp
    cooling_rate = 0.9995
    step_size = 0.005
    
    t_start = time.time()
    
    while time.time() - t_start < max_time:
        if temp < 1e-7:
            break
            
        # Generate a random move
        # Perturb a random subset of points? Or all points?
        # Perturbing all points is more expensive but smoother.
        # Let's perturb all points slightly.
        move = np.random.randn(*centers.shape) * step_size
        
        new_centers = centers + move
        # Project to square
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.5, 0.5)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.5, 0.5)
        
        new_r = _get_radius(new_centers)
        
        delta = new_r - current_r
        
        if delta > 0 or np.random.random() < math.exp(-delta / temp):
            centers = new_centers
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
                
        # Adapt step size
        # If we accept a lot, increase step size?
        # If we reject a lot, decrease?
        # Simple: cool down.
        temp *= cooling_rate
        
        if time.time() > time.time() + 10: # Safety break
            break
            
    return best_centers, best_r

def _local_search(centers, max_time=120.0):
    """
    A faster, greedy local search that directly pushes bottlenecks apart.
    """
    centers = centers.copy()
    current_r = _get_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    t_start = time.time()
    
    while time.time() - t_start < max_time:
        r, btype, i, j = _bottleneck_info(centers)
        
        if btype == 'pair':
            # Push i and j apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-9:
                diff = np.random.randn(2)
                norm = np.linalg.norm(diff)
            dir_vec = diff / norm
            
            # Move i by +dir_vec * eps, j by -dir_vec * eps
            eps = 0.001
            centers[i] += dir_vec * eps
            centers[j] -= dir_vec * eps
        elif btype == 'wall':
            # Push i away from wall
            axis = j
            # Determine direction away from wall
            # If centers[i, axis] > 0, push left (-). Else push right (+).
            sign = -np.sign(centers[i, axis])
            if sign == 0:
                sign = 1.0
            eps = 0.001
            centers[i, axis] += sign * eps
        
        # Project to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        new_r = _get_radius(centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            
        # Occasionally add jitter to escape local minima
        if np.random.random() < 0.01:
            jitter = np.random.randn(*centers.shape) * 0.005
            centers += jitter
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Initialize
    centers = _generate_hex_initial()
    
    # 2. Local Search (Greedy Bottleneck)
    centers, r1 = _local_search(centers, max_time=30.0)
    
    # 3. Simulated Annealing (to escape local minima)
    centers, r2 = _sa_local_search(centers, max_time=40.0, initial_temp=0.005)
    
    # 4. Final Polish (Greedy Bottleneck again)
    centers, r3 = _local_search(centers, max_time=60.0)
    
    # Ensure validity
    if not np.all(np.isfinite(centers)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
        
    return centers

# EVOLVE-BLOCK-END
