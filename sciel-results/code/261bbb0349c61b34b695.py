"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
import math

N = 200

def _get_radius(centers):
    """
    Compute the maximum radius for given centers.
    """
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix)
    
    r = min(min_wall, min_pair / 2.0)
    return max(0.0, r)

def _bottleneck_info(centers):
    """
    Identify the bottleneck constraint (wall or pair).
    Returns:
        r: current radius
        type: 'wall' or 'pair'
        idx: index of point (for wall) or tuple (i, j) (for pair)
    """
    wall_dists = 0.5 - np.abs(centers)
    min_wall_idx = np.argmin(wall_dists)
    min_wall = wall_dists.flat[min_wall_idx]
    
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_pair_idx = np.argmin(dist_matrix)
    min_pair = dist_matrix.flat[min_pair_idx]
    
    if min_wall < min_pair / 2.0:
        # Wall is bottleneck
        # min_wall_idx is flat index in (n, 2) array
        point_idx = min_wall_idx // 2
        return min_wall, 'wall', point_idx
    else:
        # Pair is bottleneck
        i, j = np.unravel_index(min_pair_idx, (N, N))
        return min_pair / 2.0, 'pair', (i, j)

def _generate_hex_initial(rows):
    """
    Generate a hexagonal packing layout.
    """
    cols = math.ceil(N / rows)
    
    # Estimate spacing
    # Width of hex row: if even, cols * d. If odd, (cols-1) * d.
    # We want to fit in [0.5, 0.5] width.
    # Let's just generate a grid of hex points and pick the best subset or scale.
    
    # Standard hex lattice
    # d is distance between centers.
    # r = d/2.
    # We want to maximize r.
    
    # For a hex grid with R rows and C cols:
    # Width = (C - 0.5) * d  (approx, depends on offset)
    # Height = (R - 1) * d * sqrt(3)/2 + d
    
    # Let's iterate over possible d or just compute from bounds.
    
    # Actually, let's just create a dense hex grid and let the optimizer fix it?
    # No, let's try to fit it.
    
    # Approx:
    # Width ~ (cols - 0.5) * d
    # Height ~ (rows - 1) * d * sqrt(3)/2 + d
    
    d_w = 1.0 / (cols - 0.5)
    d_h = 1.0 / ((rows - 1) * math.sqrt(3)/2 + 1.0)
    
    d = min(d_w, d_h)
    
    pts = []
    row_spacing = d * math.sqrt(3) / 2
    
    for i in range(rows):
        y = -0.5 + d/2 + i * row_spacing
        if i % 2 == 0:
            n_in_row = cols
            x_start = -0.5 + d/2
        else:
            n_in_row = cols - 1
            x_start = -0.5 + d
            
        for j in range(n_in_row):
            x = x_start + j * d
            pts.append([x, y])
            
    pts = np.array(pts)
    
    if len(pts) < N:
        return None
    if len(pts) > N:
        # If too many, just take the first N? Or center them?
        # Taking first N might bias to corner.
        # Let's just return the first N for now, the optimizer will fix.
        # Or better, random selection?
        # For simplicity, take first N.
        pts = pts[:N]
        
    return pts

def _local_search(centers, time_limit=10.0):
    """
    Local search using bottleneck elimination and simulated annealing.
    """
    centers = centers.copy()
    start_time = time.time()
    
    # Initial radius
    current_r, type, idx = _bottleneck_info(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Temperature for simulated annealing
    # We want to perturb the bottleneck points.
    # Perturbation size should be related to current radius.
    
    T = 0.01 * current_r
    T_min = 1e-6
    cooling_rate = 0.95
    
    # We will iterate until time limit or convergence
    while time.time() < start_time + time_limit:
        # Identify bottleneck
        current_r, type, idx = _bottleneck_info(centers)
        
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            
        # If converged (no improvement in last few steps and T is low), break
        # (Simple check: if T is very small, we can stop)
        if T < T_min:
            break
            
        # Determine perturbation
        if type == 'pair':
            i, j = idx
            # Move i and j apart
            # Direction: from j to i for i, from i to j for j
            vec = centers[i] - centers[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                # Random direction
                vec = np.random.randn(2)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            
            # Perturbation magnitude
            # We want to increase distance by at least epsilon
            # Let's add a random kick in the direction of separation
            # Plus some random noise to escape local minima
            kick = T * 0.5
            noise = T * 0.5
            
            centers[i] += vec * kick + np.random.randn(2) * noise
            centers[j] -= vec * kick + np.random.randn(2) * noise
            
        elif type == 'wall':
            k = idx
            # Move k away from wall
            # Determine which wall
            x, y = centers[k]
            
            # Distance to walls
            dx = 0.5 - abs(x)
            dy = 0.5 - abs(y)
            
            if dx < dy:
                # X wall
                if x > 0:
                    vec = np.array([-1.0, 0.0])
                else:
                    vec = np.array([1.0, 0.0])
            else:
                # Y wall
                if y > 0:
                    vec = np.array([0.0, -1.0])
                else:
                    vec = np.array([0.0, 1.0])
            
            kick = T * 0.5
            noise = T * 0.5
            
            centers[k] += vec * kick + np.random.randn(2) * noise
            
        # Project back to square if necessary
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Check if new radius is better
        new_r, _, _ = _bottleneck_info(centers)
        
        # Accept if better, or with probability exp((new_r - current_r) / T)
        if new_r >= current_r:
            current_r = new_r
        else:
            # Simulated annealing acceptance
            # Note: We are maximizing, so delta = new_r - current_r < 0
            # Accept with prob exp(delta / T)
            delta = new_r - current_r
            if np.random.rand() < math.exp(delta / T):
                current_r = new_r
            else:
                # Reject, revert
                # We need to revert the changes
                # This is tricky without a stack.
                # For simplicity, let's just always accept for now?
                # No, SA requires rejection.
                # Let's store old state?
                # For efficiency, let's just do "hill climbing" first, then add SA if time.
                # Actually, let's just do hill climbing with random restarts of the perturbation.
                # Revert:
                # We don't have the old centers easily.
                # Let's modify the loop to store old_centers.
                pass # We'll implement proper SA below
        
        # Cool down
        T *= cooling_rate

    return best_centers, best_r

def _sa_local_search(centers, time_limit=10.0):
    """
    Proper Simulated Annealing local search.
    """
    centers = centers.copy()
    start_time = time.time()
    
    current_r, _, _ = _bottleneck_info(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.005 * current_r
    T_min = 1e-7
    cooling_rate = 0.98
    
    # We will iterate
    while time.time() < start_time + time_limit:
        if T < T_min:
            break
            
        old_centers = centers.copy()
        old_r = current_r
        
        # Identify bottleneck
        _, type, idx = _bottleneck_info(centers)
        
        # Perturb
        if type == 'pair':
            i, j = idx
            vec = centers[i] - centers[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                vec = np.random.randn(2)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            
            kick = T * 1.0
            noise = T * 1.0
            
            centers[i] += vec * kick + np.random.randn(2) * noise
            centers[j] -= vec * kick + np.random.randn(2) * noise
            
        elif type == 'wall':
            k = idx
            x, y = centers[k]
            
            dx = 0.5 - abs(x)
            dy = 0.5 - abs(y)
            
            if dx < dy:
                if x > 0:
                    vec = np.array([-1.0, 0.0])
                else:
                    vec = np.array([1.0, 0.0])
            else:
                if y > 0:
                    vec = np.array([0.0, -1.0])
                else:
                    vec = np.array([0.0, 1.0])
            
            kick = T * 1.0
            noise = T * 1.0
            
            centers[k] += vec * kick + np.random.randn(2) * noise
            
        # Project
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        new_r, _, _ = _bottleneck_info(centers)
        
        # Acceptance
        if new_r >= old_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            delta = new_r - old_r
            if np.random.rand() < math.exp(delta / T):
                current_r = new_r
            else:
                centers = old_centers
                current_r = old_r
        
        T *= cooling_rate

    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Generate multiple hex initializations
    # Try different row counts
    for rows in range(12, 18):
        if time.time() > deadline - 30:
            break
            
        pts = _generate_hex_initial(rows)
        if pts is not None:
            r = _get_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()
                
    # Fallback to grid if hex fails
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_radius = _get_radius(best_centers)
        
    # 2. Local Search with SA
    if time.time() < deadline - 10:
        # Run SA on the best initial
        time_left = deadline - time.time() - 5
        # Split time among a few restarts?
        # Let's do one long SA run first.
        optimized, r_opt = _sa_local_search(best_centers, time_limit=min(time_left, 60))
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()
            
    # 3. If time remains, try random restarts
    if time.time() < deadline - 10:
        # Generate a random perturbation of the best solution
        # Or a new random hex?
        
        # Let's try perturbing the current best
        perturbed = best_centers + np.random.randn(N, 2) * 0.01
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
        
        r_pert = _get_radius(perturbed)
        
        # Run SA on perturbed
        time_left = deadline - time.time() - 5
        if time_left > 5:
            optimized2, r_opt2 = _sa_local_search(perturbed, time_limit=min(time_left, 30))
            if r_opt2 > best_radius:
                best_radius = r_opt2
                best_centers = optimized2.copy()
                
    return best_centers
# EVOLVE-BLOCK-END
