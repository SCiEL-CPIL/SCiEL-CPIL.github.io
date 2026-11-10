"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: 150s limit, keep 5s margin
DEADLINE = 150.0
MARGIN = 5.0

def _compute_radius(centers):
    """
    Compute the maximum possible radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if centers.shape[0] < 2:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundaries
    # Square is [-0.5, 0.5] x [-0.5, 0.5]
    # Distance to nearest boundary for point (x, y) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
    # Which is 0.5 - max(|x|, |y|)
    max_coord = np.max(np.abs(centers), axis=1)
    min_boundary_dist = np.min(0.5 - max_coord)
    
    return min(min_dist / 2.0, min_boundary_dist)

def _negative_radius(centers):
    return -_compute_radius(centers)

def _hexagonal_packing(n):
    """
    Generate a hexagonal packing of n points in a unit square.
    """
    # Approximate number of rows
    # Hexagonal packing density is pi / (2 * sqrt(3)) ~ 0.9069
    # Area per circle ~ 1/n
    # r ~ 0.5 / sqrt(n * density * 4/pi) ... simpler to just iterate
    # Let's estimate m such that m * ceil(m/sqrt(3)) >= n
    # Actually, let's just generate a large grid and trim
    
    # Start with a reasonable guess for the grid size
    # For n=200, sqrt(200) ~ 14.1
    # Hex grid: rows have alternating counts or same count with offset
    # Let's try to fit into [-0.5, 0.5]
    
    # We want to maximize the minimum distance.
    # Let's use a simple approach: create a grid of size m x m, 
    # but with hexagonal offset.
    
    # Estimate m
    m = int(np.ceil(np.sqrt(n)))
    
    # Generate points
    points = []
    # Vertical spacing dy, horizontal spacing dx
    # In hex packing, dy = sqrt(3)/2 * dx
    # Let's pick dx such that we fit m points horizontally
    # Total width ~ m * dx
    # Total height ~ (m-1) * dy + dy = m * dy? No, (m-1)*dy
    
    # Let's just generate a large candidate set and pick the best n?
    # No, let's construct a specific hex grid that fits in the square.
    
    # Let dx be the horizontal distance.
    # We want to fit m columns.
    # Width = (m-1)*dx
    # Height = (m-1)*sqrt(3)/2 * dx
    # To fit in [-0.5, 0.5], we need (m-1)*dx <= 1 and (m-1)*sqrt(3)/2 * dx <= 1
    # So dx <= 1/(m-1) and dx <= 2/(sqrt(3)*(m-1))
    # Since 2/sqrt(3) > 1, the height constraint is looser.
    # So dx = 1/(m-1) is a good start, but we can scale up if we reduce m.
    
    # Let's iterate m from 14 down to 10 to find a good fit
    best_points = None
    best_r = 0
    
    for m in range(14, 8, -1):
        # dx = 1.0 / (m - 1)
        dx = 1.0 / (m - 1)
        dy = dx * np.sqrt(3) / 2
        
        # Check if it fits
        if (m - 1) * dy > 1.0:
            # Scale down
            scale = 1.0 / ((m - 1) * dy)
            dx *= scale
            dy *= scale
            
        # Generate points
        pts = []
        for row in range(m):
            for col in range(m):
                x = -0.5 + col * dx
                y = -0.5 + row * dy
                if row % 2 == 1:
                    x += dx / 2.0
                pts.append([x, y])
        pts = np.array(pts)
        
        # Filter points inside the square (with small margin for safety)
        # Actually, the construction above ensures x is in [-0.5, 0.5] if we don't shift out
        # But shifting x += dx/2 might push it out.
        mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
        pts = pts[mask]
        
        if len(pts) >= n:
            # Take the first n points? Or the ones closest to center?
            # Taking first n might be biased. Let's take a subset that is well-distributed.
            # For now, just take the first n.
            subset = pts[:n]
            r = _compute_radius(subset)
            if r > best_r:
                best_r = r
                best_points = subset
        
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    return best_points

def _local_optimization(centers, max_time=140.0):
    """
    Perform local optimization to maximize the minimum distance.
    Uses Nelder-Mead on the negative radius function.
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    
    # We will run multiple short Nelder-Mead optimizations
    # Or one long one with a callback to check time?
    # Nelder-Mead doesn't have a great time callback in scipy, but we can break out.
    
    # Let's use a loop of steps
    # Actually, scipy.optimize.minimize with method='Nelder-Mead' can be slow.
    # Let's try a simpler gradient-free approach: 
    # Move each point slightly in the direction that increases the minimum distance?
    # This is complex.
    
    # Alternative: Use scipy.optimize.minimize with a custom objective that is differentiable?
    # The min distance function is not differentiable.
    
    # Let's stick to Nelder-Mead but run it on subsets or with limited iterations.
    # Given 200 points, the dimension is 400. Nelder-Mead in 400D is very slow.
    
    # Better strategy: 
    # 1. Identify the "bottleneck" pairs (closest pairs and points close to boundary).
    # 2. Move those points away from each other and towards the center.
    # This is a heuristic local search.
    
    # Let's implement a simple hill-climbing:
    # 1. Find the current minimum distance d_min and the pair (i, j) achieving it.
    # 2. Also find points close to boundary.
    # 3. Propose a move: move i and j apart, move boundary points inward.
    # 4. Accept if radius increases.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
            
        # Compute distances
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find min distance
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i == j:
            continue
        d_min = dists[i, j]
        
        # Find boundary violations (points too close to edge)
        max_coord = np.max(np.abs(current_centers), axis=1)
        boundary_dists = 0.5 - max_coord
        min_boundary_idx = np.argmin(boundary_dists)
        d_boundary = boundary_dists[min_boundary_idx]
        
        # Current radius is min(d_min/2, d_boundary)
        r = min(d_min / 2.0, d_boundary)
        
        # Determine which constraint is active
        if d_min / 2.0 < d_boundary:
            # Move i and j apart
            vec = current_centers[i] - current_centers[j]
            norm = np.linalg.norm(vec)
            if norm == 0:
                vec = np.array([1.0, 0.0])
                norm = 1.0
            else:
                vec /= norm
            
            # Move i by +vec * step, j by -vec * step
            step = 0.001
            new_i = current_centers[i] + vec * step
            new_j = current_centers[j] - vec * step
            
            # Check boundaries for new_i, new_j
            if np.max(np.abs(new_i)) > 0.5 or np.max(np.abs(new_j)) > 0.5:
                # Try smaller step or different move
                step = 0.0001
                new_i = current_centers[i] + vec * step
                new_j = current_centers[j] - vec * step
                if np.max(np.abs(new_i)) > 0.5 or np.max(np.abs(new_j)) > 0.5:
                    continue
                    
            new_centers = current_centers.copy()
            new_centers[i] = new_i
            new_centers[j] = new_j
            
            new_r = _compute_radius(new_centers)
            if new_r > r:
                current_centers = new_centers
            else:
                # If no improvement, try random small perturbations?
                # Or break if stuck?
                # Let's try a random perturbation of a random point
                k = np.random.randint(N)
                new_k = current_centers[k] + np.random.normal(0, 0.001, 2)
                if np.max(np.abs(new_k)) <= 0.5:
                    new_centers = current_centers.copy()
                    new_centers[k] = new_k
                    new_r = _compute_radius(new_centers)
                    if new_r > r:
                        current_centers = new_centers
        else:
            # Move boundary point towards center
            k = min_boundary_idx
            vec = -current_centers[k]
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vec /= norm
            step = 0.001
            new_k = current_centers[k] + vec * step
            if np.max(np.abs(new_k)) > 0.5:
                continue
            new_centers = current_centers.copy()
            new_centers[k] = new_k
            new_r = _compute_radius(new_centers)
            if new_r > r:
                current_centers = new_centers
            else:
                # Random perturbation
                m_idx = np.random.randint(N)
                new_m = current_centers[m_idx] + np.random.normal(0, 0.001, 2)
                if np.max(np.abs(new_m)) <= 0.5:
                    new_centers = current_centers.copy()
                    new_centers[m_idx] = new_m
                    new_r = _compute_radius(new_centers)
                    if new_r > r:
                        current_centers = new_centers
                        
        # Check for timeout again
        if time.time() - start_time > max_time:
            break
            
    return current_centers

def construct():
    start_time = time.time()
    # 1. Generate hexagonal packing
    centers = _hexagonal_packing(N)
    
    # 2. Local optimization
    # Reserve 5 seconds for safety
    time_left = DEADLINE - MARGIN - (time.time() - start_time)
    if time_left > 1:
        centers = _local_optimization(centers, max_time=time_left)
        
    # Final check
    if time.time() - start_time > DEADLINE - 1:
        # Return current best
        pass
        
    return centers

# EVOLVE-BLOCK-END
