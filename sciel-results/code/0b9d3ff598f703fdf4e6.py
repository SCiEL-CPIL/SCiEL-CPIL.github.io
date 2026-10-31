"""Circle packing in unit square - 26 circles maximizing sum of radii.
Built on a Fast Marching / distance-field greedy placement approach.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt, maximum_filter

N = 26

def _compute_distance_field_fast(centers, radii, grid_n=250):
    """
    Compute the Euclidean distance field using scipy's exact distance transform.
    Returns a 2D array of the 'free space' distance.
    """
    xs = np.linspace(0, 1, grid_n)
    ys = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(xs, ys)
    
    # Distance to square boundary (positive inside)
    free_dist = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
    
    # For each existing circle, compute distance to its surface and take minimum
    n_circ = len(centers)
    if n_circ > 0:
        for i in range(n_circ):
            if radii[i] <= 0:
                continue
            cx, cy = centers[i]
            d_to_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            dist_to_surface = d_to_center - radii[i]
            free_dist = np.minimum(free_dist, dist_to_surface)
    
    free_dist = np.maximum(free_dist, 0.0)
    return free_dist, xs, ys

# EVOLVE-BLOCK-START

def _compute_distance_field(centers, radii, grid_n=200):
    """
    Compute the Euclidean distance field from the square boundary and
    subtract the influence of existing circles.
    Returns a 2D array of the 'free space' distance.
    """
    xs = np.linspace(0, 1, grid_n)
    ys = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(xs, ys)
    
    # Distance to square boundary (positive inside)
    free_dist = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
    
    # For each existing circle, compute distance to its surface and take minimum
    n_circ = len(centers)
    if n_circ > 0:
        for i in range(n_circ):
            if radii[i] <= 0:
                continue
            cx, cy = centers[i]
            d_to_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            dist_to_surface = d_to_center - radii[i]
            free_dist = np.minimum(free_dist, dist_to_surface)
    
    free_dist = np.maximum(free_dist, 0.0)
    return free_dist, xs, ys

def _max_radius_at(x, y, centers, radii):
    """Compute the maximum radius for a circle at position (x,y)."""
    r = min(x, 1 - x, y, 1 - y)
    n = len(centers)
    if n > 0:
        dx = x - centers[:, 0]
        dy = y - centers[:, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        r = min(r, np.min(dists - radii))
    return max(r, 0.0)

def _max_radius_at_vectorized(x, y, centers, radii):
    """Vectorized max radius computation for speed."""
    r = min(x, 1 - x, y, 1 - y)
    n = len(centers)
    if n > 0:
        dx = x - centers[:, 0]
        dy = y - centers[:, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        r = min(r, np.min(dists - radii))
    return max(r, 0.0)

def _refine_position(x, y, centers, radii, max_iter=40):
    """
    Refine a position to find the exact maximum inscribed circle radius
    at that location using iterative gradient ascent with adaptive steps.
    """
    for _ in range(max_iter):
        r = _max_radius_at(x, y, centers, radii)
        
        if r <= 0:
            return x, y, 0.001
        
        best_r = r
        best_x, best_y = x, y
        # Adaptive step sizes: start large, shrink
        step_sizes = [0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        
        for step in step_sizes:
            for angle in np.linspace(0, 2*np.pi, 32, endpoint=False):
                nx = x + step * np.cos(angle)
                ny = y + step * np.sin(angle)
                
                if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                    continue
                
                nr = _max_radius_at(nx, ny, centers, radii)
                
                if nr > best_r + 1e-7:
                    best_r = nr
                    best_x, best_y = nx, ny
        
        if abs(best_x - x) < 1e-6 and abs(best_y - y) < 1e-6:
            break
        x, y = best_x, best_y
    
    r = _max_radius_at(x, y, centers, radii)
    return x, y, max(r, 0.001)

def _refine_all_positions(pos, radii, centers_excl, radii_excl, max_iter=20):
    """
    Refine all positions simultaneously with multiple passes.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(len(pos)):
            cx, cy = pos[i]
            # Build centers/radii excluding circle i
            c_idx = [j for j in range(len(centers_excl)) if j != i]
            centers = np.array([centers_excl[j] for j in c_idx]) if len(c_idx) > 0 else np.empty((0, 2))
            radii_c = np.array([radii_excl[j] for j in c_idx]) if len(c_idx) > 0 else np.empty(0)
            
            nx, ny, nr = _refine_position(cx, cy, centers, radii_c)
            if nr > radii[i] + 1e-6:
                pos[i] = [nx, ny]
                radii[i] = nr
                improved = True
        
        if not improved:
            break
    
    return pos, radii

def _find_local_maxima(free_dist, min_distance=5):
    """
    Find local maxima in the distance field using a sliding window maximum filter.
    Returns list of (x, y, r) candidates where r is the local max value.
    """
    grid_n = free_dist.shape[0]
    
    size = min_distance * 2 + 1
    max_filter = maximum_filter(free_dist, size=size)
    local_max_mask = (free_dist == max_filter) & (free_dist > 0.005)
    
    rows, cols = np.where(local_max_mask)
    candidates = []
    for r, c in zip(rows, cols):
        x = c / (grid_n - 1)
        y = r / (grid_n - 1)
        radius = free_dist[r, c]
        candidates.append((x, y, radius))
    
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates

def _find_multiple_candidates(free_dist, centers, radii, n_candidates=20, min_distance=3):
    """
    Find multiple good candidate positions for placing the next circle.
    Uses the distance field but also considers actual geometric constraints.
    """
    grid_n = free_dist.shape[0]
    
    size = min_distance * 2 + 1
    max_filter = maximum_filter(free_dist, size=size)
    local_max_mask = (free_dist == max_filter) & (free_dist > 0.005)
    
    rows, cols = np.where(local_max_mask)
    candidates = []
    
    for r, c in zip(rows, cols):
        x = c / (grid_n - 1)
        y = r / (grid_n - 1)
        est_radius = free_dist[r, c]
        candidates.append((x, y, est_radius))
    
    # Sort by estimated radius
    candidates.sort(key=lambda t: t[2], reverse=True)
    
    # Refine top candidates and keep the best ones
    refined = []
    for x, y, est_r in candidates[:80]:
        rx, ry, actual_r = _refine_position(x, y, centers, radii)
        if actual_r > 0.005:
            refined.append((rx, ry, actual_r))
    
    # Remove duplicates (positions too close together)
    unique_refined = []
    for x, y, r in refined:
        is_unique = True
        for ux, uy, ur in unique_refined:
            if np.sqrt((x - ux)**2 + (y - uy)**2) < 0.015:
                is_unique = False
                break
        if is_unique:
            unique_refined.append((x, y, r))
    
    unique_refined.sort(key=lambda t: t[2], reverse=True)
    return unique_refined[:n_candidates]

def _greedy_distance_field_placement(grid_n=250, n_circles=N, seed=42):
    """
    Greedy placement using distance field:
    1. Compute initial distance field from boundary
    2. Find local maxima, place largest circle there
    3. Update distance field by subtracting circle influence
    4. Repeat until all circles placed or no more valid locations
    """
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Initial distance field
    free_dist, xs, ys = _compute_distance_field_fast(np.empty(0), np.empty(0), grid_n)
    
    placed_centers = []
    placed_radii = []
    
    for i in range(n_circles):
        # Find multiple candidates and pick the best
        min_dist = max(2, int(0.02 * grid_n))
        candidates = _find_multiple_candidates(free_dist, placed_centers, placed_radii, 
                                                n_candidates=20, min_distance=min_dist)
        
        if not candidates:
            # Try with smaller min_distance
            candidates = _find_local_maxima(free_dist, min_distance=1)
            if candidates:
                candidates = [candidates[0]]
        
        if not candidates:
            # No more valid locations; use small fallback
            centers[i] = [0.5, 0.5]
            radii[i] = 0.005
            placed_centers.append(centers[i])
            placed_radii.append(radii[i])
            continue
        
        # Pick the best candidate (largest radius)
        best_x, best_y, best_r = candidates[0]
        
        # Verify and adjust: ensure the circle fits
        actual_r = best_r * 0.998
        
        # Check against boundary
        max_r_boundary = min(best_x, 1 - best_x, best_y, 1 - best_y)
        actual_r = min(actual_r, max_r_boundary * 0.999)
        
        # Check against existing circles
        for j in range(len(placed_centers)):
            dx = best_x - placed_centers[j][0]
            dy = best_y - placed_centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r_circle = dist - placed_radii[j]
            if max_r_circle < actual_r:
                actual_r = max_r_circle * 0.999
        
        actual_r = max(actual_r, 0.005)
        
        centers[i] = [best_x, best_y]
        radii[i] = actual_r
        placed_centers.append(centers[i])
        placed_radii.append(radii[i])
        
        # Update distance field
        free_dist, xs, ys = _compute_distance_field_fast(
            np.array(placed_centers), np.array(placed_radii), grid_n
        )
    
    return centers, radii

def _greedy_with_refinement(grid_n=250, n_circles=N, seed=42):
    """
    Enhanced greedy: after initial placement, try to refine positions
    by local search to increase radii.
    """
    centers, radii = _greedy_distance_field_placement(grid_n, n_circles, seed)
    
    # Local refinement: for each circle, try to grow it by moving it slightly
    for pass_num in range(5):
        for i in range(n_circles):
            improved = True
            while improved:
                improved = False
                best_r = radii[i]
                best_x, best_y = centers[i]
                
                # Multi-scale local search
                for step in [0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
                    for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
                        nx = centers[i][0] + step * np.cos(angle)
                        ny = centers[i][1] + step * np.sin(angle)
                        
                        # Check boundary
                        if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                            continue
                        
                        max_r = min(nx, 1 - nx, ny, 1 - ny)
                        # Check other circles
                        for j in range(n_circles):
                            if j == i:
                                continue
                            d = np.sqrt((nx - centers[j][0])**2 + (ny - centers[j][1])**2)
                            max_r = min(max_r, d - radii[j])
                        
                        if max_r > best_r + 1e-6:
                            best_r = max_r
                            best_x, best_y = nx, ny
                            improved = True
                
                if improved:
                    centers[i] = [best_x, best_y]
                    radii[i] = best_r * 0.999
    
    # Second pass: try to grow all circles simultaneously
    for _ in range(30):
        any_grew = False
        for i in range(n_circles):
            # Try to grow circle i by shrinking others slightly if needed
            target_r = radii[i] * 1.005
            # Check if we can grow without violating constraints
            can_grow = True
            for j in range(n_circles):
                if j == i:
                    continue
                d = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                if d < target_r + radii[j]:
                    can_grow = False
                    break
            if can_grow and min(centers[i][0], 1-centers[i][0], centers[i][1], 1-centers[i][1]) >= target_r:
                radii[i] = target_r
                any_grew = True
        if not any_grew:
            break
    
    return centers, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization as fallback."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.015:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.98
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.008
            k += 1
            
    return pos, radii

def _initial_corner_focus():
    """Initialize circles focusing on corners and edges where larger circles can fit."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with corner circles
    corner_positions = [
        (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)
    ]
    corner_radii = [0.14, 0.14, 0.14, 0.14]
    
    for i in range(4):
        pos[i] = corner_positions[i]
        radii[i] = corner_radii[i]
    
    # Edge centers
    edge_positions = [
        (0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)
    ]
    edge_radii = [0.07, 0.07, 0.07, 0.07]
    
    for i in range(4, 8):
        pos[i] = edge_positions[i-4]
        radii[i] = edge_radii[i-4]
    
    # Center region
    remaining = N - 8
    rng = np.random.RandomState(42)
    for k in range(8, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(500):
            x = rng.uniform(0.2, 0.8)
            y = rng.uniform(0.2, 0.8)
            r = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x - pos[j,0])**2 + (y - pos[j,1])**2)
                r = min(r, d - radii[j])
            if r > best_r:
                best_x, best_y, best_r = x, y, r
        if best_r > 0.01:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.98
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
    
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _objective(x, penalty_weight=100.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _simulated_annealing(pos, radii, deadline, n_iter=8000):
    """
    Simulated annealing to improve the packing.
    Tries random moves and accepts worse solutions with decreasing probability.
    """
    pos = pos.copy()
    radii = radii.copy()
    cur_score = np.sum(radii)
    best_score = cur_score
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    T = 0.03  # Initial temperature
    T_min = 0.0005
    cooling_rate = 0.9997
    
    for it in range(n_iter):
        if time.time() > deadline - 2:
            break
        
        # Pick a random circle to move
        i = np.random.randint(N)
        old_x, old_y = pos[i]
        old_r = radii[i]
        
        # Random perturbation - position move
        scale = T * 0.8
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Determine max radius at new position
        new_r = _max_radius_at(new_x, new_y, pos, radii)
        # Optionally try to keep similar radius or grow
        new_r = min(new_r, old_r * 1.15)
        new_r = max(new_r, 0.005)
        
        # Check if move is valid
        valid = True
        # Boundary check
        if new_x - new_r < -1e-7 or new_x + new_r > 1 + 1e-7:
            valid = False
        if new_y - new_r < -1e-7 or new_y + new_r > 1 + 1e-7:
            valid = False
        
        if valid:
            # Overlap check
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((new_x - pos[j,0])**2 + (new_y - pos[j,1])**2)
                if d < new_r + radii[j] - 1e-7:
                    valid = False
                    break
        
        if valid:
            new_score = cur_score - old_r + new_r
            delta = new_score - cur_score
            
            if delta > 0:
                # Always accept better solutions
                pos[i] = [new_x, new_y]
                radii[i] = new_r
                cur_score = new_score
                if cur_score > best_score:
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    best_score = cur_score
            else:
                # Accept worse solutions with probability exp(delta/T)
                if T > T_min:
                    ratio = np.exp(delta / T)
                    if np.random.random() < ratio:
                        pos[i] = [new_x, new_y]
                        radii[i] = new_r
                        cur_score = new_score
        
        T = max(T * cooling_rate, T_min)
    
    return best_pos, best_radii

def _local_search_swap_grow(pos, radii, deadline, n_iter=2000):
    """
    Local search that tries to swap positions of two circles and then
    grow both to their maximum possible radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    cur_score = np.sum(radii)
    best_score = cur_score
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for it in range(n_iter):
        if time.time() > deadline - 2:
            break
        
        # Pick two random circles
        i, j = np.random.choice(N, 2, replace=False)
        
        # Save old state
        old_pos_i, old_pos_j = pos[i].copy(), pos[j].copy()
        old_r_i, old_r_j = radii[i], radii[j]
        
        # Swap positions
        pos[i], pos[j] = pos[j].copy(), pos[i].copy()
        
        # Try to grow both circles to their max radius
        # First grow circle i
        new_r_i = _max_radius_at(pos[i, 0], pos[i, 1], pos, radii)
        # Then grow circle j (with updated radius for i)
        radii[i] = min(new_r_i, old_r_i * 1.2)
        new_r_j = _max_radius_at(pos[j, 0], pos[j, 1], pos, radii)
        radii[j] = min(new_r_j, old_r_j * 1.2)
        
        # Check validity
        valid = True
        for k in range(N):
            if pos[k, 0] - radii[k] < -1e-7 or pos[k, 0] + radii[k] > 1 + 1e-7:
                valid = False
                break
            if pos[k, 1] - radii[k] < -1e-7 or pos[k, 1] + radii[k] > 1 + 1e-7:
                valid = False
                break
        if valid:
            for k in range(N):
                for l in range(k+1, N):
                    d = np.sqrt((pos[k,0]-pos[l,0])**2 + (pos[k,1]-pos[l,1])**2)
                    if d < radii[k] + radii[l] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
        
        new_score = np.sum(radii)
        if valid and new_score > cur_score:
            cur_score = new_score
            if cur_score > best_score:
                best_pos = pos.copy()
                best_radii = radii.copy()
                best_score = cur_score
        else:
            # Revert
            pos[i], pos[j] = old_pos_i, old_pos_j
            radii[i], radii[j] = old_r_i, old_r_j
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with L-BFGS-B and simulated annealing."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.45))
        
    # L-BFGS-B with increasing penalty
    for p_w in [100.0, 500.0, 2000.0, 8000.0, 20000.0]:
        if time.time() > deadline - 8:
            break
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(p_w,),
                              options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
            if result.fun < _objective(x, p_w) - 1e-14:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    # Simulated annealing for further improvement
    if time.time() < deadline - 5:
        try:
            sa_pos, sa_radii = _simulated_annealing(pos_out, radii_out, deadline, n_iter=6000)
            sa_score = np.sum(sa_radii)
            lbfgs_score = np.sum(radii_out)
            if sa_score > lbfgs_score:
                pos_out, radii_out = sa_pos, sa_radii
        except Exception:
            pass
    
    # Local search with swap and grow
    if time.time() < deadline - 3:
        try:
            ls_pos, ls_radii = _local_search_swap_grow(pos_out, radii_out, deadline, n_iter=1500)
            ls_score = np.sum(ls_radii)
            if ls_score > np.sum(radii_out):
                pos_out, radii_out = ls_pos, ls_radii
        except Exception:
            pass
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Strategy 1: Distance field greedy placement
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Try different grid resolutions
    for grid_n in [150, 200, 250, 300]:
        if time.time() > deadline - 50:
            break
        try:
            pos_gd, rad_gd = _greedy_distance_field_placement(grid_n, N)
            pos_gd, rad_gd = _post_process(pos_gd, rad_gd)
            if _is_valid(pos_gd, rad_gd):
                score = np.sum(rad_gd)
                if score > best_score:
                    best_score = score
                    best_pos = pos_gd.copy()
                    best_radii = rad_gd.copy()
        except Exception:
            pass
    
    # Strategy 2: Greedy with refinement
    if time.time() < deadline - 30:
        try:
            pos_gr, rad_gr = _greedy_with_refinement(250, N)
            pos_gr, rad_gr = _post_process(pos_gr, rad_gr)
            if _is_valid(pos_gr, rad_gr):
                score = np.sum(rad_gr)
                if score > best_score:
                    best_score = score
                    best_pos = pos_gr.copy()
                    best_radii = rad_gr.copy()
        except Exception:
            pass
    
    # Strategy 3: Hexagonal fallback
    if time.time() < deadline - 30:
        try:
            pos_h, rad_h = _initial_hexagonal()
            pos_h, rad_h = _post_process(pos_h, rad_h)
            if _is_valid(pos_h, rad_h):
                score = np.sum(rad_h)
                if score > best_score:
                    best_score = score
                    best_pos = pos_h.copy()
                    best_radii = rad_h.copy()
        except Exception:
            pass
    
    # Strategy 4: Corner-focused initialization
    if time.time() < deadline - 30:
        try:
            pos_c, rad_c = _initial_corner_focus()
            pos_c, rad_c = _post_process(pos_c, rad_c)
            if _is_valid(pos_c, rad_c):
                score = np.sum(rad_c)
                if score > best_score:
                    best_score = score
                    best_pos = pos_c.copy()
                    best_radii = rad_c.copy()
        except Exception:
            pass
    
    # Strategy 5: Random initializations
    for seed in [42, 123, 789]:
        if time.time() > deadline - 25:
            break
        try:
            pos_r, rad_r = _initial_random(seed)
            if _is_valid(pos_r, rad_r):
                score = np.sum(rad_r)
                if score > best_score:
                    best_score = score
                    best_pos = pos_r.copy()
                    best_radii = rad_r.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a safe default
    if best_pos is None:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Final polish - run multiple times with different starting points
    if time.time() < deadline - 8:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pol, rad_pol = _post_process(pos_pol, rad_pol)
            if _is_valid(pos_pol, rad_pol):
                score = np.sum(rad_pol)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pol.copy()
                    best_radii = rad_pol.copy()
        except Exception:
            pass
    
    # Second polish pass if time remains
    if time.time() < deadline - 5:
        try:
            pos_pol2, rad_pol2 = _polish(best_pos, best_radii, deadline)
            pos_pol2, rad_pol2 = _post_process(pos_pol2, rad_pol2)
            if _is_valid(pos_pol2, rad_pol2):
                score = np.sum(rad_pol2)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pol2.copy()
                    best_radii = rad_pol2.copy()
        except Exception:
            pass
    
    # Third polish pass if time remains
    if time.time() < deadline - 3:
        try:
            pos_pol3, rad_pol3 = _polish(best_pos, best_radii, deadline)
            pos_pol3, rad_pol3 = _post_process(pos_pol3, rad_pol3)
            if _is_valid(pos_pol3, rad_pol3):
                score = np.sum(rad_pol3)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pol3.copy()
                    best_radii = rad_pol3.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
