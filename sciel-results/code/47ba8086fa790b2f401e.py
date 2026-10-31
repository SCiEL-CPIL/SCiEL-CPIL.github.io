"""Circle packing in unit square - 26 circles maximizing sum of radii.
Built on a Fast Marching Method distance-field greedy placement approach.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt, maximum_filter

N = 26

# EVOLVE-BLOCK-START

def _fast_marching_distance_field(grid_n=256):
    """
    Compute the exact Euclidean distance field from the square boundary
    using Fast Marching Method (via scipy's distance_transform_edt).
    Returns (dist_field, xs, ys) where dist_field[i,j] is the distance
    from grid point (xs[j], ys[i]) to the nearest boundary.
    """
    # Create binary mask: 1 inside the unit square, 0 outside
    # We want distance from each interior point to the boundary
    # distance_transform_edt gives distance to nearest zero
    # So we create a mask where boundary is 0 and interior is 1
    
    # Actually, distance_transform_edt computes distance from each point
    # to the nearest zero in the input. We want the distance from each
    # interior point to the boundary.
    
    # Create a large enough grid where the square is the "object"
    # and outside is "background"
    
    # Simpler approach: compute analytically for the square
    xs = np.linspace(0, 1, grid_n)
    ys = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(xs, ys)
    
    # Distance to boundary of unit square
    dist_to_left = X
    dist_to_right = 1.0 - X
    dist_to_bottom = Y
    dist_to_top = 1.0 - Y
    
    dist_field = np.minimum(np.minimum(dist_to_left, dist_to_right),
                            np.minimum(dist_to_bottom, dist_to_top))
    
    return dist_field, xs, ys

def _update_distance_field_with_circles(dist_field, xs, ys, centers, radii):
    """
    Update the distance field by subtracting the influence of placed circles.
    For each circle, the distance to its surface is sqrt(dx^2+dy^2) - r.
    The free space distance is the minimum of boundary distance and
    distances to all circle surfaces.
    """
    grid_n = dist_field.shape[0]
    free_dist = dist_field.copy()
    
    n_circ = len(centers)
    if n_circ == 0:
        return free_dist
    
    # Vectorized update for better performance
    X, Y = np.meshgrid(xs, ys)
    
    for i in range(n_circ):
        if radii[i] <= 0:
            continue
        cx, cy = centers[i]
        d_to_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        dist_to_surface = d_to_center - radii[i]
        free_dist = np.minimum(free_dist, dist_to_surface)
    
    free_dist = np.maximum(free_dist, 0.0)
    return free_dist

def _refine_circle_position(x, y, centers, radii, max_iter=100):
    """
    Refine a circle's position and radius using gradient-based local search.
    Finds the local maximum of the inscribed circle radius.
    """
    x, y = float(x), float(y)
    
    def _radius_at(px, py):
        r = min(px, 1.0 - px, py, 1.0 - py)
        for j in range(len(centers)):
            if radii[j] <= 0:
                continue
            d = np.sqrt((px - centers[j][0])**2 + (py - centers[j][1])**2)
            r = min(r, d - radii[j])
        return r
    
    r = _radius_at(x, y)
    if r <= 0:
        return x, y, 0.001
    
    for iteration in range(max_iter):
        best_r = r
        best_x, best_y = x, y
        improved = False
        
        # Multi-scale radial search
        for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
                nx = x + step * np.cos(angle)
                ny = y + step * np.sin(angle)
                
                if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                    continue
                
                nr = _radius_at(nx, ny)
                if nr > best_r + 1e-7:
                    best_r = nr
                    best_x, best_y = nx, ny
                    improved = True
        
        if not improved:
            break
        
        x, y = best_x, best_y
        r = best_r
    
    r = _radius_at(x, y)
    return x, y, max(r, 0.001)

def _find_best_candidates(free_dist, xs, ys, centers, radii, n_candidates=15, min_sep=0.025):
    """
    Find the best candidate positions for the next circle.
    Uses local maxima of the distance field, then refines each candidate.
    Returns list of (x, y, radius) sorted by radius descending.
    """
    grid_n = free_dist.shape[0]
    
    # Find local maxima using a sliding window
    # Use a window size that corresponds to min_sep in grid units
    sep_pixels = max(1, int(min_sep * grid_n))
    size = sep_pixels * 2 + 1
    
    max_filter = maximum_filter(free_dist, size=size)
    local_max_mask = (free_dist == max_filter) & (free_dist > 0.003)
    
    rows, cols = np.where(local_max_mask)
    
    candidates = []
    for r_idx, c_idx in zip(rows, cols):
        x = xs[c_idx]
        y = ys[r_idx]
        est_r = free_dist[r_idx, c_idx]
        candidates.append((x, y, est_r))
    
    # Sort by estimated radius
    candidates.sort(key=lambda t: t[2], reverse=True)
    
    # Refine top candidates
    refined = []
    for x, y, est_r in candidates[:min(len(candidates), 40)]:
        rx, ry, actual_r = _refine_circle_position(x, y, centers, radii)
        if actual_r > 0.003:
            refined.append((rx, ry, actual_r))
    
    # Remove duplicates
    unique = []
    for x, y, r in refined:
        is_unique = True
        for ux, uy, ur in unique:
            if np.sqrt((x - ux)**2 + (y - uy)**2) < min_sep:
                is_unique = False
                break
        if is_unique:
            unique.append((x, y, r))
    
    unique.sort(key=lambda t: t[2], reverse=True)
    return unique[:n_candidates]

def _greedy_fmm_placement(grid_n=256, n_circles=N, seed=42):
    """
    Fast Marching Method based greedy placement:
    1. Compute exact distance field from boundary
    2. Find local maxima, place largest inscribed circle there
    3. Update distance field by subtracting circle's influence
    4. Repeat until all circles placed
    """
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Initial distance field
    free_dist, xs, ys = _fast_marching_distance_field(grid_n)
    
    placed_centers = []
    placed_radii = []
    
    for i in range(n_circles):
        # Find best candidates
        min_sep = max(0.015, 0.04 - 0.001 * i)  # Decreasing separation as we fill
        candidates = _find_best_candidates(free_dist, xs, ys, placed_centers, placed_radii,
                                            n_candidates=12, min_sep=min_sep)
        
        if not candidates:
            # Fallback: try with smaller separation
            candidates = _find_best_candidates(free_dist, xs, ys, placed_centers, placed_radii,
                                                n_candidates=5, min_sep=0.01)
        
        if not candidates:
            # Last resort: random search
            rng = np.random.RandomState(seed + i)
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            for _ in range(200):
                x = rng.uniform(0.02, 0.98)
                y = rng.uniform(0.02, 0.98)
                r = min(x, 1-x, y, 1-y)
                for j in range(len(placed_centers)):
                    d = np.sqrt((x - placed_centers[j][0])**2 + (y - placed_centers[j][1])**2)
                    r = min(r, d - placed_radii[j])
                if r > best_r:
                    best_x, best_y, best_r = x, y, r
            
            if best_r < 0.003:
                best_x, best_y, best_r = 0.5, 0.5, 0.003
            
            centers[i] = [best_x, best_y]
            radii[i] = max(best_r * 0.999, 0.002)
            placed_centers.append(centers[i])
            placed_radii.append(radii[i])
            free_dist = _update_distance_field_with_circles(free_dist, xs, ys,
                                                              np.array(placed_centers),
                                                              np.array(placed_radii))
            continue
        
        # Pick the best candidate
        best_x, best_y, best_r = candidates[0]
        
        # Safety margin
        actual_r = best_r * 0.998
        
        # Verify against boundary
        max_r_boundary = min(best_x, 1 - best_x, best_y, 1 - best_y)
        actual_r = min(actual_r, max_r_boundary * 0.9995)
        
        # Verify against existing circles
        for j in range(len(placed_centers)):
            dx = best_x - placed_centers[j][0]
            dy = best_y - placed_centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r_circle = dist - placed_radii[j]
            if max_r_circle < actual_r:
                actual_r = max_r_circle * 0.9995
        
        actual_r = max(actual_r, 0.002)
        
        centers[i] = [best_x, best_y]
        radii[i] = actual_r
        placed_centers.append(centers[i])
        placed_radii.append(radii[i])
        
        # Update distance field
        free_dist = _update_distance_field_with_circles(
            free_dist, xs, ys,
            np.array(placed_centers), np.array(placed_radii)
        )
    
    return centers, radii

def _greedy_with_refinement(grid_n=256, n_circles=N, seed=42):
    """
    Enhanced greedy: after initial FMM placement, refine each circle's
    position and radius to maximize the total sum.
    """
    centers, radii = _greedy_fmm_placement(grid_n, n_circles, seed)
    
    # Local refinement: for each circle, try to improve its position/radius
    for i in range(n_circles):
        improved = True
        iters = 0
        while improved and iters < 20:
            improved = False
            iters += 1
            best_r = radii[i]
            best_x, best_y = centers[i][0], centers[i][1]
            
            # Multi-scale local search
            for step in [0.04, 0.02, 0.01, 0.005, 0.002]:
                for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                    nx = centers[i][0] + step * np.cos(angle)
                    ny = centers[i][1] + step * np.sin(angle)
                    
                    if nx < 0.001 or nx > 0.999 or ny < 0.001 or ny > 0.999:
                        continue
                    
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
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
                radii[i] = best_r * 0.9995
    
    # Grow pass: try to increase radii where possible
    for _ in range(15):
        any_grew = False
        for i in range(n_circles):
            target_r = radii[i] * 1.005
            can_grow = True
            for j in range(n_circles):
                if j == i:
                    continue
                d = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                if d < target_r + radii[j] - 1e-8:
                    can_grow = False
                    break
            if can_grow and min(centers[i][0], 1-centers[i][0],
                                centers[i][1], 1-centers[i][1]) >= target_r - 1e-8:
                radii[i] = target_r
                any_grew = True
        if not any_grew:
            break
    
    return centers, radii

def _initial_corner_focus():
    """Initialize circles focusing on corners and edges where larger circles can fit."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with corner circles
    corner_positions = [
        (0.13, 0.13), (0.87, 0.13), (0.13, 0.87), (0.87, 0.87)
    ]
    corner_radii = [0.125, 0.125, 0.125, 0.125]
    
    for i in range(4):
        pos[i] = corner_positions[i]
        radii[i] = corner_radii[i]
    
    # Edge midpoints
    edge_positions = [
        (0.5, 0.065), (0.5, 0.935), (0.065, 0.5), (0.935, 0.5)
    ]
    edge_radii = [0.06, 0.06, 0.06, 0.06]
    
    for i in range(4, 8):
        pos[i] = edge_positions[i-4]
        radii[i] = edge_radii[i-4]
    
    # Remaining: greedy placement in center
    rng = np.random.RandomState(42)
    for k in range(8, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(800):
            x = rng.uniform(0.15, 0.85)
            y = rng.uniform(0.15, 0.85)
            r = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x - pos[j,0])**2 + (y - pos[j,1])**2)
                r = min(r, d - radii[j])
            if r > best_r:
                best_x, best_y, best_r = x, y, r
        if best_r > 0.008:
            pos[k] = [best_x, best_y]
            radii[k] = best_r * 0.97
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.008
    
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

def _simulated_annealing(pos, radii, deadline, n_iter=5000):
    """
    Simulated annealing to improve the packing.
    Tries random moves and accepts worse solutions with decreasing probability.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    T = 0.04  # Initial temperature
    T_min = 0.0005
    cooling_rate = 0.9997
    
    for it in range(n_iter):
        if time.time() > deadline - 2:
            break
        
        # Pick a random circle to move
        i = np.random.randint(N)
        old_x, old_y = pos[i]
        old_r = radii[i]
        
        # Random perturbation - try to grow the circle
        scale = T * 0.6
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        # Bias toward growing
        dr = np.random.normal(0, scale * 0.4) + scale * 0.1
        
        new_x = old_x + dx
        new_y = old_y + dy
        new_r = max(old_r + dr, 0.003)
        
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
            new_score = np.sum(radii) - old_r + new_r
            delta = new_score - best_score
            
            if delta > 0:
                # Always accept better solutions
                pos[i] = [new_x, new_y]
                radii[i] = new_r
                best_pos = pos.copy()
                best_radii = radii.copy()
                best_score = new_score
            else:
                # Accept worse solutions with probability exp(delta/T)
                if T > T_min and np.random.random() < np.exp(delta / T):
                    pos[i] = [new_x, new_y]
                    radii[i] = new_r
        
        T = max(T * cooling_rate, T_min)
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with L-BFGS-B and simulated annealing."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.003, 0.997))
        bounds.append((0.003, 0.997))
    for i in range(N):
        bounds.append((0.003, 0.45))
        
    # L-BFGS-B with increasing penalty
    for p_w in [500.0, 2000.0, 8000.0, 30000.0]:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(p_w,),
                              options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < _objective(x, p_w) - 1e-14:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    # Simulated annealing for further improvement
    if time.time() < deadline - 3:
        try:
            sa_pos, sa_radii = _simulated_annealing(pos_out, radii_out, deadline, n_iter=4000)
            sa_score = np.sum(sa_radii)
            lbfgs_score = np.sum(radii_out)
            if sa_score > lbfgs_score:
                pos_out, radii_out = sa_pos, sa_radii
        except Exception:
            pass
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # Strategy 1: FMM-based greedy placement at multiple grid resolutions
    for grid_n in [192, 256, 320]:
        if time.time() > deadline - 50:
            break
        try:
            pos_gd, rad_gd = _greedy_fmm_placement(grid_n, N)
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
    if time.time() < deadline - 35:
        try:
            pos_gr, rad_gr = _greedy_with_refinement(256, N)
            pos_gr, rad_gr = _post_process(pos_gr, rad_gr)
            if _is_valid(pos_gr, rad_gr):
                score = np.sum(rad_gr)
                if score > best_score:
                    best_score = score
                    best_pos = pos_gr.copy()
                    best_radii = rad_gr.copy()
        except Exception:
            pass
    
    # Strategy 3: Corner-focused initialization
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
    
    # Strategy 4: Random initializations
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
    
    # Final polish
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
    if time.time() < deadline - 4:
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
