"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Uses hexagonal lattice with affine deformation optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _hexagonal_positions(n, spacing=1.0):
    """Generate hexagonal lattice positions for n circles.
    Returns (n, 2) array of positions in a local coordinate system.
    Uses a proper hexagonal close-packed arrangement.
    """
    positions = []
    row_spacing = spacing * np.sqrt(3) / 2
    row = 0
    
    # For hexagonal packing, alternate rows have different numbers of circles
    # We'll fill rows greedily to get exactly n circles
    while len(positions) < n:
        # Determine how many circles to place in this row
        # Even rows: full row, odd rows: offset by half spacing
        remaining = n - len(positions)
        # Place up to remaining circles in this row
        cols = remaining
        
        for col in range(cols):
            x = col * spacing + (row % 2) * spacing / 2
            y = row * row_spacing
            positions.append([x, y])
            if len(positions) >= n:
                break
        row += 1
    
    return np.array(positions[:n])


def _compute_max_radius(positions, W, H):
    """Compute the maximum uniform radius such that all circles fit in W x H rectangle.
    positions: (n, 2) array of centers.
    Returns the max r such that all circles fit and are disjoint.
    """
    n = len(positions)
    if n == 0 or W <= 0 or H <= 0:
        return 0.0
    
    # Constraint 1: each circle must fit in rectangle (vectorized)
    min_x = np.min(positions[:, 0])
    max_x = np.max(positions[:, 0])
    min_y = np.min(positions[:, 1])
    max_y = np.max(positions[:, 1])
    
    r_boundary = min(min_x, W - max_x, min_y, H - max_y)
    
    # Constraint 2: pairwise distances (vectorized)
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    r_pair = np.min(dists) / 2.0
    
    return max(0.0, min(r_boundary, r_pair))


def _validate_uniform(circles, radii, W, H):
    """Check if circles are valid (non-overlapping, within bounds)."""
    n = len(circles)
    # Check bounds
    for i in range(n):
        if circles[i, 0] < radii[i] - 1e-6 or circles[i, 0] > W - radii[i] + 1e-6:
            return False
        if circles[i, 1] < radii[i] - 1e-6 or circles[i, 1] > H - radii[i] + 1e-6:
            return False
        if radii[i] < -1e-6:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((circles[i, 0] - circles[j, 0])**2 + 
                       (circles[i, 1] - circles[j, 1])**2)
            if d < radii[i] + radii[j] - 1e-8:
                return False
    return True


def _apply_affine(positions, sx, sy, shear):
    """Apply affine transformation to positions.
    sx: x-scale, sy: y-scale, shear: x-shear based on y
    """
    transformed = np.zeros_like(positions)
    transformed[:, 0] = positions[:, 0] * sx + positions[:, 1] * shear
    transformed[:, 1] = positions[:, 1] * sy
    return transformed


def _pack_hexagonal(W, H, deadline, n_circles=N):
    """Pack circles in W x H rectangle using hexagonal lattice with direct position optimization."""
    start_time = time.time()
    
    # Generate initial hexagonal positions scaled to fit roughly in W x H
    base_positions = _hexagonal_positions(n_circles, spacing=1.0)
    
    # Normalize to unit bounding box
    min_x, min_y = np.min(base_positions, axis=0)
    max_x, max_y = np.max(base_positions, axis=0)
    base_w = max_x - min_x
    base_h = max_y - min_y
    
    if base_w <= 0 or base_h <= 0:
        # Fallback to grid
        cols = max(1, int(np.round(np.sqrt(n_circles * W / H))))
        rows = max(1, int(np.ceil(n_circles / cols)))
        grid_positions = np.zeros((n_circles, 2))
        for i in range(n_circles):
            c = i % cols
            r_idx = i // cols
            grid_positions[i, 0] = (c + 0.5) * W / cols
            grid_positions[i, 1] = (r_idx + 0.5) * H / rows
        r = _compute_max_radius(grid_positions, W, H)
        if r > 0:
            circles = grid_positions.copy()
            radii = np.full(n_circles, r)
            if _validate_uniform(circles, radii, W, H):
                return np.column_stack([circles, radii]), np.sum(radii)
        return None, 0.0
    
    # Scale and center to fit in W x H with some margin
    scale = min(W / (base_w + 0.5), H / (base_h + 0.5))
    positions = base_positions * scale
    positions[:, 0] += (W - positions[:, 0].max() + positions[:, 0].min()) / 2
    positions[:, 1] += (H - positions[:, 1].max() + positions[:, 1].min()) / 2
    
    # Direct position optimization with smooth objective
    def smooth_objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x.reshape(n_circles, 2)
        
        # Soft penalty for going out of bounds
        bound_penalty = 0.0
        for i in range(n_circles):
            if pos[i, 0] < 0:
                bound_penalty += (pos[i, 0])**2
            if pos[i, 0] > W:
                bound_penalty += (pos[i, 0] - W)**2
            if pos[i, 1] < 0:
                bound_penalty += (pos[i, 1])**2
            if pos[i, 1] > H:
                bound_penalty += (pos[i, 1] - H)**2
        
        # Compute min pairwise distance (smooth approximation using log-sum-exp)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Use soft-min: -log-sum-exp(-dists/sigma) / log(n_pairs)
        sigma = 0.01
        log_sum_exp = -np.log(np.sum(np.exp(-dists / sigma))) * sigma
        soft_min_dist = log_sum_exp / 2.0
        
        # Boundary constraint (soft min distance to walls)
        wall_dists = np.concatenate([
            pos[:, 0], W - pos[:, 0], pos[:, 1], H - pos[:, 1]
        ])
        wall_dists = np.clip(wall_dists, 0, None)
        soft_min_wall = -np.log(np.sum(np.exp(-wall_dists / sigma))) * sigma / 4.0
        
        # Objective: maximize min(soft_min_dist, soft_min_wall)
        # Use soft-min of the two
        combined = -np.log(np.exp(-soft_min_dist / sigma) + np.exp(-soft_min_wall / sigma)) * sigma
        
        return -combined + 10.0 * bound_penalty
    
    x0 = positions.flatten()
    
    best_result = None
    best_score = -np.inf
    
    # Try Nelder-Mead with multiple restarts
    for restart in range(3):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0
        else:
            # Add small perturbation
            x_start = x0 + np.random.randn(n_circles * 2) * 0.01 * W
        
        try:
            result = minimize(
                smooth_objective,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if time.time() < deadline:
                pos = result.x.reshape(n_circles, 2)
                # Clamp to bounds
                pos[:, 0] = np.clip(pos[:, 0], 0, W)
                pos[:, 1] = np.clip(pos[:, 1], 0, H)
                
                r = _compute_max_radius(pos, W, H)
                if r > 0:
                    circles = pos.copy()
                    radii = np.full(n_circles, r)
                    if _validate_uniform(circles, radii, W, H):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([circles, radii])
        except Exception:
            pass
    
    # Fallback to grid if optimization failed
    if best_result is None:
        cols = max(1, int(np.round(np.sqrt(n_circles * W / H))))
        rows = max(1, int(np.ceil(n_circles / cols)))
        grid_positions = np.zeros((n_circles, 2))
        for i in range(n_circles):
            c = i % cols
            r_idx = i // cols
            grid_positions[i, 0] = (c + 0.5) * W / cols
            grid_positions[i, 1] = (r_idx + 0.5) * H / rows
        r = _compute_max_radius(grid_positions, W, H)
        if r > 0:
            circles = grid_positions.copy()
            radii = np.full(n_circles, r)
            if _validate_uniform(circles, radii, W, H):
                return np.column_stack([circles, radii]), np.sum(radii)
    
    return best_result, best_score


def _pack_with_position_optimization(W, H, deadline):
    """Pack circles by optimizing positions directly with smooth objective."""
    start_time = time.time()
    
    # Initial grid placement
    cols = max(1, int(np.round(np.sqrt(N * W / H))))
    rows = max(1, int(np.ceil(N / cols)))
    
    positions = np.zeros((N, 2))
    for i in range(N):
        col = i % cols
        row = i // cols
        positions[i, 0] = (col + 0.5) * W / cols
        positions[i, 1] = (row + 0.5) * H / rows
    
    def smooth_objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x.reshape(N, 2)
        
        # Soft penalty for going out of bounds
        bound_penalty = 0.0
        for i in range(N):
            if pos[i, 0] < 0:
                bound_penalty += (pos[i, 0])**2
            if pos[i, 0] > W:
                bound_penalty += (pos[i, 0] - W)**2
            if pos[i, 1] < 0:
                bound_penalty += (pos[i, 1])**2
            if pos[i, 1] > H:
                bound_penalty += (pos[i, 1] - H)**2
        
        # Compute min pairwise distance (smooth approximation)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Soft-min using log-sum-exp
        sigma = 0.01
        soft_min_dist = -np.log(np.sum(np.exp(-dists / sigma))) * sigma / 2.0
        
        # Boundary constraint (soft min distance to walls)
        wall_dists = np.concatenate([
            pos[:, 0], W - pos[:, 0], pos[:, 1], H - pos[:, 1]
        ])
        wall_dists = np.clip(wall_dists, 0, None)
        soft_min_wall = -np.log(np.sum(np.exp(-wall_dists / sigma))) * sigma / 4.0
        
        # Combined soft-min
        combined = -np.log(np.exp(-soft_min_dist / sigma) + np.exp(-soft_min_wall / sigma)) * sigma
        
        return -combined + 10.0 * bound_penalty
    
    x0 = positions.flatten()
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple restarts
    for restart in range(2):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.randn(N * 2) * 0.01 * W
        
        try:
            result = minimize(
                smooth_objective,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if time.time() < deadline:
                pos = result.x.reshape(N, 2)
                pos[:, 0] = np.clip(pos[:, 0], 0, W)
                pos[:, 1] = np.clip(pos[:, 1], 0, H)
                
                r = _compute_max_radius(pos, W, H)
                if r > 0:
                    circles = pos.copy()
                    radii = np.full(N, r)
                    if _validate_uniform(circles, radii, W, H):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack([circles, radii])
        except Exception:
            pass
    
    # Fallback to grid
    if best_result is None:
        r = _compute_max_radius(positions, W, H)
        if r > 0:
            circles = positions.copy()
            radii = np.full(N, r)
            if _validate_uniform(circles, radii, W, H):
                return np.column_stack([circles, radii]), np.sum(radii)
    
    return best_result, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second margin for safety
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios - focus on promising ranges
    aspect_ratios = [1.0, 1.2, 0.83, 1.5, 0.67, 1.33, 0.75, 1.1, 0.9, 1.4, 0.7, 1.6, 0.6, 1.25, 0.8]
    
    # Sort by expected promise (near-square first)
    aspect_ratios.sort(key=lambda ar: abs(ar - 1.0))
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Method 1: Hexagonal lattice with direct position optimization
        result, score = _pack_hexagonal(w, h, deadline)
        if result is not None and score > best_score:
            best_score = score
            best_result = result
        
        # Method 2: Direct position optimization from grid
        if time.time() < deadline:
            result2, score2 = _pack_with_position_optimization(w, h, deadline)
            if result2 is not None and score2 > best_score:
                best_score = score2
                best_result = result2
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid in unit square (perimeter = 4)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
