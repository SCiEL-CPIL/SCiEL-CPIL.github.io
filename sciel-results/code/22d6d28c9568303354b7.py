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
    """
    positions = []
    # Hexagonal lattice: rows offset by half spacing
    row_spacing = spacing * np.sqrt(3) / 2
    row = 0
    while len(positions) < n:
        cols_in_row = int(np.ceil(n / max(1, (n - len(positions)))))
        # Alternate row lengths for hexagonal packing
        if row % 2 == 0:
            cols_needed = int(np.ceil((n - len(positions)) / 1.0))
        else:
            cols_needed = int(np.ceil((n - len(positions)) / 1.0))
        
        # Calculate how many columns fit in this row
        remaining = n - len(positions)
        cols = min(cols_needed, int(np.ceil(remaining / 1.0)))
        
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
    
    # Constraint 1: each circle must fit in rectangle
    r_boundary = np.inf
    for i in range(n):
        min_dist = min(
            positions[i, 0],
            W - positions[i, 0],
            positions[i, 1],
            H - positions[i, 1]
        )
        r_boundary = min(r_boundary, min_dist)
    
    # Constraint 2: pairwise distances
    r_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                       (positions[i, 1] - positions[j, 1])**2)
            r_pair = min(r_pair, d / 2.0)
    
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
    """Pack circles in W x H rectangle using hexagonal lattice with affine optimization."""
    start_time = time.time()
    
    # Generate base hexagonal positions
    base_positions = _hexagonal_positions(n_circles, spacing=1.0)
    
    # Center the positions
    base_positions[:, 0] -= np.mean(base_positions[:, 0])
    base_positions[:, 1] -= np.mean(base_positions[:, 1])
    
    # Compute bounding box of base positions
    min_x, min_y = np.min(base_positions, axis=0)
    max_x, max_y = np.max(base_positions, axis=0)
    base_w = max_x - min_x
    base_h = max_y - min_y
    
    if base_w <= 0 or base_h <= 0:
        return None, 0.0
    
    # Objective: minimize negative radius
    def objective(params):
        if time.time() > deadline:
            return 1e10
        sx, sy, shear = params
        # Ensure positive scales
        sx = max(0.01, sx)
        sy = max(0.01, sy)
        
        transformed = _apply_affine(base_positions, sx, sy, shear)
        
        # Translate so min corner is at (r, r) where r is the radius
        # First, compute bounding box of transformed positions
        tmin_x, tmin_y = np.min(transformed, axis=0)
        tmax_x, tmax_y = np.max(transformed, axis=0)
        t_w = tmax_x - tmin_x
        t_h = tmax_y - tmin_y
        
        # The rectangle is W x H. We need to scale the transformed positions
        # to fit in the rectangle with some margin for radius
        # Scale factor to fit in rectangle
        scale_x = W / (t_w + 1e-10)
        scale_y = H / (t_h + 1e-10)
        scale = min(scale_x, scale_y)
        
        # Apply scaling and translation
        scaled = transformed * scale
        scaled[:, 0] += (W - scaled[:, 0].max() + scaled[:, 0].min()) / 2
        scaled[:, 1] += (H - scaled[:, 1].max() + scaled[:, 1].min()) / 2
        
        # Compute max radius
        r = _compute_max_radius(scaled, W, H)
        
        return -r
    
    # Initial guess: scale to fit rectangle
    init_sx = W / base_w
    init_sy = H / base_h
    init_shear = 0.0
    
    x0 = np.array([init_sx, init_sy, init_shear])
    
    # Bounds for parameters
    bounds = [
        (0.1, 10.0),  # sx
        (0.1, 10.0),  # sy
        (-2.0, 2.0)   # shear
    ]
    
    best_result = None
    best_score = -np.inf
    
    # Try Nelder-Mead
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if time.time() < deadline:
            sx, sy, shear = result.x
            sx = max(0.01, sx)
            sy = max(0.01, sy)
            
            transformed = _apply_affine(base_positions, sx, sy, shear)
            tmin_x, tmin_y = np.min(transformed, axis=0)
            tmax_x, tmax_y = np.max(transformed, axis=0)
            t_w = tmax_x - tmin_x
            t_h = tmax_y - tmin_y
            
            scale_x = W / (t_w + 1e-10)
            scale_y = H / (t_h + 1e-10)
            scale = min(scale_x, scale_y)
            
            scaled = transformed * scale
            scaled[:, 0] += (W - scaled[:, 0].max() + scaled[:, 0].min()) / 2
            scaled[:, 1] += (H - scaled[:, 1].max() + scaled[:, 1].min()) / 2
            
            r = _compute_max_radius(scaled, W, H)
            
            if r > 0:
                # Create circles with uniform radius
                circles = scaled.copy()
                radii = np.full(n_circles, r)
                
                # Validate
                if _validate_uniform(circles, radii, W, H):
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([circles, radii])
    except Exception:
        pass
    
    # Also try grid-based initialization as fallback
    if best_result is None or time.time() < deadline:
        cols = int(np.ceil(np.sqrt(n_circles * W / H)))
        rows = int(np.ceil(n_circles / cols))
        
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
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([circles, radii])
    
    return best_result, best_score


def _pack_with_position_optimization(W, H, deadline):
    """Pack circles by optimizing positions directly with Nelder-Mead."""
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
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        pos = x.reshape(N, 2)
        # Check bounds
        if np.any(pos[:, 0] < 0) or np.any(pos[:, 0] > W) or \
           np.any(pos[:, 1] < 0) or np.any(pos[:, 1] > H):
            return 1e10
        r = _compute_max_radius(pos, W, H)
        if r <= 0:
            return 1e10
        return -r
    
    x0 = positions.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
        )
        
        if time.time() < deadline:
            pos = result.x.reshape(N, 2)
            # Clamp to bounds
            pos[:, 0] = np.clip(pos[:, 0], 0, W)
            pos[:, 1] = np.clip(pos[:, 1], 0, H)
            
            r = _compute_max_radius(pos, W, H)
            if r > 0:
                circles = pos.copy()
                radii = np.full(N, r)
                if _validate_uniform(circles, radii, W, H):
                    return np.column_stack([circles, radii]), np.sum(radii)
    except Exception:
        pass
    
    # Fallback to grid
    r = _compute_max_radius(positions, W, H)
    if r > 0:
        circles = positions.copy()
        radii = np.full(N, r)
        if _validate_uniform(circles, radii, W, H):
            return np.column_stack([circles, radii]), np.sum(radii)
    
    return None, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20 second margin for safety
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.83, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9, 1.4, 0.7]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Method 1: Hexagonal lattice with affine optimization
        result, score = _pack_hexagonal(w, h, deadline)
        if result is not None and score > best_score:
            best_score = score
            best_result = result
        
        # Method 2: Direct position optimization
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
