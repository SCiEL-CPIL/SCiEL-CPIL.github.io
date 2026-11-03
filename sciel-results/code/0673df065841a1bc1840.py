"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Uses direct position optimization with L-BFGS-B and multiple aspect ratios."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate(circles, W, H):
    """Check if circles are valid (non-overlapping, within bounds).
    circles: (n, 3) array of [x, y, r]
    """
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x < r - 1e-7 or x > W - r + 1e-7:
            return False
        if y < r - 1e-7 or y > H - r + 1e-7:
            return False
        if r < -1e-7:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((circles[i,0] - circles[j,0])**2 + (circles[i,1] - circles[j,1])**2)
            if d < circles[i,2] + circles[j,2] - 1e-7:
                return False
    return True


def _total_radius(circles):
    """Sum of all radii."""
    return np.sum(circles[:, 2])


def _make_hex_init(W, H, n=N):
    """Create hexagonal lattice initialization scaled to fit W x H."""
    # Hexagonal lattice: rows of circles with alternating offset
    # For n=21, use a 5x5 grid with hexagonal offset pattern
    # Rows: 5,4,5,4,3 or similar
    
    # Try different row configurations
    best_init = None
    best_r = -1
    
    row_configs = [
        [5, 4, 5, 4, 3],
        [4, 5, 4, 5, 3],
        [5, 4, 5, 3, 4],
        [4, 5, 4, 3, 5],
        [5, 5, 4, 4, 3],
        [5, 4, 4, 5, 3],
    ]
    
    for config in row_configs:
        if sum(config) != n:
            continue
        
        positions = []
        y_pos = 0.0
        for row_idx, count in enumerate(config):
            # Hexagonal offset: alternate rows shifted by half spacing
            offset = 0.5 * (row_idx % 2)
            for col in range(count):
                x = (col + offset) 
                y = y_pos
                positions.append([x, y])
            y_pos += np.sqrt(3) / 2
        
        if len(positions) != n:
            continue
        
        positions = np.array(positions)
        
        # Normalize to fit in [0,1] x [0,1]
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        span_x = max_x - min_x
        span_y = max_y - min_y
        
        if span_x < 1e-10 or span_y < 1e-10:
            continue
        
        # Scale to fit W x H with margin
        scale_x = W / (span_x + 1.0)
        scale_y = H / (span_y + 1.0)
        scale = min(scale_x, scale_y)
        
        scaled = positions * scale
        scaled[:, 0] += (W - scaled[:, 0].max() + scaled[:, 0].min()) / 2
        scaled[:, 1] += (H - scaled[:, 1].max() + scaled[:, 1].min()) / 2
        
        # Compute max uniform radius
        r = np.inf
        for i in range(n):
            min_dist = min(
                scaled[i, 0],
                W - scaled[i, 0],
                scaled[i, 1],
                H - scaled[i, 1]
            )
            r = min(r, min_dist)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((scaled[i,0] - scaled[j,0])**2 + (scaled[i,1] - scaled[j,1])**2)
                r = min(r, d / 2.0)
        
        if r > best_r:
            best_r = r
            best_init = scaled.copy()
    
    if best_init is None:
        # Fallback: simple grid
        cols = int(np.ceil(np.sqrt(n * W / H)))
        rows = int(np.ceil(n / cols))
        positions = np.zeros((n, 2))
        for i in range(n):
            c = i % cols
            r_idx = i // cols
            positions[i, 0] = (c + 0.5) * W / cols
            positions[i, 1] = (r_idx + 0.5) * H / rows
        best_init = positions
    
    return best_init


def _objective(x, W, H):
    """Objective function: minimize negative sum of radii.
    x: [x1,y1,...,xn,yn, r1,...,rn]
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Penalty for out-of-bounds
    penalty = 0.0
    for i in range(n):
        if pos[i, 0] < radii[i]:
            penalty += (radii[i] - pos[i, 0])**2
        if pos[i, 0] > W - radii[i]:
            penalty += (pos[i, 0] - (W - radii[i]))**2
        if pos[i, 1] < radii[i]:
            penalty += (radii[i] - pos[i, 1])**2
        if pos[i, 1] > H - radii[i]:
            penalty += (pos[i, 1] - (H - radii[i]))**2
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((pos[i,0] - pos[j,0])**2 + (pos[i,1] - pos[j,1])**2)
            min_dist = radii[i] + radii[j]
            if d < min_dist:
                penalty += (min_dist - d)**2
    
    if penalty > 1e-4:
        return -np.sum(radii) + 100.0 * penalty
    
    return -np.sum(radii)


def _gradient(x, W, H):
    """Gradient of objective function."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    grad = np.zeros_like(x)
    
    # Gradient of negative sum of radii
    grad[2*n:] = -1.0
    
    # Gradient of boundary penalty
    for i in range(n):
        if pos[i, 0] < radii[i]:
            diff = radii[i] - pos[i, 0]
            grad[2*i] += 200.0 * diff
            grad[2*n + i] -= 200.0 * diff
        if pos[i, 0] > W - radii[i]:
            diff = pos[i, 0] - (W - radii[i])
            grad[2*i] -= 200.0 * diff
            grad[2*n + i] += 200.0 * diff
        if pos[i, 1] < radii[i]:
            diff = radii[i] - pos[i, 1]
            grad[2*i + 1] += 200.0 * diff
            grad[2*n + i] -= 200.0 * diff
        if pos[i, 1] > H - radii[i]:
            diff = pos[i, 1] - (H - radii[i])
            grad[2*i + 1] -= 200.0 * diff
            grad[2*n + i] += 200.0 * diff
    
    # Gradient of overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if d < min_dist and d > 1e-10:
                diff = min_dist - d
                # d/dx_i of (min_dist - d)^2 = 2*(min_dist-d) * (-dx/d)
                grad[2*i] += 200.0 * 2 * diff * (-dx / d)
                grad[2*i + 1] += 200.0 * 2 * diff * (-dy / d)
                grad[2*j] += 200.0 * 2 * diff * (dx / d)
                grad[2*j + 1] += 200.0 * 2 * diff * (dy / d)
                # Radii gradients
                grad[2*n + i] += 200.0 * 2 * diff
                grad[2*n + j] += 200.0 * 2 * diff
    
    return grad


def _optimize_for_aspect(W, H, deadline, init_positions=None):
    """Optimize circle positions and radii for given rectangle dimensions."""
    if init_positions is None:
        init_positions = _make_hex_init(W, H)
    
    n = N
    # Initial radii: uniform based on max possible
    r_init = np.inf
    for i in range(n):
        min_dist = min(
            init_positions[i, 0],
            W - init_positions[i, 0],
            init_positions[i, 1],
            H - init_positions[i, 1]
        )
        r_init = min(r_init, min_dist)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((init_positions[i,0] - init_positions[j,0])**2 + 
                       (init_positions[i,1] - init_positions[j,1])**2)
            r_init = min(r_init, d / 2.0)
    
    r_init = max(0.01, r_init)
    radii_init = np.full(n, r_init)
    
    x0 = np.concatenate([init_positions.flatten(), radii_init])
    
    # Bounds: positions in [0, W] x [0, H], radii positive
    bounds = []
    for i in range(n):
        bounds.append((0, W))  # x
        bounds.append((0, H))  # y
    for i in range(n):
        bounds.append((0.001, min(W, H)/2))  # radius
    
    start_time = time.time()
    
    try:
        result = minimize(
            _objective,
            x0,
            jac=_gradient,
            args=(W, H),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if time.time() > deadline:
            return None, 0.0
        
        x_opt = result.x
        pos = x_opt[:2*n].reshape(n, 2)
        radii = x_opt[2*n:]
        
        # Clamp to valid range
        pos[:, 0] = np.clip(pos[:, 0], 0, W)
        pos[:, 1] = np.clip(pos[:, 1], 0, H)
        radii = np.clip(radii, 0.001, min(W, H)/2)
        
        circles = np.column_stack([pos, radii])
        
        if _validate(circles, W, H):
            score = _total_radius(circles)
            return circles, score
        else:
            # Try to fix by reducing radii
            for i in range(n):
                circles[i, 2] *= 0.99
            if _validate(circles, W, H):
                score = _total_radius(circles)
                return circles, score
            return None, 0.0
            
    except Exception as e:
        return None, 0.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second margin
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.833, 1.5, 0.667, 2.0, 0.5, 1.333, 0.75, 1.1, 0.9, 1.4, 0.7, 1.8, 0.55]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Method: Direct position + radius optimization with L-BFGS-B
        result, score = _optimize_for_aspect(w, h, deadline)
        if result is not None and score > best_score:
            best_score = score
            best_result = result
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid in unit square (perimeter = 4)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
