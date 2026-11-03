"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 21
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _validate(arr):
    """Check if array of (x, y, r) is valid: disjoint, inside rect with perimeter <= 4."""
    if arr.shape != (N, 3):
        return False
    if np.any(arr[:, 2] <= 0):
        return False
    # Bounding box
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    top = np.max(ys + rs)
    bottom = np.min(ys - rs)
    W = right - left
    H = top - bottom
    if W <= 0 or H <= 0:
        return False
    if 2 * (W + H) > 4.0 + 1e-7:
        return False
    # Check disjointness
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-7:
                return False
    return True

def _score(arr):
    """Compute the raw value: sum of radii, if valid."""
    if not _validate(arr):
        return 0.0
    return float(np.sum(arr[:, 2]))

def construct():
    t0 = time.time()
    
    # Start with a hexagonal-like arrangement
    # We'll use a grid of 5x5 = 25 positions, pick 21 of them
    # Actually, let's use a smarter initial guess.
    # For 21 circles, a 5x5 grid with 4 removed (corners or edges) works.
    # Or a hexagonal packing in a rectangle.
    
    # Let's try a 5-column by 5-row grid, remove 4 circles to get 21
    # Better: use the known optimal or near-optimal configuration for 21 circles.
    # For now, let's use a grid-based initialization.
    
    # Grid initialization: 5x5 = 25, remove 4 from corners to get 21
    side = 5
    all_positions = []
    for i in range(side):
        for j in range(side):
            all_positions.append((i, j))
    
    # Remove 4 corner positions
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    selected = [p for p in all_positions if p not in corners]
    assert len(selected) == 21
    
    # Place them in a unit square first
    xs = np.array([p[0] / (side - 1) for p in selected])
    ys = np.array([p[1] / (side - 1) for p in selected])
    
    # Normalize to fit in a rectangle with perimeter 4
    # Start with W=2, H=0 (degenerate) or some aspect ratio
    # Let's try W=1.5, H=0.5 -> perimeter = 2*(1.5+0.5) = 4
    W_init = 1.5
    H_init = 0.5
    
    x0 = np.zeros(2 * N + 2)
    x0[:N] = xs * W_init
    x0[N:2*N] = ys * H_init
    x0[2*N] = W_init
    x0[2*N + 1] = H_init
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    # Variables: x[0:N], y[N:2N], r[2N:3N], W[3N], H[3N+1]
    # Actually let's use 3N+2 variables: x(N), y(N), r(N), W, H
    
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N + 1]
        # Penalty for invalid constraints
        # Perimeter constraint: 2*(W+H) <= 4
        if 2 * (W + H) > 4.0:
            return 1e10
        # Circles must be inside rectangle
        for i in range(N):
            if x[i] - r[i] < 0 or x[i] + r[i] > W or y[i] - r[i] < 0 or y[i] + r[i] > H:
                return 1e10
        # Disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j]:
                    return 1e10
        return -np.sum(r)
    
    # Use SLSQP with constraints
    # This might be slow for 21 circles. Let's try a different approach.
    # Use a penalty-based method with many restarts.
    
    best_arr = None
    best_score = 0.0
    
    # Initial grid solution
    # For a 5x5 grid with corners removed, in a 1.5 x 0.5 rectangle
    # The spacing is 1.5/4 = 0.375 horizontally, 0.5/4 = 0.125 vertically
    # Max radius is limited by vertical spacing: 0.125/2 = 0.0625
    # But we can do better with a different aspect ratio.
    
    # Let's try multiple aspect ratios and optimize
    aspect_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    for ar in aspect_ratios:
        # W/H = ar, 2*(W+H) = 4 -> W = 2*ar/(ar+1), H = 2/(ar+1)
        W = 2 * ar / (ar + 1)
        H = 2 / (ar + 1)
        
        # Place 21 circles in a grid pattern
        # Try 5 columns x 5 rows, remove 4 corners
        cols = 5
        rows = 5
        positions = []
        for i in range(cols):
            for j in range(rows):
                if (i, j) not in corners:
                    positions.append((i, j))
        
        x = np.array([p[0] / (cols - 1) * W for p in positions])
        y = np.array([p[1] / (rows - 1) * H for p in positions])
        
        # Compute max possible radius for this configuration
        # Limited by nearest neighbor distances
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_dist = min(min_dist, d)
        
        # Also limited by distance to walls
        for i in range(N):
            min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
        
        r = np.full(N, min_dist / 2)
        
        arr = np.column_stack([x, y, r])
        s = _score(arr)
        if s > best_score:
            best_score = s
            best_arr = arr.copy()
    
    # Now try to improve with local optimization
    # Use a simple gradient-free method: try small perturbations
    
    if time.time() - t0 > TIME_LIMIT - 5:
        return best_arr if best_arr is not None else np.zeros((N, 3))
    
    # Try to improve by adjusting positions
    # For each aspect ratio, try to optimize
    for ar in aspect_ratios:
        if time.time() - t0 > TIME_LIMIT - 10:
            break
        
        W = 2 * ar / (ar + 1)
        H = 2 / (ar + 1)
        
        cols = 5
        rows = 5
        positions = []
        for i in range(cols):
            for j in range(rows):
                if (i, j) not in corners:
                    positions.append((i, j))
        
        x = np.array([p[0] / (cols - 1) * W for p in positions])
        y = np.array([p[1] / (rows - 1) * H for p in positions])
        
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
        
        r = np.full(N, min_dist / 2)
        arr = np.column_stack([x, y, r])
        s = _score(arr)
        
        if s <= best_score:
            continue
        
        # Try to improve by moving circles
        # Simple hill climbing
        step = 0.01
        for iteration in range(100):
            if time.time() - t0 > TIME_LIMIT - 10:
                break
            
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_x = x[i] + dx
                        new_y = y[i] + dy
                        # Check bounds
                        if new_x - r[i] < 0 or new_x + r[i] > W or new_y - r[i] < 0 or new_y + r[i] > H:
                            continue
                        # Check disjointness
                        valid = True
                        for j in range(N):
                            if j == i:
                                continue
                            ddx = new_x - x[j]
                            ddy = new_y - y[j]
                            d = np.sqrt(ddx * ddx + ddy * ddy)
                            if d < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if valid:
                            # Try to increase radius
                            # Compute new max radius
                            new_min_dist = np.inf
                            for j in range(N):
                                if j == i:
                                    continue
                                ddx = new_x - x[j]
                                ddy = new_y - y[j]
                                d = np.sqrt(ddx * ddx + ddy * ddy)
                                new_min_dist = min(new_min_dist, d)
                            new_min_dist = min(new_min_dist, new_x, W - new_x, new_y, H - new_y)
                            new_r = new_min_dist / 2
                            
                            # Check if all circles still fit
                            # Recompute all radii
                            all_valid = True
                            for k in range(N):
                                rx = x[k] if k != i else new_x
                                ry = y[k] if k != i else new_y
                                if rx - r[k] < 0 or rx + r[k] > W or ry - r[k] < 0 or ry + r[k] > H:
                                    all_valid = False
                                    break
                            if all_valid:
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        # Recompute optimal radii
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_dist = min(min_dist, d)
        for i in range(N):
            min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
        
        r = np.full(N, min_dist / 2)
        arr = np.column_stack([x, y, r])
        s = _score(arr)
        if s > best_score:
            best_score = s
            best_arr = arr.copy()
    
    if best_arr is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr

# EVOLVE-BLOCK-END
