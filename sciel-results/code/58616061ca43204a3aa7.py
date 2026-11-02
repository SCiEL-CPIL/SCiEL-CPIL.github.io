"""Improved circle packing for perimeter-4 rectangle with 21 circles."""
import numpy as np
import time

N = 21


def _validate(arr):
    """Check if arrangement is valid: all circles inside rectangle, no overlaps."""
    if arr.shape != (N, 3):
        return False
    if np.any(arr[:, 2] <= 0):
        return False
    # Check bounding box perimeter
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    width = max_x - min_x
    height = max_y - min_y
    if width + height > 1.0 + 1e-9:  # perimeter = 2*(w+h) <= 4 => w+h <= 2, but wait...
        # Actually perimeter of bounding rectangle = 2*(width + height) <= 4
        # So width + height <= 2
        pass
    # Wait, let me re-read: "bounding rectangle has perimeter at most 4"
    # Perimeter = 2*(width + height) <= 4 => width + height <= 2
    if width + height > 2.0 + 1e-9:
        return False
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True


def _score(arr):
    """Compute the objective: sum of radii, only if valid."""
    if not _validate(arr):
        return 0.0
    return float(np.sum(arr[:, 2]))


def _grid_init(rows, cols, W, H):
    """Create a grid of circles in a W x H rectangle."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * W / cols
            y = (i + 0.5) * H / rows
            r = min(W / cols, H / rows) / 2 - 1e-4
            circles.append([x, y, r])
    # If we have more than N, take first N
    circles = circles[:N]
    # If fewer, add remaining at bottom
    while len(circles) < N:
        idx = len(circles)
        x = ((idx % cols) + 0.5) * W / cols
        y = H - 0.1
        r = 0.05
        circles.append([x, y, r])
    return np.array(circles)


def _local_optimize(arr, W, H, max_iter=500, lr=0.01):
    """Simple gradient-free local optimization using coordinate descent + perturbations."""
    best = arr.copy()
    best_score = _score(best)
    
    # Use a simple iterative approach: try to increase radii and adjust positions
    for it in range(max_iter):
        improved = False
        # Try increasing each radius slightly
        for i in range(N):
            new_r = best[i, 2] * 1.005
            trial = best.copy()
            trial[i, 2] = new_r
            if _validate(trial):
                best = trial
                improved = True
                best_score = _score(best)
        
        # Try moving circles to allow larger radii
        if not improved:
            # Random perturbation
            trial = best.copy()
            idx = np.random.randint(N)
            trial[idx, 0] += np.random.uniform(-0.05, 0.05)
            trial[idx, 1] += np.random.uniform(-0.05, 0.05)
            if _validate(trial):
                best = trial
                best_score = _score(best)
                improved = True
        
        if not improved and it > 100:
            break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s budget, leave 20s margin
    
    best_arr = None
    best_score = 0.0
    
    # Try different aspect ratios
    # Perimeter constraint: 2*(W+H) <= 4 => W+H <= 2
    # Let's try W+H = 2 with various W/H ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.25]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W + H = 2, W/H = ar => W = 2*ar/(1+ar), H = 2/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Try different grid configurations
        grid_configs = [
            (5, 5),  # 25 cells, take 21
            (4, 6),  # 24 cells, take 21
            (3, 7),  # 21 cells exactly
            (6, 4),
            (7, 3),
            (5, 4),  # 20 cells, add 1
            (4, 5),
        ]
        
        for rows, cols in grid_configs:
            if time.time() > deadline:
                break
            
            if rows * cols < N:
                continue
                
            arr = _grid_init(rows, cols, W, H)
            
            # Center the arrangement
            xs = arr[:, 0]
            ys = arr[:, 1]
            rs = arr[:, 2]
            min_x = np.min(xs - rs)
            max_x = np.max(xs + rs)
            min_y = np.min(ys - rs)
            max_y = np.max(ys + rs)
            
            # Shift to center in [0, W] x [0, H]
            cx = (W - (max_x - min_x)) / 2 - min_x
            cy = (H - (max_y - min_y)) / 2 - min_y
            arr[:, 0] += cx
            arr[:, 1] += cy
            
            score = _score(arr)
            if score > best_score:
                best_score = score
                best_arr = arr.copy()
            
            # Try local optimization
            if time.time() < deadline - 10:
                opt_arr = _local_optimize(arr, W, H, max_iter=200, lr=0.01)
                opt_score = _score(opt_arr)
                if opt_score > best_score:
                    best_score = opt_score
                    best_arr = opt_arr.copy()
    
    # Additional strategy: uniform radius packing
    # For 21 circles in a rectangle with W+H=2
    # Try to pack them in a more compact arrangement
    
    if time.time() < deadline - 5:
        # Try a hexagonal-like packing
        for ar in [1.0, 1.2, 0.8, 1.5]:
            if time.time() > deadline:
                break
            
            W = 2.0 * ar / (1.0 + ar)
            H = 2.0 / (1.0 + ar)
            
            # Hexagonal packing
            r = 0.1
            dx = 2 * r
            dy = r * np.sqrt(3)
            
            circles = []
            row = 0
            while len(circles) < N:
                col = 0
                offset = (row % 2) * r
                while True:
                    x = r + col * dx + offset
                    y = r + row * dy
                    if x + r <= W and y + r <= H:
                        circles.append([x, y, r])
                        if len(circles) >= N:
                            break
                    col += 1
                row += 1
                if row * dy > H:
                    break
            
            if len(circles) >= N:
                arr = np.array(circles[:N])
                score = _score(arr)
                if score > best_score:
                    best_score = score
                    best_arr = arr.copy()
    
    if best_arr is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr
