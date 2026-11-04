import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # Safety margin for 150s limit

def generate_hex_grid(n, w, h, r_init):
    """
    Generate n points in a hexagonal pattern scaled to fit w x h.
    r_init is a nominal radius to determine spacing.
    """
    # Estimate number of rows and cols
    # Area of hex cell ~ 2 * r^2 * sqrt(3) / 2 ? No.
    # Spacing x: sqrt(3)*r, y: 1.5*r
    # Let's just create a large enough grid and take the first n, or center them.
    
    # Approximate number of points per row
    # If r is small, we can fit many.
    # Let's assume a target density.
    # We want to fill the area w*h with n circles of radius r.
    # Area per circle ~ pi * r^2.
    # Total area ~ n * pi * r^2.
    # But hex packing is denser.
    
    # Simple approach: Create a grid of points based on r_init, then scale to fit w,h.
    # Number of points in hex grid to get ~n
    # Rows: j, Cols: i
    # x_i = i * sqrt(3) * r + (j%2) * 0.5 * sqrt(3) * r
    # y_j = j * 1.5 * r
    
    # Let's guess rows and cols
    # n ~ rows * cols
    # Try different aspect ratios for the grid itself?
    # Just make a square-ish grid and scale.
    
    side = int(np.ceil(np.sqrt(n)))
    # Generate side x side grid
    pts = []
    for j in range(side):
        for i in range(side):
            x = i * np.sqrt(3) * r_init + (j % 2) * 0.5 * np.sqrt(3) * r_init
            y = j * 1.5 * r_init
            pts.append([x, y])
            
    pts = np.array(pts[:n])
    
    # Center and scale
    if len(pts) == 0:
        return np.zeros((n, 2))
        
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    
    w_grid = max_x - min_x
    h_grid = max_y - min_y
    
    if w_grid == 0: w_grid = 1
    if h_grid == 0: h_grid = 1
    
    # Scale to fit w, h with margin
    # We want to leave space for radius r_init
    margin = r_init
    scale_w = (w - 2*margin) / w_grid
    scale_h = (h - 2*margin) / h_grid
    scale = min(scale_w, scale_h)
    
    pts_scaled = (pts - [min_x, min_y]) * scale
    # Center in rectangle
    offset_x = (w - w_grid * scale) / 2
    offset_y = (h - h_grid * scale) / 2
    
    pts_final = pts_scaled + [offset_x, offset_y]
    
    return pts_final

def construct():
    start_time = time.time()
    
    best_val = -np.inf
    best_x = None
    
    # Penalty weight
    lam = 100.0
    
    def objective(x):
        # x = [w, x1, y1, r1, ..., xN, yN, rN]
        w = x[0]
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            return 1e6
            
        coords = x[1:]
        # coords shape: (N, 3)
        coords = coords.reshape(N, 3)
        xs = coords[:, 0]
        ys = coords[:, 1]
        rs = coords[:, 2]
        
        # Penalty for negative radii (should be handled by bounds, but just in case)
        if np.any(rs <= 0):
            return 1e6
            
        # Boundary penalties
        # x - r >= 0 => max(0, r - x)^2
        # x + r <= w => max(0, x + r - w)^2
        # y - r >= 0 => max(0, r - y)^2
        # y + r <= h => max(0, y + r - h)^2
        
        pen_boundary = 0.0
        pen_boundary += np.sum(np.maximum(0, rs - xs)**2)
        pen_boundary += np.sum(np.maximum(0, xs + rs - w)**2)
        pen_boundary += np.sum(np.maximum(0, rs - ys)**2)
        pen_boundary += np.sum(np.maximum(0, ys + rs - h)**2)
        
        # Overlap penalties
        # Vectorized distance calculation
        # dx[i, j] = xs[i] - xs[j]
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        
        # Required distance: rs[i] + rs[j]
        req_d = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        # Violation: max(0, req_d - d)
        # We only care about i < j to avoid double counting and self distance 0
        # But since d[i,i]=0 and req_d[i,i]=2r_i, violation is 2r_i.
        # We should mask the diagonal.
        
        # Create mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        viol = np.maximum(0, req_d - d)
        viol = viol[mask]
        
        pen_overlap = np.sum(viol**2)
        
        total_pen = pen_boundary + pen_overlap
        
        # Objective: Maximize sum(rs) => Minimize -sum(rs)
        # Add penalty
        return -np.sum(rs) + lam * total_pen

    def is_valid(x):
        w = x[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return False, 0.0
            
        coords = x[1:].reshape(N, 3)
        xs = coords[:, 0]
        ys = coords[:, 1]
        rs = coords[:, 2]
        
        if np.any(rs <= 1e-6):
            return False, 0.0
            
        # Check boundaries with small tolerance
        tol = 1e-5
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol) or \
           np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False, 0.0
            
        # Check overlaps
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        req_d = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        # d must be >= req_d
        # If d < req_d - tol, invalid
        if np.any(d[mask] < req_d[mask] - tol):
            return False, 0.0
            
        return True, np.sum(rs)

    # Bounds
    # w in [0.1, 1.9]
    # xs, ys in [0, 2]
    # rs in [1e-5, 1]
    
    bounds = [(0.1, 1.9)]
    for _ in range(N):
        bounds.append((0.0, 2.0))
        bounds.append((0.0, 2.0))
        bounds.append((1e-5, 1.0))
        
    # Initial guesses
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.1, 0.9]
    
    for ar in aspect_ratios:
        if time.time() > DEADLINE:
            break
            
        # Determine w and h
        # w + h = 2
        # w / h = ar => w = ar * h => h (ar + 1) = 2 => h = 2 / (ar + 1)
        h_init = 2.0 / (ar + 1.0)
        w_init = ar * h_init
        
        # Initial radius guess
        # Area = 2.0 * 2.0 = 4.0 (if w=2, h=0, impossible. Max area is 1x1=1)
        # Max area of rect with perimeter 4 is 1x1 = 1.
        # 21 circles in area 1.
        # avg area per circle ~ 1/21 ~ 0.047
        # r ~ sqrt(0.047/pi) ~ 0.12
        # But packing efficiency ~ 0.9, so r ~ 0.1
        r_init = 0.05
        
        # Generate grid
        pts = generate_hex_grid(N, w_init, h_init, r_init)
        
        # Initial x vector
        x0 = np.zeros(1 + N * 3)
        x0[0] = w_init
        for i in range(N):
            x0[1 + i*3] = pts[i, 0]
            x0[1 + i*3 + 1] = pts[i, 1]
            x0[1 + i*3 + 2] = r_init
            
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success:
                valid, val = is_valid(res.x)
                if valid and val > best_val:
                    best_val = val
                    best_x = res.x
        except Exception:
            pass

    # If no valid solution found, return a trivial valid one (tiny circles)
    if best_x is None:
        # Square 1x1
        w = 1.0
        h = 1.0
        x0 = np.zeros(1 + N * 3)
        x0[0] = w
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            row = i // side
            col = i % side
            x0[1 + i*3] = (col + 0.5) / side
            x0[1 + i*3 + 1] = (row + 0.5) / side
            x0[1 + i*3 + 2] = 0.001
        best_x = x0
        best_val = 0.001 * N

    # Format output
    # array of shape (n, 3): [x, y, r]
    w = best_x[0]
    h = 2.0 - w
    coords = best_x[1:].reshape(N, 3)
    
    return coords
