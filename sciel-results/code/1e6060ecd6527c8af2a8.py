"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """
    Check if the packing is valid.
    Returns True if all circles are within [0, W]x[0, H] and disjoint.
    """
    n = len(cx)
    # Wall constraints
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True

def _hexagonal_init(W, H, n, seed=0):
    """
    Generate an initial configuration using a hexagonal lattice.
    This is structurally better for circle packing than a square grid.
    """
    rng = np.random.default_rng(seed)
    
    # Determine a grid size that roughly fits N circles
    # Hexagonal packing has density ~0.9069 vs 1.0 for squares (if we could pack squares)
    # But circles in a box... let's just pick a reasonable number of columns.
    # For N=21, a 5x5 grid is too big, 4x5 or 5x4 is good.
    # Let's try to fill a rectangle of W x H.
    
    # Estimate radius
    # Area of box = W*H. Area of N circles = N * pi * r^2.
    # Max density ~0.9. So N*pi*r^2 <= 0.9 * W*H -> r <= sqrt(0.9*W*H/(N*pi))
    r_est = np.sqrt(0.8 * W * H / (N * np.pi))
    
    # Create a grid of points
    # Let's use a staggered grid
    cols = 5
    rows = int(np.ceil(n / cols))
    
    # Cell dimensions
    # For hex packing, vertical spacing is sqrt(3)/2 * horizontal spacing
    # Let's define a unit cell width w and height h
    # If we want them to touch, r = w/2 and h = sqrt(3)*r = w*sqrt(3)/2
    
    # We want to scale this to fit W and H.
    # Let's generate a normalized hex grid and scale it.
    
    xs = []
    ys = []
    for row in range(rows):
        for col in range(cols):
            if len(xs) >= n:
                break
            # Hex offset
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            
            xs.append(x)
            ys.append(y)
            
    if len(xs) < n:
        # If we ran out of grid points, add more rows or columns
        # For N=21, 5 cols * 5 rows = 25 points is enough.
        # Let's ensure we have at least N points.
        while len(xs) < n:
            row = len(xs) // cols
            col = len(xs) % cols
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            xs.append(x)
            ys.append(y)
            
    xs = np.array(xs[:n])
    ys = np.array(ys[:n])
    
    # Scale to fit W and H
    # We need to map [0, max_x] to [0, W] and [0, max_y] to [0, H]
    # But we also need to determine the radius.
    # The "natural" radius in this normalized grid is 0.5 (distance to neighbor is 1.0).
    # Scale factor sx = W / (max_x + 1) ? No, we want to fill the box.
    
    # Let's calculate the bounding box of the normalized grid
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    
    # Add some margin so they don't touch the boundary initially?
    # Actually, SLSQP will push them. Let's scale to fill most of the box.
    # If we scale x to [0, W] and y to [0, H], the relative distances change.
    # It's better to scale uniformly to maintain the hexagonal geometry as much as possible,
    # then center it, then let SLSQP adjust.
    
    # However, the aspect ratio of the grid might not match W/H.
    # Let's just scale x and y independently to fit the box, which distorts the hexagons
    # but provides a good spread of points.
    
    cx = (xs - min_x) / (max_x - min_x + 1e-9) * W
    cy = (ys - min_y) / (max_y - min_y + 1e-9) * H
    
    # Center them
    cx = (cx - np.mean(cx)) + W / 2
    cy = (cy - np.mean(cy)) + H / 2
    
    # Clamp to box
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    
    # Estimate radius
    # The minimum distance between any two points in the normalized grid is 1.0.
    # After scaling, the min distance is roughly min(W/(max_x-min_x+1), H/(max_y-min_y+1)).
    # Let's compute actual min dist.
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            min_dist = min(min_dist, d)
            
    r = np.full(n, min_dist * 0.4) # Start small to avoid overlaps
    
    # Add random jitter to break symmetry
    cx += rng.uniform(-0.05, 0.05, n)
    cy += rng.uniform(-0.05, 0.05, n)
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    
    return cx, cy, r

def _force_relax(cx, cy, r, W, H, iterations=50):
    """
    Quick force-directed relaxation to remove hard overlaps.
    """
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for _ in range(iterations):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-6:
                    overlap = min_d - d
                    fx[i] += (dx/d) * overlap
                    fy[i] += (dy/d) * overlap
                    fx[j] -= (dx/d) * overlap
                    fy[j] -= (dy/d) * overlap
        
        # Wall repulsion
        for i in range(n):
            if cx[i] < r[i]: fx[i] += (r[i] - cx[i]) * 10
            if cx[i] > W - r[i]: fx[i] -= (cx[i] - (W - r[i])) * 10
            if cy[i] < r[i]: fy[i] += (r[i] - cy[i]) * 10
            if cy[i] > H - r[i]: fy[i] -= (cy[i] - (H - r[i])) * 10
            
        cx += 0.1 * fx
        cy += 0.1 * fy
        
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
    return cx, cy, r

def _optimize_aspect(W, H, deadline, seed=0):
    """
    Optimize for a specific aspect ratio using hexagonal init and SLSQP.
    """
    if time.time() > deadline:
        return None, -float('inf')
        
    # Generate initial configuration
    cx, cy, r = _hexagonal_init(W, H, N, seed=seed)
    
    # Pre-relax to remove obvious overlaps
    cx, cy, r = _force_relax(cx, cy, r, W, H, iterations=50)
    
    # SLSQP Optimization
    # Variables: x, y, r for each circle
    n = N
    x0 = np.concatenate([cx, cy, r])
    
    def objective(x):
        cx = x[:n]
        cy = x[n:2*n]
        r = x[2*n:3*n]
        
        # We want to maximize sum(r), so minimize -sum(r)
        # Add penalties for violations to guide the optimizer
        pen = 0.0
        
        # Wall penalties
        for i in range(n):
            if cx[i] < r[i]:
                pen += (r[i] - cx[i]) ** 2
            if cx[i] > W - r[i]:
                pen += (cx[i] - (W - r[i])) ** 2
            if cy[i] < r[i]:
                pen += (r[i] - cy[i]) ** 2
            if cy[i] > H - r[i]:
                pen += (cy[i] - (H - r[i])) ** 2
                
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    pen += (min_d - d) ** 2
                    
        return -np.sum(r) + 100.0 * pen

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, W))
    for i in range(n):
        bounds.append((0, H))
    for i in range(n):
        # Radius upper bound: half of the min dimension
        bounds.append((1e-4, min(W, H) / 2))

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10})
        
        cx_opt = res.x[:n]
        cy_opt = res.x[n:2*n]
        r_opt = res.x[2*n:3*n]
        
        # Validate
        if _is_valid(cx_opt, cy_opt, r_opt, W, H):
            return (cx_opt, cy_opt, r_opt), np.sum(r_opt)
            
        # If SLSQP failed validity check, try to fix with force relaxation
        # and check again
        cx_fix, cy_fix, r_fix = _force_relax(cx_opt, cy_opt, r_opt, W, H, iterations=100)
        if _is_valid(cx_fix, cy_fix, r_fix, W, H):
            return (cx_fix, cy_fix, r_fix), np.sum(r_fix)
            
    except Exception:
        pass
        
    return None, -float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_val = -float('inf')
    best_config = None
    
    # Try different aspect ratios
    # Perimeter 4 => W + H = 2 => H = 2 - W
    # W ranges from small to large.
    # Square (1,1) is often good, but elongated shapes can allow larger circles in corners.
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 1.1, 0.9]
    
    # Shuffle to try diverse shapes early? Or sort by likelihood?
    # Let's try square first, then others.
    
    np.random.seed(42)
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1: # Too thin
            continue
            
        # Perform a few restarts for this aspect ratio
        # Number of restarts depends on remaining time
        remaining_time = deadline - time.time()
        num_restarts = max(1, int(remaining_time / 5.0)) # ~5s per restart
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            config, val = _optimize_aspect(W, H, deadline, seed=restart)
            
            if val > best_val:
                best_val = val
                best_config = config
                
            # If we find a very good value, we might want to stop early?
            # Theoretical max for N=21 in unit square is around 3.6-3.8?
            # In perimeter 4 (area 1), sum of radii for unit square packing is ~3.6.
            # For 21 circles, sum of radii is roughly proportional to sqrt(N).
            # Let's just keep searching.
            
    if best_config is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx, cy, r = best_config
    return np.column_stack([cx, cy, r])
# EVOLVE-BLOCK-END
