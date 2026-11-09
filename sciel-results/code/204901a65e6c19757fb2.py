# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _penalty(centers, sides, tol=1e-7):
    """Compute violation penalty: boundary + overlap violations."""
    n = len(sides)
    cx = centers[:, 0]
    cy = centers[:, 1]
    s = sides
    
    # Boundary violations
    lo_x = cx - s / 2.0
    hi_x = cx + s / 2.0
    lo_y = cy - s / 2.0
    hi_y = cy + s / 2.0
    bv = np.maximum(0, np.maximum(np.maximum(-lo_x, hi_x - 1.0),
                                   np.maximum(-lo_y, hi_y - 1.0)))
    boundary_pen = np.sum(bv)
    
    # Overlap violations
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (s[i] + s[j]) / 2.0 - tol
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            # If both dx < t and dy < t, there's overlap
            # Penalty: how much they overlap
            ox = max(0.0, t - dx)
            oy = max(0.0, t - dy)
            if ox > 0 and oy > 0:
                overlap_pen += ox * oy
    
    return boundary_pen + overlap_pen

def _objective(params, n=10, tol=1e-7):
    """Objective: negative sum of sides + large penalty for violations."""
    cx = params[0:n]
    cy = params[n:2*n]
    s = params[2*n:3*n]
    
    # Ensure sides are positive
    s = np.maximum(s, 1e-6)
    
    centers = np.stack([cx, cy], axis=1)
    pen = _penalty(centers, s, tol)
    
    # If penalty is large, return large value
    if pen > 1e-4:
        return 1e6 + pen
    
    return -np.sum(s) + 1e4 * pen

def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in the unit square.
    Uses a structured initial packing + scipy optimization.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    # Initial packing: 4 large squares in corners + 6 smaller in middle
    # Corner squares: side ~0.35, placed at corners
    # Middle: 6 squares of side ~0.18
    
    # Try multiple initial configurations
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Configuration 1: 4 corners + 6 in cross pattern
    a = 0.35  # corner square side
    b = 0.18  # middle square side
    
    # Corner centers
    c1 = np.array([a/2, a/2])
    c2 = np.array([1-a/2, a/2])
    c3 = np.array([a/2, 1-a/2])
    c4 = np.array([1-a/2, 1-a/2])
    
    # Middle 6 squares: place in a 3x2 grid in the center region
    # Center region roughly [a, 1-a] x [a, 1-a]
    mid_x = a + (1 - 2*a) / 2  # center x of middle region
    mid_y = a + (1 - 2*a) / 2
    
    # 3 columns, 2 rows of small squares
    small_centers = []
    for row in range(2):
        for col in range(3):
            sx = a + b/2 + col * (1 - 2*a - b) / 2.0
            sy = a + b/2 + row * (1 - 2*a - b) / 2.0
            small_centers.append([sx, sy])
    
    centers_init = np.array([c1, c2, c3, c4] + small_centers)
    sides_init = np.array([a, a, a, a] + [b]*6)
    
    # Verify initial is valid
    pen0 = _penalty(centers_init, sides_init)
    if pen0 < 1e-6:
        score0 = np.sum(sides_init)
        best_score = score0
        best_centers = centers_init.copy()
        best_sides = sides_init.copy()
    
    # Optimize with scipy
    params0 = np.concatenate([centers_init.ravel(), sides_init])
    
    # Try different starting points
    configs = []
    
    # Config 1: as above
    configs.append((centers_init, sides_init))
    
    # Config 2: 5 large + 5 small
    a2 = 0.30
    b2 = 0.20
    # 5 large in a pattern, 5 small
    # Large: 4 corners + 1 center
    c_large = np.array([
        [a2/2, a2/2],
        [1-a2/2, a2/2],
        [a2/2, 1-a2/2],
        [1-a2/2, 1-a2/2],
        [0.5, 0.5]
    ])
    # 5 small in gaps
    # Place small squares in the 4 edge midpoints and one more
    gap_size = (1 - 2*a2) / 2  # half gap
    # Actually let's just place them carefully
    small2 = np.array([
        [0.5, a2/2 + b2/2],      # bottom center
        [0.5, 1-a2/2-b2/2],      # top center
        [a2/2 + b2/2, 0.5],      # left center
        [1-a2/2-b2/2, 0.5],      # right center
        [0.5, 0.5]               # but this overlaps with large center... skip
    ])
    # This config is tricky, let's use a simpler one
    # Config 3: uniform grid 3x4 = 12, use 10
    k1, k2 = 3, 4
    side3 = 1.0 / max(k1, k2)
    idx = np.arange(n)
    cols = idx % k1
    rows = idx // k1
    cx3 = (cols + 0.5) * (1.0/k1)
    cy3 = (rows + 0.5) * (1.0/k2)
    centers3 = np.stack([cx3, cy3], axis=1)
    sides3 = np.full(n, side3)
    pen3 = _penalty(centers3, sides3)
    if pen3 < 1e-6:
        configs.append((centers3, sides3))
    
    # Config 4: 2x5 grid
    k1, k2 = 2, 5
    side4 = 1.0 / k2
    idx = np.arange(n)
    cols = idx % k1
    rows = idx // k1
    cx4 = (cols + 0.5) * (1.0/k1)
    cy4 = (rows + 0.5) * (1.0/k2)
    centers4 = np.stack([cx4, cy4], axis=1)
    sides4 = np.full(n, side4)
    pen4 = _penalty(centers4, sides4)
    if pen4 < 1e-6:
        configs.append((centers4, sides4))
    
    for centers_c, sides_c in configs:
        if time.time() > deadline - 60:
            break
        
        params_c = np.concatenate([centers_c.ravel(), sides_c])
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # cx
        for i in range(n):
            bounds.append((0.0, 1.0))  # cy
        for i in range(n):
            bounds.append((1e-6, 0.5))  # side
        
        try:
            result = minimize(
                _objective,
                params_c,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'maxfun': 50000, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < _objective(params_c):
                opt_params = result.x
                cx_opt = opt_params[0:n]
                cy_opt = opt_params[n:2*n]
                s_opt = opt_params[2*n:3*n]
                
                centers_opt = np.stack([cx_opt, cy_opt], axis=1)
                pen_opt = _penalty(centers_opt, s_opt)
                
                if pen_opt < 1e-7:
                    score_opt = np.sum(s_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_centers = centers_opt.copy()
                        best_sides = s_opt.copy()
        except Exception:
            pass
    
    # If no valid solution found, return the grid baseline
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)

# EVOLVE-BLOCK-END
