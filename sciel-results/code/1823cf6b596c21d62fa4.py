"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21

# EVOLVE-BLOCK-START

def _hexagonal_initialization(n, width, height):
    """
    Generate a hexagonal packing initialization within a rectangle of given width/height.
    Returns arrays x, y.
    """
    xs = []
    ys = []
    # Hexagonal packing parameters
    # Row height = sqrt(3)/2 * r. Let's estimate r based on area.
    # Area = n * pi * r^2 <= w*h => r <= sqrt(w*h/(n*pi))
    # Hex packing density is ~0.9069. 
    # Effective area per circle in hex packing is ~ (2/sqrt(3)) * r^2
    # n * (2/sqrt(3)) * r^2 <= w*h => r <= sqrt(w*h * sqrt(3) / (2*n))
    
    # We want to fit n circles.
    # Let's try to fit rows of circles.
    # For a given r, width needed for k circles in a row is (2k-1)r.
    # Height needed for m rows is (sqrt(3)/2)*(m-1)r + 2r.
    
    # We'll search for a good r that fits in w, h.
    # Since we just need initialization, we can be a bit conservative.
    
    # Estimate max r
    r_est = np.sqrt(w * h * np.sqrt(3) / (2 * n))
    
    # Try to place circles
    # We will generate a grid of potential centers and pick the first n that fit.
    # Or simpler: generate a hex grid covering the rectangle and take first n.
    
    # Let's use a slightly smaller r to ensure they fit initially, optimizer will expand.
    r_init = r_est * 0.8
    
    row_height = np.sqrt(3) / 2 * r_init
    col_width = 2 * r_init
    
    y = r_init
    row_idx = 0
    while len(xs) < n:
        x = r_init
        if row_idx % 2 == 1:
            x = r_init + r_init # offset by r
        while x + r_init <= w and len(xs) < n:
            if y + r_init <= h:
                xs.append(x)
                ys.append(y)
            x += col_width
        y += row_height
        row_idx += 1
        
    if len(xs) < n:
        # Fallback to grid if hex fails (shouldn't happen for reasonable w,h)
        cols = int(np.ceil(np.sqrt(n * w / h)))
        rows = int(np.ceil(n / cols))
        xs = []
        ys = []
        for i in range(rows):
            for j in range(cols):
                if len(xs) >= n: break
                xs.append((j + 0.5) * w / cols)
                ys.append((i + 0.5) * h / rows)
                
    return np.array(xs), np.array(ys)

def _pack_circles(n, max_time=100.0):
    """
    Pack n circles in a rectangle of perimeter <= 4 to maximize sum of radii.
    Returns array of shape (n, 3) with columns [x, y, r].
    """
    start_time = time.time()
    
    # We optimize x, y, r for each circle. 
    # The rectangle dimensions w, h are determined by the bounding box of the circles.
    # Constraint: 2 * (max(x+r) - min(x-r) + max(y+r) - min(y-r)) <= 4
    # This is non-smooth due to min/max. 
    # To make it smooth for SLSQP, we can introduce auxiliary variables or just use a penalty 
    # for the perimeter constraint in the objective, but keep overlap as hard constraints.
    # Alternatively, fix w and h and optimize circles inside, then try multiple w,h ratios.
    
    # Let's fix the rectangle aspect ratio and size to simplify constraints.
    # Perimeter = 4 => w + h = 2.
    # Let's try a few aspect ratios: 1:1 (w=1, h=1), 1.5:1 (w=1.2, h=0.8), 2:1 (w=1.33, h=0.66)
    
    best_result = None
    best_value = -np.inf
    
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5]
    
    for ar in aspect_ratios:
        if time.time() - start_time > max_time - 5:
            break
            
        # w * ar + h = 2, w = ar * h => h(ar + 1) = 2 => h = 2 / (ar + 1)
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Initialization
        x0, y0 = _hexagonal_initialization(n, w, h)
        
        # Estimate initial radius
        # Area available = w*h. Area per circle ~ pi*r^2.
        r0 = np.full(n, np.sqrt(w * h / (n * np.pi)) * 0.9)
        
        # Variables: x (n), y (n), r (n)
        x_init = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0, w))  # x
        for i in range(n):
            bounds.append((0, h))  # y
        for i in range(n):
            bounds.append((0.01, min(w, h) / 2))  # r
            
        def _objective(vars_):
            if time.time() - start_time > max_time:
                return 1e10
            x = vars_[:n]
            y = vars_[n:2*n]
            r = vars_[2*n:3*n]
            
            # Check if circles are within [0, w] x [0, h]
            # This is handled by bounds, but let's add a small penalty if they touch edges too hard?
            # No, bounds handle it.
            
            # Perimeter constraint is on the bounding box of the circles.
            # But we fixed the container w, h.
            # The problem says: "bounding rectangle has perimeter at most 4".
            # If we force circles to be inside [0, w]x[0, h] where 2(w+h)=4, 
            # the bounding box of the circles will be <= w x h, so perimeter <= 4.
            # So we just need to maximize sum(r) subject to circles being inside [0,w]x[0,h] and non-overlapping.
            
            return -np.sum(r)
            
        def _non_overlap_constraints(vars_):
            if time.time() - start_time > max_time:
                return np.ones(n * (n-1) // 2)
            x = vars_[:n]
            y = vars_[n:2*n]
            r = vars_[2*n:3*n]
            
            cons = []
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r[i] + r[j]
                    # dist >= min_dist => dist_sq >= min_dist^2
                    # dist_sq - min_dist^2 >= 0
                    cons.append(dist_sq - min_dist*min_dist)
            return np.array(cons)
            
        # SLSQP requires constraints in the form g(vars) >= 0
        # We can pass a list of dictionaries or a single function returning array.
        # SLSQP in scipy.optimize.minimize accepts 'ineq' as a function returning array of >= 0.
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': _non_overlap_constraints},
                options={'maxiter': 100, 'ftol': 1e-6}
            )
        except Exception:
            continue
            
        if result.success:
            x_opt = result.x[:n]
            y_opt = result.x[n:2*n]
            r_opt = result.x[2*n:3*n]
            
            # Verify validity
            # 1. Inside box
            if np.any(x_opt < -1e-6) or np.any(x_opt > w + 1e-6) or np.any(y_opt < -1e-6) or np.any(y_opt > h + 1e-6):
                continue
            # 2. Non-overlap
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    if dist < r_opt[i] + r_opt[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                value = np.sum(r_opt)
                if value > best_value:
                    best_value = value
                    # Clip to ensure strict validity for evaluator
                    x_res = np.clip(x_opt, 0, w)
                    y_res = np.clip(y_opt, 0, h)
                    best_result = np.column_stack([x_res, y_res, r_opt])
    
    if best_result is None:
        # Fallback: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        # Ensure perimeter constraint for fallback
        # Grid is in [0,1]x[0,1], perimeter 4.
        # Radii are ~ 0.5/side.
        # Check overlap: dist between centers is 1/side. 2*r = 1/side - 2e-3. OK.
        return np.array(cells[:N])
        
    return best_result

def construct():
    """
    Returns array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4.
    """
    result = _pack_circles(N, max_time=110.0)
    return result

# EVOLVE-BLOCK-END
