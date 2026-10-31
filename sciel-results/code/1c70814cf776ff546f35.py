"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Determine Rectangle Dimensions
    # Perimeter = 4 => w + h = 2
    # For 21 circles, a 5x4 or 4x5 grid of "slots" is a good heuristic.
    # Hexagonal packing is denser. Let's try a few aspect ratios.
    # w = 2 * r_aspect / (1 + r_aspect), h = 2 / (1 + r_aspect)
    
    best_score = 0.0
    best_result = None
    
    # Candidate aspect ratios (w/h)
    # Square: 1.0
    # 5x4 grid approximation: (5/4) * sqrt(3)/2? No, just try simple ratios.
    # Hexagonal close packing in a rectangle often has aspect ratio ~ sqrt(3) or 2/sqrt(3)
    aspect_ratios = [1.0, 1.2, 1.4, 1.5, 1.732, 2.0, 0.8, 0.6]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # 2. Generate Initial Layout
        # Use a hexagonal lattice as the seed for positions.
        # We need to fit 21 circles.
        # Let's estimate the radius r such that the hex lattice fits.
        # For hex lattice, row spacing is r*sqrt(3), col spacing is 2r.
        # Approximate bounding box for n circles in hex:
        # Let's just place them in a grid and let the optimizer fix it.
        
        # Initialize centers in a 5x5 grid (drop 4 corners or similar) or 4x5
        # Let's try a 5x4 grid of centers, shifted.
        rows = 5
        cols = 4
        # Total 20, need 21. Maybe 6x4=24? Drop 3?
        # Let's use a 5x5 grid and drop the 4 corners? 25-4=21.
        # Or just 21 points in a hex pattern.
        
        # Let's create a hex pattern of 21 points.
        # Rows: 5, 4, 5, 4, 3? Sum = 21.
        # Or 4, 5, 4, 4, 4?
        # Let's use a standard hex seed.
        
        centers = []
        # Generate hex points
        # Row 0: 5 points
        # Row 1: 4 points (offset)
        # Row 2: 5 points
        # Row 3: 4 points (offset)
        # Row 4: 3 points (offset)
        # Total: 5+4+5+4+3 = 21
        
        # Scale factor s will be determined by optimization.
        # Initial guess for r: w / (5 * 2) roughly?
        # Let's define the geometry relative to a unit hex cell size 'a' = 2r.
        
        # Hex coordinates (x, y)
        # Row 0 (y=0): x = 0, 2, 4, 6, 8
        # Row 1 (y=sqrt(3)): x = 1, 3, 5, 7
        # Row 2 (y=2sqrt(3)): x = 0, 2, 4, 6, 8
        # Row 3 (y=3sqrt(3)): x = 1, 3, 5, 7
        # Row 4 (y=4sqrt(3)): x = 1, 3, 5  (Centered? 0,2,4 or 1,3,5?)
        # Let's center the pattern.
        
        hex_points = []
        # Row 0: 5 pts
        for i in range(5):
            hex_points.append((2*i, 0))
        # Row 1: 4 pts
        for i in range(4):
            hex_points.append((2*i+1, np.sqrt(3)))
        # Row 2: 5 pts
        for i in range(5):
            hex_points.append((2*i, 2*np.sqrt(3)))
        # Row 3: 4 pts
        for i in range(4):
            hex_points.append((2*i+1, 3*np.sqrt(3)))
        # Row 4: 3 pts (centered under row 3? Row 3 is 1,3,5,7. Center 4. Row 4 should be 2,4,6? Or 0,2,4?)
        # To keep compactness, let's try 2,4,6.
        for i in range(3):
            hex_points.append((2*i+2, 4*np.sqrt(3)))
            
        hex_points = np.array(hex_points)
        
        # Normalize the pattern to fit in [0,1]x[0,1]
        min_x, min_y = hex_points.min(axis=0)
        max_x, max_y = hex_points.max(axis=0)
        
        # Pattern width in units of '2r' is (max_x - min_x).
        # Pattern height in units of '2r' is (max_y - min_y).
        # We want to scale this pattern to fit the rectangle w x h.
        # Let scale factor k be such that:
        # (max_x - min_x) * k = w
        # (max_y - min_y) * k = h
        # This is an over-constraint. We need to find k that minimizes the max extent.
        # Actually, we can optimize the position and scale of this rigid body,
        # but the circles can also deform slightly (non-rigid) if we allow radius variation?
        # The task allows different radii.
        
        # Strategy:
        # 1. Fix the relative positions of centers based on the hex pattern (rigid body).
        # 2. Optimize: Scale (s), Translation (tx, ty), and individual radii (r_i).
        #    Constraints:
        #    - Circles inside rectangle: r_i <= x_i, r_i <= y_i, r_i <= w-x_i, r_i <= h-y_i
        #    - Disjoint interiors: dist(i,j) >= r_i + r_j
        #    - Maximize sum(r_i)
        
        # Let's parameterize:
        # x_i = tx + s * (hex_x_i - center_x)
        # y_i = ty + s * (hex_y_i - center_y)
        # Wait, if we fix the relative geometry, we can't change the "shape" of the packing.
        # Better: Optimize all x_i, y_i, r_i directly, but use the hex pattern as initial guess.
        
        # Initial Guess for x, y:
        # Scale hex pattern to fit w, h roughly.
        # Aspect ratio of hex pattern:
        pat_w = max_x - min_x
        pat_h = max_y - min_y
        pat_aspect = pat_w / pat_h
        
        # Scale to fit
        if w/h > pat_aspect:
            # Rectangle is wider, fit height
            s = h / pat_h
            tx = (w - pat_w * s) / 2.0
            ty = 0.0
        else:
            # Rectangle is taller, fit width
            s = w / pat_w
            tx = 0.0
            ty = (h - pat_h * s) / 2.0
            
        # Initial centers
        x_init = (hex_points[:, 0] - min_x) * s + tx
        y_init = (hex_points[:, 1] - min_y) * s + ty
        
        # Initial radii
        # Minimum distance between centers in hex pattern is 2 (in units of '2r').
        # So if scale is s, min distance is 2*s.
        # r_i + r_j <= dist(i,j). If all r equal, 2r <= 2s => r <= s.
        # Let's start with r_i = 0.5 * s (leaving room for adjustment).
        r_init = np.full(N, 0.5 * s)
        
        # Variables: [x_0...x_20, y_0...y_20, r_0...r_20]
        # Total 63 variables.
        
        def objective(vars):
            if time.time() > deadline:
                return np.inf
            x = vars[:N]
            y = vars[N:2*N]
            r = vars[2*N:3*N]
            
            # Penalty for being outside rectangle
            pen = 0.0
            # Soft constraints:
            # r_i > 0
            r_neg = np.clip(-r, 0, None)
            pen += np.sum(r_neg**2) * 100
            
            # Inside rectangle
            # x_i >= r_i => x_i - r_i >= 0
            # x_i <= w - r_i => w - x_i - r_i >= 0
            # y_i >= r_i => y_i - r_i >= 0
            # y_i <= h - r_i => h - y_i - r_i >= 0
            
            viol_x_min = np.clip(r - x, 0, None)
            viol_x_max = np.clip(r - (w - x), 0, None)
            viol_y_min = np.clip(r - y, 0, None)
            viol_y_max = np.clip(r - (h - y), 0, None)
            
            pen += np.sum(viol_x_min**2) * 100
            pen += np.sum(viol_x_max**2) * 100
            pen += np.sum(viol_y_min**2) * 100
            pen += np.sum(viol_y_max**2) * 100
            
            # Disjoint interiors: dist(i,j) >= r_i + r_j
            # dist^2 - (r_i + r_j)^2 >= 0
            # For efficiency, we can check all pairs. N=21, pairs=210.
            # Using vectorized distance matrix is faster.
            
            # Create distance matrix
            # x: (N,), y: (N,)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist_sq = dx**2 + dy**2
            
            r_sum = r[:, None] + r[None, :]
            # We only care about i < j
            # dist_sq >= r_sum^2
            # viol = max(0, r_sum^2 - dist_sq)
            
            # Mask for i < j
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Calculate violation for all pairs
            # To avoid sqrt, use squared distances?
            # Constraint: sqrt(dx^2+dy^2) >= r_i + r_j
            # Square both sides (valid since both positive):
            # dx^2+dy^2 >= (r_i+r_j)^2
            
            r_sum_sq = r_sum**2
            viol_sq = np.clip(r_sum_sq - dist_sq, 0, None)
            
            # Sum of violations for i<j
            pen += np.sum(viol_sq[mask]) * 100
            
            # Objective: Minimize -sum(r) + penalty
            return -np.sum(r) + pen

        # Initial guess
        x0 = np.concatenate([x_init, y_init, r_init])
        
        # Bounds
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        r_max = min(w, h) / 2.0
        bounds = ([0.0]*N + [0.0]*N + [0.0]*N, 
                  [w]*N + [h]*N + [r_max]*N)
        
        try:
            res = least_squares(objective, x0, bounds=bounds, method='trf', max_nfev=5000)
            
            if res.success or res.status > 0:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Validate result
                valid = True
                # Check bounds
                if np.any(x_opt < -1e-5) or np.any(x_opt > w + 1e-5): valid = False
                if np.any(y_opt < -1e-5) or np.any(y_opt > h + 1e-5): valid = False
                if np.any(r_opt < -1e-5): valid = False
                
                # Check disjointness
                if valid:
                    dx = x_opt[:, None] - x_opt[None, :]
                    dy = y_opt[:, None] - y_opt[None, :]
                    dist_sq = dx**2 + dy**2
                    r_sum = r_opt[:, None] + r_opt[None, :]
                    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                    # dist >= r_i + r_j
                    # dist_sq >= r_sum^2
                    if np.any(dist_sq[mask] < r_sum[mask]**2 - 1e-5):
                        valid = False
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        # Clip tiny negatives just in case
                        r_final = np.maximum(r_opt, 0.0)
                        best_result = np.column_stack((x_opt, y_opt, r_final))
                        
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to simple grid if optimization failed completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result

# EVOLVE-BLOCK-END
