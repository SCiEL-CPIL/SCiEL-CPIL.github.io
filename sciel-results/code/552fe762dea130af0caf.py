import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 21

def _hexagonal_init(w, h, n):
    """
    Generate a hexagonal-like packing of n circles in a w x h rectangle.
    Returns x, y, r arrays.
    """
    # Estimate radius based on area
    # Area per circle ~ 1.15 * pi * r^2 for hex packing efficiency
    # Total area w*h ~ n * 1.15 * pi * r^2
    r_est = np.sqrt((w * h) / (n * 1.15 * np.pi))
    
    # We want to place n circles.
    # Let's try a grid of columns.
    # In hex packing, row spacing is r*sqrt(3), col spacing is 2r.
    # Let's just place them in a tight grid and let optimizer fix it.
    
    # Simple grid init is often enough if optimizer is good, but hex is better.
    # Let's create a list of centers.
    centers = []
    r = r_est * 0.9 # Start slightly smaller to allow expansion
    
    # Try to fit as many as possible in a hex lattice
    # Width of one column is 2r. Height of one row is r*sqrt(3).
    # Let's determine how many columns and rows we can fit.
    
    # Iterate over possible number of columns
    best_centers = None
    best_score = -1
    
    for cols in range(1, int(w / (2*r)) + 2):
        if cols == 0: continue
        # Calculate height needed for 'n' circles in 'cols' columns
        rows = np.ceil(n / cols)
        # Hex height: first row at r, next at r + r*sqrt(3), etc.
        # Actually, y-coords: r, r + r*sqrt(3), ...
        # Total height needed: r + (rows-1)*r*sqrt(3) + r = 2r + (rows-1)*r*sqrt(3)
        # If this height > h, this cols is invalid.
        h_needed = 2*r + (rows-1)*r*np.sqrt(3)
        if h_needed > h:
            continue
            
        # Generate centers
        c_list = []
        for i in range(n):
            col = i % cols
            row = i // cols
            # x position
            x = r + col * 2 * r
            # y position
            # Alternate rows are shifted by r
            if row % 2 == 0:
                y = r + row * r * np.sqrt(3)
            else:
                y = r + row * r * np.sqrt(3)
                # Shift x by r for odd rows? 
                # Standard hex: (0,0), (2r, 0), (r, r*sqrt(3)), (3r, r*sqrt(3))
                # So if row is odd, x += r
                x += r
            
            # Clip to bounds to ensure validity for init
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
            
            c_list.append((x, y))
        
        if len(c_list) == n:
            # Check for overlaps in this init (should be minimal if r_est is good)
            # Just use the first valid one
            xs = np.array([c[0] for c in c_list])
            ys = np.array([c[1] for c in c_list])
            rs = np.full(n, r)
            
            # Score: sum of radii (all same here)
            # We prefer tighter packing, so smaller h_needed is better? 
            # Or just take the first valid one.
            return xs, ys, rs
            
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(n)))
    xs = np.array([((i % side) + 0.5) * (w / side) for i in range(n)])
    ys = np.array([((i // side) + 0.5) * (h / side) for i in range(n)])
    rs = np.full(n, np.min(w, h) / (2 * side) * 0.9)
    return xs, ys, rs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s (evaluator kills at 150s, but contract says 120s? "Keep the function names... Each evaluation must finish within 120 s." -> Use 110s to be safe)
    
    # Variables: [w, h, x_1...x_N, y_1...y_N, r_1...r_N]
    # Total vars: 2 + 3*N = 65
    
    def unflatten(v):
        w = v[0]
        h = v[1]
        x = v[2:2+N]
        y = v[2+N:2+2*N]
        r = v[2+2*N:2+3*N]
        return w, h, x, y, r

    def objective(v):
        w, h, x, y, r = unflatten(v)
        # Minimize -sum(r)
        return -np.sum(r)

    def constraints(v):
        w, h, x, y, r = unflatten(v)
        cons = []
        
        # Rectangle perimeter constraint: w + h <= 2
        # w, h > 0 is handled by bounds
        
        # Wall constraints
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[2+2*N+i]})
            # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[2+2*N+i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[2+2*N+i]})
            # h - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1] - v[2+N+i] - v[2+2*N+i]})
            
        # Circle-circle constraints
        for i in range(N):
            for j in range(i + 1, N):
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[2+i] - v[2+j])**2 + (v[2+N+i] - v[2+N+j])**2 - (v[2+2*N+i] + v[2+2*N+j])**2
                })
                
        # Non-negativity of radii is handled by bounds
        return cons

    # Bounds
    # w in [0.1, 1.9], h in [0.1, 1.9] (since w+h<=2, and we want non-degenerate)
    # Actually, if w=2, h=0, it's degenerate. 
    # Let's bound w, h to [0.2, 1.8] to keep them reasonable.
    bounds = [(0.2, 1.8), (0.2, 1.8)]
    for i in range(N):
        bounds.append((0.0, 2.0)) # x_i
    for i in range(N):
        bounds.append((0.0, 2.0)) # y_i
    for i in range(N):
        bounds.append((0.0, 1.0)) # r_i (max radius in 2x0 rect? No, max r is 1 if 2x0, but w+h=2. Max r is 1 if w=2,h=0? No, circle inside 2x0 is impossible. Max r is 0.5 in 1x1. So 1.0 is safe upper bound.)
        
    best_value = -np.inf
    best_v = None
    
    np.random.seed(42)
    
    # List of initializations to try
    # 1. Square 1x1, Hex init
    # 2. Rect 1.2x0.8, Hex init
    # 3. Rect 0.8x1.2, Hex init
    # 4. Random scatter in 1x1
    # 5. Random scatter in 1.2x0.8
    
    inits = []
    
    # Aspect ratios to try
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67]
    
    for ar in aspects:
        # w * h = 1 (max area for perimeter 4 is 1, at 1x1)
        # Let's fix area to 1 for init, optimizer can change w,h.
        # w = sqrt(ar), h = 1/sqrt(ar)
        w_init = np.sqrt(ar)
        h_init = 1.0 / np.sqrt(ar)
        
        # Hex init
        xs, ys, rs = _hexagonal_init(w_init, h_init, N)
        v0 = np.concatenate([[w_init, h_init], xs, ys, rs])
        inits.append(v0)
        
        # Random init
        xs_r = np.random.uniform(0.1, w_init - 0.1, N)
        ys_r = np.random.uniform(0.1, h_init - 0.1, N)
        rs_r = np.full(N, 0.1)
        v0_r = np.concatenate([[w_init, h_init], xs_r, ys_r, rs_r])
        inits.append(v0_r)
        
    # Shuffle inits to ensure diversity in time
    np.random.shuffle(inits)
    
    for i, v0 in enumerate(inits):
        if time.time() > deadline - 20: # Leave 20s for final polish/check
            break
            
        # Clip initial values to be strictly inside bounds to help SLSQP
        v0_clipped = v0.copy()
        v0_clipped[0] = np.clip(v0_clipped[0], 0.2, 1.8)
        v0_clipped[1] = np.clip(v0_clipped[1], 0.2, 1.8)
        for k in range(N):
            v0_clipped[2+k] = np.clip(v0_clipped[2+k], 0.01, 1.99)
            v0_clipped[2+N+k] = np.clip(v0_clipped[2+N+k], 0.01, 1.99)
            v0_clipped[2+2*N+k] = np.clip(v0_clipped[2+2*N+k], 0.01, 0.99)
            
        try:
            result = minimize(
                objective, v0_clipped, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success:
                w, h, x, y, r = unflatten(result.x)
                # Verify feasibility
                feasible = True
                if w + h > 2.0 + 1e-6:
                    feasible = False
                for k in range(N):
                    if x[k] - r[k] < -1e-5 or w - x[k] - r[k] < -1e-5 or \
                       y[k] - r[k] < -1e-5 or h - y[k] - r[k] < -1e-5 or r[k] < -1e-5:
                        feasible = False
                        break
                    for j in range(k + 1, N):
                        dist2 = (x[k] - x[j])**2 + (y[k] - y[j])**2
                        if dist2 < (r[k] + r[j])**2 - 1e-5:
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if feasible:
                    value = np.sum(r)
                    if value > best_value:
                        best_value = value
                        best_v = result.x.copy()
        except Exception:
            continue

    if best_v is None:
        # Fallback: return a valid trivial packing
        # 21 circles in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        xs = np.array([((i % side) + 0.5) / side for i in range(N)])
        ys = np.array([((i // side) + 0.5) / side for i in range(N)])
        rs = np.full(N, 0.5 / side - 0.01)
        return np.column_stack([xs, ys, rs])
    else:
        w, h, x, y, r = unflatten(best_v)
        return np.column_stack([x, y, r])
