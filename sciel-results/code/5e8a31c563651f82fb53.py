import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s buffer for safety
    
    # Variables: [x1, y1, r1, ..., xN, yN, rN]
    # Objective: minimize -sum(r)
    def objective(z):
        radii = z[2::3]
        return -np.sum(radii)
    
    # Constraints
    # 1. Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    # 2. Non-overlap:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def get_constraints():
        cons = []
        # Boundary constraints
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z: z[0::3] - z[2::3]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z: 1 - z[0::3] - z[2::3]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z: z[1::3] - z[2::3]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z: 1 - z[1::3] - z[2::3]})
        
        # Non-overlap constraints
        # We can vectorize this or use a loop. With N=26, there are 325 pairs.
        # A loop is fine and clearer, but vectorized is faster.
        # Let's use a vectorized approach for the distance check if possible, 
        # but scipy minimize expects a function returning an array.
        
        def non_overlap(z):
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            
            # Create matrices of positions and radii
            # We need (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2
            # This is O(N^2).
            
            # Efficient vectorization
            # dx = x[:, None] - x[None, :]
            # dy = y[:, None] - y[None, :]
            # dr = r[:, None] + r[None, :]
            # dist_sq = dx**2 + dy**2
            # min_dist_sq = dr**2
            # constraint = dist_sq - min_dist_sq
            # We only need the upper triangle (i < j)
            
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dr = r[idx_i] + r[idx_j]
            return dx**2 + dy**2 - dr**2
            
        cons.append({'type': 'ineq', 'fun': non_overlap})
        return cons
    
    constraints = get_constraints()
    
    # Initial guesses
    # 1. Grid based
    # 2. Hexagonal based
    # 3. Random
    
    best_val = -np.inf
    best_z = None
    
    # Helper to check validity and update best
    def update_best(res):
        nonlocal best_val, best_z
        if res.success:
            val = -res.fun
            # Check constraints strictly to ensure validity
            # SLSQP might return slightly invalid points
            # Let's do a quick check
            z = res.x
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            
            # Check bounds
            if np.any(x < r - 1e-8) or np.any(1 - x < r - 1e-8) or \
               np.any(y < r - 1e-8) or np.any(1 - y < r - 1e-8):
                return
            
            # Check overlaps
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dr = r[idx_i] + r[idx_j]
            dist_sq = dx**2 + dy**2
            if np.any(dist_sq < dr**2 - 1e-8):
                return
                
            if val > best_val:
                best_val = val
                best_z = z.copy()

    # Generate initial guesses
    inits = []
    
    # 1. Simple Grid (6x5 grid, take 26)
    side = 6
    grid_x = (np.arange(side) + 0.5) / side
    grid_y = (np.arange(5) + 0.5) / 5
    gx, gy = np.meshgrid(grid_x, grid_y)
    gx = gx.ravel()
    gy = gy.ravel()
    # Take first 26
    gx = gx[:26]
    gy = gy[:26]
    gr = np.full(26, 0.5/6 - 0.01)
    z_init = np.zeros(78)
    z_init[0::3] = gx
    z_init[1::3] = gy
    z_init[2::3] = gr
    inits.append(z_init)
    
    # 2. Hexagonal Pattern
    # Approximate hex packing
    # r = 0.09 approx for 26 circles in unit square?
    # Let's try a few random hex-like starts
    for trial in range(5):
        # Random center
        cx = 0.5 + np.random.uniform(-0.2, 0.2)
        cy = 0.5 + np.random.uniform(-0.2, 0.2)
        r = 0.08 + np.random.uniform(0, 0.02)
        
        # Generate hex points
        pts = []
        # Center
        pts.append((cx, cy))
        # Rings
        for ring in range(1, 4):
            count = 6 * ring
            for i in range(count):
                angle = 2 * np.pi * i / count
                # Hex lattice spacing is sqrt(3)*r
                dist = ring * np.sqrt(3) * r
                # Adjust angle for hex vs square grid
                # Simple polar is okay for init
                px = cx + dist * np.cos(angle)
                py = cy + dist * np.sin(angle)
                pts.append((px, py))
        
        # Select 26 points
        # We might have more or fewer, let's just take the first 26 or pad
        while len(pts) < 26:
            # Add random points
            pts.append((np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)))
            
        pts = pts[:26]
        hx = np.array([p[0] for p in pts])
        hy = np.array([p[1] for p in pts])
        hr = np.full(26, r)
        
        z_init = np.zeros(78)
        z_init[0::3] = hx
        z_init[1::3] = hy
        z_init[2::3] = hr
        inits.append(z_init)
        
    # 3. Random Starts
    for trial in range(10):
        rx = np.random.uniform(0.1, 0.9, 26)
        ry = np.random.uniform(0.1, 0.9, 26)
        rr = np.random.uniform(0.05, 0.15, 26)
        z_init = np.zeros(78)
        z_init[0::3] = rx
        z_init[1::3] = ry
        z_init[2::3] = rr
        inits.append(z_init)

    # Optimize
    for z0 in inits:
        if time.time() - start_time > deadline:
            break
            
        try:
            # Bounds: x,y in [0,1], r in [0, 0.5]
            bounds = []
            for i in range(N):
                bounds.append((0, 1)) # x
                bounds.append((0, 1)) # y
                bounds.append((0, 0.5)) # r
                
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
            update_best(res)
        except Exception:
            pass

    # If no valid result found, fallback to a very small valid configuration
    if best_z is None:
        # Small circles in grid
        side = 6
        cells = []
        for i in range(6):
            for j in range(5):
                if len(cells) >= 26: break
                x = (i + 0.5) / 6
                y = (j + 0.5) / 5
                r = 0.01
                cells.append((x, y, r))
        best_z = np.array([v for c in cells for v in c]).reshape(78)
        
    # Reshape to (26, 3)
    result = best_z.reshape(26, 3)
    
    # Final sanity check and clamp
    # Ensure inside [0,1]
    result[:, 0] = np.clip(result[:, 0], 0, 1)
    result[:, 1] = np.clip(result[:, 1], 0, 1)
    result[:, 2] = np.clip(result[:, 2], 0, 0.5)
    
    # Ensure disjoint by shrinking if necessary (last resort)
    # This is a simple iterative fix
    for _ in range(10):
        valid = True
        x = result[:, 0]
        y = result[:, 1]
        r = result[:, 2]
        idx_i, idx_j = np.triu_indices(26, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dr = r[idx_i] + r[idx_j]
        dist_sq = dx**2 + dy**2
        overlaps = dist_sq < dr**2
        if not np.any(overlaps):
            break
        # Shrink all radii slightly
        result[:, 2] *= 0.95
        
    return result

# EVOLVE-BLOCK-START
# The code above is the new implementation
# EVOLVE-BLOCK-END
