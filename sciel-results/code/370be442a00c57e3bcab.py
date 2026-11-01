import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110  # Leave 10s margin for 120s limit

    # Initial guess: Hexagonal lattice
    # Approximate number of rows/cols
    # For N=32, a 6x6 grid is 36, so we can take 32 from a 6x6 hex lattice.
    # Or better, generate a hex lattice that fits in [0,1]^2.
    
    def generate_hex_initial():
        # Try to fit hex lattice
        # Let's try a few different scales and pick the best fit for N circles
        best_init = None
        best_score = -np.inf
        
        # Scale factor s controls the size of the hex cells
        # Radius r = s * 0.5 * sqrt(3) / 2 ? No, let's just use spacing.
        # Hex spacing dx = 2*r, dy = r*sqrt(3)
        
        for scale in np.linspace(0.1, 0.3, 20):
            r = scale * 0.4 # approximate radius
            dx = 2 * r * 1.05 # small gap
            dy = 2 * r * 0.866 * 1.05
            
            pts = []
            row = 0
            y = r
            while y <= 1 - r:
                x_offset = 0 if row % 2 == 0 else dx / 2
                x = r + x_offset
                while x <= 1 - r:
                    pts.append((x, y, r))
                    x += dx
                y += dy
                row += 1
            
            if len(pts) < N:
                continue
                
            # If we have more than N, take the first N? Or center them?
            # Taking first N might bias to corner.
            # Let's just use the first N points from a centered grid if possible.
            # For now, simple truncation.
            if len(pts) >= N:
                init = np.array(pts[:N])
                # Calculate initial penalty to see if it's valid
                # Just pick the one with largest sum of radii (which is constant here)
                # or just pick the first one that fits.
                if best_init is None:
                    best_init = init
                    best_score = np.sum(init[:, 2])
                    
        if best_init is None:
            # Fallback to grid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            best_init = np.array(cells[:N])
            
        return best_init

    x0 = generate_hex_initial()
    
    # Flatten x0 to 1D array: [x1, y1, r1, x2, y2, r2, ...]
    v0 = x0.flatten()

    def objective(v):
        if time.time() - start_time > timeout:
            return np.inf
            
        v = v.reshape(-1, 3)
        x = v[:, 0]
        y = v[:, 1]
        r = v[:, 2]
        
        # Base objective: maximize sum of radii -> minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalty for overlaps
        # Distance matrix
        # d_ij^2 = (xi-xj)^2 + (yi-yj)^2
        # We want d_ij >= ri + rj
        # penalty = sum max(0, ri + rj - d_ij)^2
        
        # Efficient calculation of pairwise distances
        # For N=32, N^2 is small, so explicit loops or vectorized are fine.
        # Vectorized:
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        # Overlap amount
        overlap = r_sum - d
        # Only consider i < j to avoid double counting and self-interaction
        # Create mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlaps = overlap[mask]
        
        # Penalty
        pen = np.sum(np.maximum(0, overlaps)**2)
        
        # Penalty for boundary violations
        # x >= r, x <= 1-r, y >= r, y <= 1-r
        b_pen = 0
        b_pen += np.sum(np.maximum(0, r - x)**2)
        b_pen += np.sum(np.maximum(0, r - (1 - x))**2)
        b_pen += np.sum(np.maximum(0, r - y)**2)
        b_pen += np.sum(np.maximum(0, r - (1 - y))**2)
        
        return obj + 10 * (pen + b_pen)

    # Bounds for each variable
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Run optimization
    res = None
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False})
        
        if time.time() - start_time > timeout:
             # If we hit timeout during optimization, use the last known good?
             # L-BFGS-B doesn't easily yield intermediate results.
             # We rely on the timeout check inside objective to return inf, 
             # but that might stop it abruptly or return a bad result.
             # It's safer to just let it run and hope it finishes, 
             # or use a smaller maxiter if we are close to deadline.
             pass

    except Exception as e:
        # If optimization fails, return initial guess
        pass

    if res is not None and res.success:
        v_opt = res.x.reshape(-1, 3)
    else:
        # If it didn't succeed, maybe use the best point found?
        # L-BFGS-B result might still be better than x0 even if not 'success'
        if res is not None:
             v_opt = res.x.reshape(-1, 3)
        else:
             v_opt = x0

    # Final validation and cleanup
    # Ensure circles are strictly inside and disjoint
    # If there are small violations, shrink radii slightly
    x = v_opt[:, 0]
    y = v_opt[:, 1]
    r = v_opt[:, 2]
    
    # Check boundaries
    r = np.minimum(r, x)
    r = np.minimum(r, 1 - x)
    r = np.minimum(r, y)
    r = np.minimum(r, 1 - y)
    r = np.maximum(r, 0)
    
    # Check overlaps
    # Iterative shrinking if overlaps exist
    # This is a simple repair step
    for _ in range(10):
        if time.time() - start_time > timeout:
            break
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlaps = r_sum - d
        if np.all(overlaps[mask] <= 1e-5):
            break
        
        # Shrink radii of overlapping pairs
        # Simple heuristic: reduce both radii slightly
        for i in range(N):
            for j in range(i+1, N):
                if r[i] + r[j] > d[i,j]:
                    # Reduce radii
                    factor = 0.95
                    r[i] *= factor
                    r[j] *= factor
                    # Also check boundaries again
                    r[i] = np.minimum(r[i], x[i], 1-x[i], y[i], 1-y[i])
                    r[j] = np.minimum(r[j], x[j], 1-x[j], y[j], 1-y[j])
                    r[i] = np.maximum(r[i], 0)
                    r[j] = np.maximum(r[j], 0)

    v_opt[:, 2] = r
    return v_opt
