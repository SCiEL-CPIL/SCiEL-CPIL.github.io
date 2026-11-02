import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Variables:
    # v[0] = w (width)
    # v[1] = r (radius)
    # v[2:2+N] = x coordinates
    # v[2+N:2+2*N] = y coordinates
    # Total size: 2 + 2*N = 44

    def unpack(v):
        w = v[0]
        r = v[1]
        xs = v[2:2+N]
        ys = v[2+N:2+2*N]
        return w, r, xs, ys

    def objective(v):
        # Maximize r, so minimize -r
        w, r, xs, ys = unpack(v)
        # Ensure valid dimensions
        h = 2.0 - w
        if w <= 0 or h <= 0 or r <= 0:
            return 1e6
        return -r

    def constraints(v):
        w, r, xs, ys = unpack(v)
        h = 2.0 - w
        cons = []
        
        # Rectangle aspect ratio constraints (keep it reasonable)
        # w in [0.5, 1.5] roughly? Perimeter 4.
        # If w is very small, h is large, packing is bad.
        cons.append({'type': 'ineq', 'fun': lambda v: v[0] - 0.2}) # w >= 0.2
        cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[0] - 0.2}) # h >= 0.2
        
        # Circles inside rectangle
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+i] - v[1]}) # x_i >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[2+i] - v[1]}) # w - x_i >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+N+i] - v[1]}) # y_i >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+N+i] - v[1]}) # h - y_i >= r
            
        # Disjoint circles
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: np.sqrt((v[2+i]-v[2+j])**2 + (v[2+N+i]-v[2+N+j])**2) - 2*v[1]
                })
        return cons

    # Initial Guess: Hexagonal Packing
    # Generate a hexagonal grid
    # Let's try to fit 21 circles in a hexagonal pattern
    # Rows: 4, 5, 4, 4, 4 ? Sum = 21.
    # Or 7, 7, 7 ? Sum = 21.
    # Let's use 7, 7, 7 for compactness.
    
    rows = [7, 7, 7]
    pts = []
    row_height = np.sqrt(3) * 0.5 # Assuming r=0.5 for generation, will scale later
    for k, n_circles in enumerate(rows):
        y_k = k * row_height
        # Center the row
        for i in range(n_circles):
            # Offset alternate rows by 0.5
            offset = 0.5 if k % 2 == 1 else 0.0
            x_k = i + offset
            pts.append((x_k, y_k))
            
    pts = np.array(pts)
    
    # Center the points
    pts -= np.mean(pts, axis=0)
    
    # Scale to fit in a 1x1 square initially?
    # Find max extent
    max_x = np.max(np.abs(pts[:, 0]))
    max_y = np.max(np.abs(pts[:, 1]))
    
    # Let's start with a square w=1, h=1
    w0 = 1.0
    h0 = 1.0
    
    # Scale points to fit in 1x1 with some margin
    # We want radius r0.
    # In hex packing, distance between centers is 2r.
    # Our generated points have distance 1.0 between horizontal neighbors.
    # So 2r_gen = 1.0 => r_gen = 0.5.
    # Scale factor s: new_r = s * 0.5.
    # We want the bounding box of circles to fit in w0 x h0.
    # Bounding box of centers: width = 2*max_x, height = 2*max_y.
    # Add 2*r to each dimension for the circle extents.
    # We want (2*max_x * s + 2*r_new) <= w0
    # (2*max_y * s + 2*r_new) <= h0
    # r_new = 0.5 * s
    # 2*max_x * s + 1.0 * s <= w0  => s(2*max_x + 1) <= w0
    # 2*max_y * s + 1.0 * s <= h0  => s(2*max_y + 1) <= h0
    # s <= min(w0 / (2*max_x + 1), h0 / (2*max_y + 1))
    
    scale_limit_x = w0 / (2*max_x + 1)
    scale_limit_y = h0 / (2*max_y + 1)
    s = min(scale_limit_x, scale_limit_y)
    
    # Apply scaling
    pts_scaled = pts * s
    r0 = 0.5 * s
    
    # Center in the rectangle
    # Current center is (0,0). Rectangle center is (w0/2, h0/2).
    xs0 = pts_scaled[:, 0] + w0/2
    ys0 = pts_scaled[:, 1] + h0/2
    
    # Initial vector
    v0 = np.zeros(2 + 2*N)
    v0[0] = w0
    v0[1] = r0
    v0[2:2+N] = xs0
    v0[2+N:2+2*N] = ys0
    
    # Bounds
    # w in [0.5, 1.5] (since h=2-w, this keeps h in [0.5, 1.5])
    # r in [0.01, 1.0]
    # x in [0, w] -> depends on w, so use loose bounds [0, 2]
    # y in [0, h] -> [0, 2]
    bounds = [(0.5, 1.5), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)
    
    # Optimization
    best_result = None
    best_val = -np.inf
    
    # Try SLSQP
    try:
        res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 500, 'ftol': 1e-8, 'disp': False})
        
        if res.success or res.fun < best_val:
            # Check validity
            w, r, xs, ys = unpack(res.x)
            h = 2.0 - w
            valid = True
            if w <= 0 or h <= 0 or r <= 0:
                valid = False
            else:
                for i in range(N):
                    if xs[i] < r or xs[i] > w - r or ys[i] < r or ys[i] > h - r:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            if np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) < 2*r - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                best_val = res.fun # -r
                best_result = res.x
    except Exception as e:
        pass

    # If SLSQP failed or didn't improve, try a few random restarts or different initializations
    # But given time constraints, one good run is better.
    # Let's also try a square grid initialization just in case.
    
    if best_result is None or best_val > -2.0: # If score is worse than 2.0 (r < 2.0/21 ~ 0.095), try another
        # Try square grid 5x5 (using 21)
        side = 5
        w0 = 1.0
        h0 = 1.0
        r0 = 0.5 / side # 0.1
        xs0 = np.array([(i + 0.5) / side for i in range(N)])
        ys0 = np.array([(j + 0.5) / side for j in range(N)])
        
        v0_grid = np.zeros(2 + 2*N)
        v0_grid[0] = w0
        v0_grid[1] = r0
        v0_grid[2:2+N] = xs0
        v0_grid[2+N:2+2*N] = ys0
        
        try:
            res_grid = minimize(objective, v0_grid, method='SLSQP', bounds=bounds, constraints=constraints, 
                                options={'maxiter': 300, 'ftol': 1e-8, 'disp': False})
            if res_grid.success and res_grid.fun < best_val:
                 # Validate
                 w, r, xs, ys = unpack(res_grid.x)
                 h = 2.0 - w
                 valid = True
                 if w <= 0 or h <= 0 or r <= 0:
                     valid = False
                 else:
                     for i in range(N):
                         if xs[i] < r or xs[i] > w - r or ys[i] < r or ys[i] > h - r:
                             valid = False
                             break
                     if valid:
                         for i in range(N):
                             for j in range(i+1, N):
                                 if np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) < 2*r - 1e-6:
                                     valid = False
                                     break
                             if not valid:
                                 break
                 if valid:
                     best_val = res_grid.fun
                     best_result = res_grid.x
        except Exception:
            pass

    # Check time
    if time.time() > deadline - 5:
         # Return the best valid result or fallback
         if best_result is not None:
             w, r, xs, ys = unpack(best_result)
             result = np.column_stack((xs, ys, np.full(N, r)))
             return result
         else:
             # Fallback to seed
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    if best_result is not None:
        w, r, xs, ys = unpack(best_result)
        result = np.column_stack((xs, ys, np.full(N, r)))
        return result
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
