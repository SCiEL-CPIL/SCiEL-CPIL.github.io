"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 150.0  # wall-clock limit

def construct():
    t0 = time.time()
    
    # Try multiple aspect ratios for the rectangle (w + h = 2)
    best_val = -1.0
    best_result = None
    
    # Candidate aspect ratios: w/h ratio
    # Perimeter 4: w + h = 2, so h = 2 - w
    # Try w from 0.5 to 1.5 (h from 1.5 to 0.5)
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for w in aspect_ratios:
        if time.time() - t0 > DEADLINE - 10:
            break
        h = 2.0 - w
        if h < 0.1:
            continue
        
        # Initial grid layout: arrange 21 circles in rows
        # Try different row/column configurations
        configs = []
        # 5 cols x 4 rows = 20, plus 1 extra -> 5x4 + 1
        configs.append((5, 4))  # 5 cols, 4 rows (20 circles) + 1
        configs.append((4, 5))  # 4 cols, 5 rows
        configs.append((7, 3))  # 7 cols, 3 rows
        configs.append((3, 7))
        configs.append((6, 4))  # 6 cols, 4 rows = 24, use 21
        configs.append((4, 6))
        configs.append((5, 5))  # 5x5 grid, use 21
        
        for (ncols, nrows) in configs:
            if time.time() - t0 > DEADLINE - 5:
                break
            if ncols * nrows < N:
                continue
            
            # Create initial positions in a grid
            xs = np.linspace(0.15 * w, 0.85 * w, ncols) if ncols > 1 else np.array([w / 2])
            ys = np.linspace(0.15 * h, 0.85 * h, nrows) if nrows > 1 else np.array([h / 2])
            
            # Generate N positions from the grid
            init_pos = []
            for j in range(nrows):
                for i in range(ncols):
                    if len(init_pos) < N:
                        init_pos.append([xs[i], ys[j]])
            
            if len(init_pos) < N:
                continue
            
            init_pos = np.array(init_pos[:N])
            
            # Initial radius estimate: based on grid spacing
            if ncols > 1:
                dx = xs[1] - xs[0]
            else:
                dx = w
            if nrows > 1:
                dy = ys[1] - ys[0]
            else:
                dy = h
            r_init = min(dx, dy) / 2.0 * 0.95
            r_init = max(r_init, 0.02)
            
            # Variables: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3*i] = init_pos[i, 0]
                x0[3*i+1] = init_pos[i, 1]
                x0[3*i+2] = r_init
            
            def objective(vars):
                # Maximize sum of radii -> minimize negative sum
                return -sum(vars[3*i+2] for i in range(N))
            
            def constraints(vars):
                cons = []
                # Non-overlap: distance between centers >= sum of radii
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
                        xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                        # (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
                        # Linearize: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                        cons.append((xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2)
                # Boundary: circle inside rectangle
                for i in range(N):
                    xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
                    # x - r >= 0
                    cons.append(xi - ri)
                    # x + r <= w
                    cons.append(w - xi - ri)
                    # y - r >= 0
                    cons.append(yi - ri)
                    # y + r <= h
                    cons.append(h - yi - ri)
                # Radius positive
                for i in range(N):
                    cons.append(vars[3*i+2])
                return cons
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0, w))  # x
                bounds.append((0, h))  # y
                bounds.append((0.01, min(w, h)/2))  # r
            
            # Convert to scipy constraint format
            scipy_cons = []
            for i in range(len(constraints(np.zeros(3*N)))):
                pass  # We'll build them properly
            
            # Build constraints list for scipy
            n_pairs = N * (N-1) // 2
            n_boundary = 4 * N
            n_radius = N
            total_cons = n_pairs + n_boundary + n_radius
            
            def make_constraint(i):
                if i < n_pairs:
                    # Non-overlap constraint i
                    # Map linear index to (a, b) pair
                    # Pairs: (0,1),(0,2),...,(0,N-1),(1,2),(1,3),...
                    a = 0
                    b = 1
                    count = 0
                    for aa in range(N):
                        for bb in range(aa+1, N):
                            if count == i:
                                a, b = aa, bb
                                break
                            count += 1
                        if count == i:
                            break
                    def non_overlap(vars):
                        xi, yi, ri = vars[3*a], vars[3*a+1], vars[3*a+2]
                        xj, yj, rj = vars[3*b], vars[3*b+1], vars[3*b+2]
                        return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
                    return {'type': 'ineq', 'fun': non_overlap}
                elif i < n_pairs + n_boundary:
                    idx = i - n_pairs
                    circle = idx // 4
                    side = idx % 4
                    def boundary(vars, c=circle, s=side):
                        x, y, r = vars[3*c], vars[3*c+1], vars[3*c+2]
                        if s == 0: return x - r
                        elif s == 1: return w - x - r
                        elif s == 2: return y - r
                        else: return h - y - r
                    return {'type': 'ineq', 'fun': boundary}
                else:
                    idx = i - n_pairs - n_boundary
                    def radius_pos(vars, c=idx):
                        return vars[3*c+2]
                    return {'type': 'ineq', 'fun': radius_pos}
            
            scipy_cons = [make_constraint(i) for i in range(total_cons)]
            
            # Run SLSQP
            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                                  constraints=scipy_cons,
                                  options={'maxiter': 500, 'ftol': 1e-8})
            except Exception:
                continue
            
            if result.success or result.fun < objective(x0):
                val = -result.fun
                if val > best_val:
                    best_val = val
                    # Build output array
                    out = np.zeros((N, 3))
                    for i in range(N):
                        out[i, 0] = result.x[3*i]
                        out[i, 1] = result.x[3*i+1]
                        out[i, 2] = result.x[3*i+2]
                    best_result = out
            
            # Check deadline
            if time.time() - t0 > DEADLINE - 15:
                break
    
    # If no valid result from optimization, fall back to grid
    if best_result is None or best_val <= 0:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    # Verify and adjust: ensure circles are valid (inside, non-overlapping)
    # Add small shrink to ensure strict validity
    if best_result is not None:
        # Check perimeter
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]
        w = 2.0  # We'll compute actual bounding box
        # Actually, we need to ensure the bounding rectangle has perimeter <= 4
        # The circles are placed in a w x h rectangle where w+h=2
        # But we need to verify the actual bounding box of circles
        
        # For safety, slightly shrink all radii
        best_result[:, 2] *= 0.999
        
        # Verify non-overlap and boundaries
        # Find the actual rectangle that contains all circles
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        rect_w = xmax - xmin
        rect_h = ymax - ymin
        perimeter = 2 * (rect_w + rect_h)
        
        if perimeter > 4.0:
            # Scale down to fit
            scale = 4.0 / perimeter
            if scale < 1.0:
                # Shift to origin and scale
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                for i in range(N):
                    best_result[i, 0] = (best_result[i, 0] - cx) * scale + cx
                    best_result[i, 1] = (best_result[i, 1] - cy) * scale + cy
                    best_result[i, 2] = best_result[i, 2] * scale
                # Re-verify
                xs = best_result[:, 0]
                ys = best_result[:, 1]
                rs = best_result[:, 2]
                xmin = np.min(xs - rs)
                xmax = np.max(xs + rs)
                ymin = np.min(ys - rs)
                ymax = np.max(ys + rs)
                rect_w = xmax - xmin
                rect_h = ymax - ymin
                perimeter = 2 * (rect_w + rect_h)
                if perimeter > 4.0:
                    # Last resort: use fallback grid
                    side = int(np.ceil(np.sqrt(N)))
                    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                             for i in range(side) for j in range(side)]
                    best_result = np.array(cells[:N])
    
    return best_result
