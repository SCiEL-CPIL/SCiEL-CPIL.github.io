"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _make_constraints(n):
    """Create inequality constraints for circle packing.
    Each pair i<j: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    Equivalent: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    """
    cons = []
    for i in range(n):
        for j in range(i+1, n):
            def con(args, i=i, j=j):
                x, y, r, w = args[:n], args[n:2*n], args[2*n:3*n], args[3*n]
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                return dx*dx + dy*dy - dr*dr
            cons.append({'type': 'ineq', 'fun': con})
    
    # Boundary constraints: circles must be inside rectangle [0,w] x [0,h]
    # where h = 2 - w (perimeter = 2(w+h) = 4 => w+h=2)
    # x_i - r_i >= 0
    for i in range(n):
        def con_left(args, i=i):
            x, y, r, w = args[:n], args[n:2*n], args[2*n:3*n], args[3*n]
            return x[i] - r[i]
        def con_right(args, i=i):
            x, y, r, w = args[:n], args[n:2*n], args[2*n:3*n], args[3*n]
            h = 2.0 - w
            return (w - x[i]) - r[i]
        def con_bottom(args, i=i):
            x, y, r, w = args[:n], args[n:2*n], args[2*n:3*n], args[3*n]
            return y[i] - r[i]
        def con_top(args, i=i):
            x, y, r, w = args[:n], args[n:2*n], args[2*n:3*n], args[3*n]
            h = 2.0 - w
            return (h - y[i]) - r[i]
        cons.append({'type': 'ineq', 'fun': con_left})
        cons.append({'type': 'ineq', 'fun': con_right})
        cons.append({'type': 'ineq', 'fun': con_bottom})
        cons.append({'type': 'ineq', 'fun': con_top})
    
    # Width constraint: w > 0 and h > 0 => 0 < w < 2
    def con_w_min(args):
        return args[3*n] - 0.1
    def con_w_max(args):
        return 1.9 - args[3*n]
    cons.append({'type': 'ineq', 'fun': con_w_min})
    cons.append({'type': 'ineq', 'fun': con_w_max})
    
    return cons


def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = N
    best_value = -np.inf
    best_result = None
    
    # Grid-based cellular decomposition for initial placement
    # Try multiple grid configurations and aspect ratios
    configs = []
    
    # Square-ish grids
    for side in range(3, 8):
        for extra in range(0, 4):
            total = side * side + extra
            if total >= n:
                configs.append((side, extra, 1.0))  # aspect ratio w/h
                configs.append((side, extra, 1.5))
                configs.append((side, extra, 0.7))
    
    # Also try some rectangular grids
    for w_cells in range(3, 8):
        for h_cells in range(3, 8):
            if w_cells * h_cells >= n:
                configs.append((w_cells, 0, float(w_cells)/h_cells))
                configs.append((w_cells, 0, float(w_cells)/(h_cells*1.2)))
    
    # Remove duplicates and limit
    seen = set()
    unique_configs = []
    for cfg in configs:
        key = (cfg[0], cfg[1], round(cfg[2], 2))
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)
    configs = unique_configs[:30]  # Limit to 30 configs
    
    for (side, extra, aspect) in configs:
        if time.time() > deadline:
            break
        
        # Create grid cells
        w = aspect  # width
        h = 2.0 / (1.0 + aspect)  # height, since w + h = 2
        # Actually: perimeter = 2(w+h) = 4 => w+h = 2
        # If aspect = w/h, then w = aspect*h, so aspect*h + h = 2 => h = 2/(1+aspect)
        
        # Place circles in grid
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.4 * min(w/side, h/side)  # Slightly less than inradius
                cells.append((cx, cy, r))
                count += 1
            if count >= n:
                break
        
        # Add extra circles if needed
        while len(cells) < n:
            # Place extra circles at random positions
            cx = np.random.uniform(0.1, w - 0.1)
            cy = np.random.uniform(0.1, h - 0.1)
            r = 0.1 * min(w, h)
            cells.append((cx, cy, r))
        
        cells = cells[:n]
        
        # Initial point
        x0 = np.array([c[0] for c in cells])
        y0 = np.array([c[1] for c in cells])
        r0 = np.array([c[2] for c in cells])
        w0 = np.array([w])
        
        x_init = np.concatenate([x0, y0, r0, w0])
        
        # Objective: maximize sum of radii => minimize -sum(r)
        def objective(args):
            r = args[2*n:3*n]
            return -np.sum(r)
        
        # Constraints
        constraints = _make_constraints(n)
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((1e-6, None))  # x_i
        for i in range(n):
            bounds.append((1e-6, None))  # y_i
        for i in range(n):
            bounds.append((1e-6, 1.0))  # r_i
        bounds.append((0.1, 1.9))  # w
        
        # Try SLSQP
        try:
            result = minimize(
                objective,
                x_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < -np.inf:
                value = -result.fun
                # Verify validity
                x_opt = result.x[:n]
                y_opt = result.x[n:2*n]
                r_opt = result.x[2*n:3*n]
                w_opt = result.x[3*n]
                h_opt = 2.0 - w_opt
                
                valid = True
                # Check boundaries
                for i in range(n):
                    if x_opt[i] - r_opt[i] < -1e-6 or w_opt - x_opt[i] - r_opt[i] < -1e-6:
                        valid = False
                        break
                    if y_opt[i] - r_opt[i] < -1e-6 or h_opt - y_opt[i] - r_opt[i] < -1e-6:
                        valid = False
                        break
                
                # Check pairwise distances
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = x_opt[i] - x_opt[j]
                            dy = y_opt[i] - y_opt[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r_opt[i] + r_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid and value > best_value:
                    best_value = value
                    best_result = np.column_stack([x_opt, y_opt, r_opt])
                    
        except Exception:
            continue
    
    # If no valid result found, return a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
