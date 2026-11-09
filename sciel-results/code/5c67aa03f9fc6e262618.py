"""Circle packing in perimeter-4 rectangle using force-directed relaxation + SLSQP polish."""
N = 21
import numpy as np
import time
import random
from scipy.optimize import minimize

def _validate(x, y, r, w, h, tol=1e-7):
    """Check if a packing is valid."""
    if w <= 0 or h <= 0 or abs(w + h - 2.0) > 1e-6:
        return False
    for i in range(N):
        if r[i] < tol:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _force_directed(x, y, r, w, h, iterations=500, alpha=0.05, beta=0.1):
    """
    Force-directed relaxation to adjust positions and radii.
    - Repulsive forces between overlapping circles push them apart.
    - Attractive forces pull circles toward boundaries to maximize radii.
    - Radii are grown gradually while maintaining feasibility.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    r = np.array(r, dtype=float)
    
    for it in range(iterations):
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Circle-circle interactions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        nx = dx / dist
                        ny = dy / dist
                    else:
                        nx = random.uniform(-1, 1)
                        ny = random.uniform(-1, 1)
                        norm = np.sqrt(nx*nx + ny*ny) + 1e-9
                        nx /= norm
                        ny /= norm
                    
                    force = alpha * overlap
                    fx[i] += force * nx
                    fy[i] += force * ny
                    fx[j] -= force * nx
                    fy[j] -= force * ny
                    
                    # Grow radii slightly if there's room
                    # We want to increase radii, so negative force on radius means grow
                    fr[i] -= beta * overlap
                    fr[j] -= beta * overlap
        
        # Boundary interactions: pull circles toward walls to maximize radii
        for i in range(N):
            # Left wall
            if x[i] - r[i] < 1e-4:
                fr[i] -= beta * 0.1  # encourage larger radius
            # Right wall
            if w - x[i] - r[i] < 1e-4:
                fr[i] -= beta * 0.1
            # Bottom wall
            if y[i] - r[i] < 1e-4:
                fr[i] -= beta * 0.1
            # Top wall
            if h - y[i] - r[i] < 1e-4:
                fr[i] -= beta * 0.1
            
            # Keep circles inside boundaries
            if x[i] < r[i]:
                fx[i] += alpha * (r[i] - x[i])
            elif x[i] > w - r[i]:
                fx[i] -= alpha * (x[i] - (w - r[i]))
            
            if y[i] < r[i]:
                fy[i] += alpha * (r[i] - y[i])
            elif y[i] > h - r[i]:
                fy[i] -= alpha * (y[i] - (h - r[i]))
            
            # Limit radius to half width/height
            if r[i] > w / 2.0:
                fr[i] += alpha * (r[i] - w / 2.0)
            if r[i] > h / 2.0:
                fr[i] += alpha * (r[i] - h / 2.0)
        
        # Update positions and radii with damping
        learning_rate = 0.1 * (1.0 - it / iterations)
        x += learning_rate * fx
        y += learning_rate * fy
        r += learning_rate * fr
        
        # Enforce constraints
        for i in range(N):
            r[i] = max(r[i], 0.01)
            r[i] = min(r[i], w / 2.0, h / 2.0)
            x[i] = max(x[i], r[i])
            x[i] = min(x[i], w - r[i])
            y[i] = max(y[i], r[i])
            y[i] = min(y[i], h - r[i])
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_solution = None
    
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary aspect ratio
        w = random.uniform(0.7, 1.3)
        h = 2.0 - w
        
        # Initialize circles using grid-like initialization with jitter
        rows = 5
        cols = 5
        grid_x = np.linspace(0.1, 0.9, cols)
        grid_y = np.linspace(0.1, 0.9, rows)
        
        points = [(i, j) for i in range(cols) for j in range(rows)]
        random.shuffle(points)
        selected = points[:N]
        
        x = np.array([grid_x[i] * w for i, j in selected])
        y = np.array([grid_y[j] * h for i, j in selected])
        
        # Add jitter
        x += random.uniform(-0.05, 0.05) * w
        y += random.uniform(-0.05, 0.05) * h
        
        cell_w = w / cols
        cell_h = h / rows
        r = np.full(N, min(cell_w, cell_h) * 0.4)
        
        # Step 1: Force-directed relaxation
        x, y, r = _force_directed(x, y, r, w, h, iterations=300)
        
        # Step 2: SLSQP polish
        def objective(params):
            w_val = params[0]
            h_val = 2.0 - w_val
            
            if w_val <= 0.01 or h_val <= 0.01:
                return 1e6
            
            x_val = params[1 : 1+N]
            y_val = params[1+N : 1+2*N]
            r_val = params[1+2*N : 1+3*N]
            
            r_val = np.maximum(r_val, 0.01)
            
            bound_pen = 0.0
            for i in range(N):
                if x_val[i] < r_val[i]:
                    bound_pen += (r_val[i] - x_val[i]) ** 2
                elif x_val[i] > w_val - r_val[i]:
                    bound_pen += (x_val[i] - (w_val - r_val[i])) ** 2
                
                if y_val[i] < r_val[i]:
                    bound_pen += (r_val[i] - y_val[i]) ** 2
                elif y_val[i] > h_val - r_val[i]:
                    bound_pen += (y_val[i] - (h_val - r_val[i])) ** 2
                
                if r_val[i] > w_val / 2.0:
                    bound_pen += (r_val[i] - w_val / 2.0) ** 2
                if r_val[i] > h_val / 2.0:
                    bound_pen += (r_val[i] - h_val / 2.0) ** 2
            
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x_val[i] - x_val[j]
                    dy = y_val[i] - y_val[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r_val[i] + r_val[j]
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq + 1e-9)
                        overlap_pen += (min_dist - dist) ** 2
            
            penalty = 100.0 * (bound_pen + overlap_pen)
            return -np.sum(r_val) + penalty
        
        x0 = np.concatenate(([w], x, y, r))
        
        try:
            bounds = [(0.1, 1.9)] + [(0, w)] * N + [(0, h)] * N + [(0.01, min(w, h)/2)] * N
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                          options={'maxiter': 50, 'ftol': 1e-10})
        except Exception:
            continue
            
        if time.time() > deadline:
            break
        
        res_w = res.x[0]
        res_h = 2.0 - res_w
        if res_w <= 0.01 or res_h <= 0.01:
            continue
            
        res_x = res.x[1 : 1+N]
        res_y = res.x[1+N : 1+2*N]
        res_r = res.x[1+2*N : 1+3*N]
        res_r = np.maximum(res_r, 0.01)
        
        if not _validate(res_x, res_y, res_r, res_w, res_h):
            continue
        
        score = np.sum(res_r)
        if score > best_score:
            best_score = score
            best_solution = np.column_stack((res_x, res_y, res_r))
            
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
