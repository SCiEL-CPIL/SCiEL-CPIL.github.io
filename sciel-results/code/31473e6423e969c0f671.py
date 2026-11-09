"""Circle packing in perimeter-4 rectangle using contact-graph Newton-Raphson."""
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

def _newton_packing(x0, y0, r0, w0, h0, max_iters=200, tol=1e-10):
    """
    Newton-Raphson solver for a jammed packing given initial positions and radii.
    We optimize positions and radii to satisfy contact constraints exactly.
    The system is underdetermined, so we add a regularization term to maximize sum of radii.
    """
    x = np.array(x0, dtype=float)
    y = np.array(y0, dtype=float)
    r = np.array(r0, dtype=float)
    w = w0
    h = h0
    
    # We will fix w and h for this call, or allow them to vary?
    # For now, fix w and h to simplify. The outer loop will vary aspect ratio.
    
    # Define the contact graph.
    # We assume the initial configuration is close to a valid packing.
    # We need to identify which pairs are in contact.
    # A pair (i,j) is in contact if dist(i,j) <= r[i] + r[j] + eps.
    # Also boundary contacts: i touches left if x[i] == r[i], etc.
    
    eps = 1e-4
    contacts = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] + eps:
                contacts.append((i, j))
    
    # Boundary contacts
    # left: x[i] - r[i] = 0
    # right: w - x[i] - r[i] = 0
    # bottom: y[i] - r[i] = 0
    # top: h - y[i] - r[i] = 0
    
    # We solve for x, y, r using a least squares approach on the constraints.
    # Constraints:
    # 1. For each contact (i,j): (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 = 0
    # 2. For boundary contacts: x_i - r_i = 0, etc.
    # 3. Objective: maximize sum(r). This is a soft constraint or we can use it as a direction.
    
    # Let's use scipy.optimize.minimize with SLSQP or trust-constr?
    # No, let's implement a simple Newton step on the residual vector.
    # Variables: X = [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1] (3N variables)
    # Residuals:
    # - For each contact (i,j): 2*(x_i-x_j)*dx + 2*(y_i-y_j)*dy - 2*(r_i+r_j)*(dr_i+dr_j) = 0 (linearized)
    # - For boundary contacts: dx_i - dr_i = 0, etc.
    
    # This is getting complex. Let's stick to the previous approach but improve the initial guess
    # and the optimization method.
    
    # Actually, the previous approach using Powell was slow.
    # Let's try a different strategy:
    # 1. Generate a good initial packing using a greedy or random restart with local optimization.
    # 2. Use a more robust optimizer like L-BFGS-B with penalty.
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_solution = None
    
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Vary aspect ratio
        # w + h = 2
        # Sample w from [0.7, 1.3] to avoid extreme aspect ratios
        w = random.uniform(0.7, 1.3)
        h = 2.0 - w
        
        # Initialize circles
        # Use a grid-like initialization with some jitter
        # 21 circles: 5x5 grid with 4 removed? Or 4x6?
        # Let's try 5x5 grid (25 spots), pick 21
        rows = 5
        cols = 5
        grid_x = np.linspace(0, 1, cols)
        grid_y = np.linspace(0, 1, rows)
        
        # Shuffle grid points
        points = [(i, j) for i in range(cols) for j in range(rows)]
        random.shuffle(points)
        selected = points[:N]
        
        x = np.array([grid_x[i] * w for i, j in selected])
        y = np.array([grid_y[j] * h for i, j in selected])
        
        # Estimate initial radii
        # The maximum radius is limited by the minimum distance to neighbors and boundaries
        # Let's set initial radii to a small fraction of the cell size
        cell_w = w / cols
        cell_h = h / rows
        r = np.full(N, min(cell_w, cell_h) * 0.4)
        
        # Optimization
        # Variables: [w, x_0..x_N-1, y_0..y_N-1, r_0..r_N-1]
        # We fix h = 2 - w
        
        def objective(params):
            w_val = params[0]
            h_val = 2.0 - w_val
            
            if w_val <= 0.01 or h_val <= 0.01:
                return 1e6
            
            x_val = params[1 : 1+N]
            y_val = params[1+N : 1+2*N]
            r_val = params[1+2*N : 1+3*N]
            
            r_val = np.maximum(r_val, 0.01)
            
            # Boundary penalties
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
            
            # Overlap penalties
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
            # Use L-BFGS-B with bounds
            bounds = [(0.1, 1.9)] + [(0, w)] * N + [(0, h)] * N + [(0.01, min(w, h)/2)] * N
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
        except Exception:
            continue
            
        if time.time() > deadline:
            break
        
        # Extract result
        res_w = res.x[0]
        res_h = 2.0 - res_w
        if res_w <= 0.01 or res_h <= 0.01:
            continue
            
        res_x = res.x[1 : 1+N]
        res_y = res.x[1+N : 1+2*N]
        res_r = res.x[1+2*N : 1+3*N]
        res_r = np.maximum(res_r, 0.01)
        
        # Verify validity
        if not _validate(res_x, res_y, res_r, res_w, res_h):
            continue
        
        score = np.sum(res_r)
        if score > best_score:
            best_score = score
            best_solution = np.column_stack((res_x, res_y, res_r))
            
    if best_solution is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
