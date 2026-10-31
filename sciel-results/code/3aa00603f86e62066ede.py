"""Circle packing in perimeter-4 rectangle: Multi-start SLSQP with explicit constraints."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _hex_init(W, H):
    """Generate a hexagonal packing initialization."""
    pts = []
    # Estimate radius
    area = W * H
    # Hex packing density is pi / (2*sqrt(3))
    # N * pi * r^2 = density * area
    # r = sqrt(density * area / (N * pi))
    r_est = np.sqrt((np.pi / (2 * np.sqrt(3))) * area / (N * np.pi)) * 0.95
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    rows = int(np.ceil(H / dy)) + 2
    cols = int(np.ceil(W / dx)) + 2
    
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= N:
                break
            x = j * dx
            y = i * dy
            if i % 2 == 1:
                x += dx / 2
            # Center the grid roughly
            x += (W - (cols - 1) * dx) / 2
            y += (H - (rows - 1) * dy) / 2
            
            # Clamp to box
            x = np.clip(x, r_est, W - r_est)
            y = np.clip(y, r_est, H - r_est)
            pts.append([x, y, r_est])
            
    # If not enough, add random
    while len(pts) < N:
        x = random.uniform(0.1*W, 0.9*W)
        y = random.uniform(0.1*H, 0.9*H)
        r = min(x, y, W-x, H-y) * 0.5
        pts.append([x, y, max(r, 1e-4)])
        
    return np.array(pts[:N])

def _random_init(W, H):
    """Generate a random initialization."""
    pts = []
    for _ in range(N):
        x = random.uniform(0.1*W, 0.9*W)
        y = random.uniform(0.1*H, 0.9*H)
        r = min(x, y, W-x, H-y) * 0.4
        pts.append([x, y, max(r, 1e-4)])
    return np.array(pts)

def _grid_init(W, H):
    """Generate a square grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            x = (i + 0.5) * W / side
            y = (j + 0.5) * H / side
            r = min(x, y, W-x, H-y) * 0.95
            pts.append([x, y, max(r, 1e-4)])
    return np.array(pts[:N])

def _constraints(x, W, H):
    """
    Returns a list of constraint functions.
    Each function returns >= 0 for feasibility.
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    cons = []
    
    # Boundary constraints
    for i in range(N):
        cons.append(lambda x, i=i: x[3*i] - x[3*i+2])       # x - r >= 0
        cons.append(lambda x, i=i: x[3*i+1] - x[3*i+2])     # y - r >= 0
        cons.append(lambda x, i=i: W - x[3*i] - x[3*i+2])   # W - x - r >= 0
        cons.append(lambda x, i=i: H - x[3*i+1] - x[3*i+2]) # H - y - r >= 0
    
    # Non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
            cons.append(lambda x, i=i, j=j: (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2)
            
    return cons

def _objective(x, W, H):
    """Maximize sum of radii -> Minimize negative sum."""
    r = x[2::3]
    return -np.sum(r)

def _objective_grad(x, W, H):
    """Gradient of the objective function."""
    grad = np.zeros_like(x)
    grad[2::3] = -1.0
    return grad

def _is_valid(x, W, H, tol=1e-6):
    """Check if a solution is valid."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    if np.any(r < -tol):
        return False
        
    # Boundary
    if np.any(cx - r < -tol) or np.any(cy - r < -tol) or np.any(W - cx - r < -tol) or np.any(H - cy - r < -tol):
        return False
        
    # Overlap
    for i in range(N):
        for j in range(i+1, N):
            dist2 = (cx[i] - cx[j])**2 + (cy[i] - cy[j])**2
            req2 = (r[i] + r[j])**2
            if dist2 < req2 - tol:
                return False
    return True

def _sum_radii(x):
    return np.sum(x[2::3])

def _optimize(x0, W, H, max_iter=100, ftol=1e-8):
    """Run SLSQP optimization."""
    if _time_left() < 5.0:
        return x0, _sum_radii(x0)
        
    cons = _constraints(x0, W, H)
    
    try:
        res = minimize(
            _objective, 
            x0, 
            jac=_objective_grad, 
            args=(W, H),
            method='SLSQP', 
            bounds=[(1e-6, None)] * (3*N), # r > 0, x,y unbounded (handled by cons)
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
        )
        
        if res.success or _is_valid(res.x, W, H):
             # Ensure radii are positive
             res.x[2::3] = np.maximum(res.x[2::3], 1e-6)
             return res.x, _sum_radii(res.x)
        else:
             # If failed, try to repair or return original
             return x0, _sum_radii(x0)
    except Exception:
        return x0, _sum_radii(x0)

def construct():
    best_x = None
    best_score = -np.inf
    
    # Aspect ratios to try
    # W + H = 2
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 1.4, 0.71, 1.1, 0.9]
    
    # Phase 1: Coarse search
    candidates = []
    
    for W in aspect_ratios:
        if _time_left() < 10.0:
            break
        H = 2.0 - W
        if W <= 0 or H <= 0:
            continue
            
        # Generate multiple initializations
        inits = [
            _hex_init(W, H),
            _grid_init(W, H),
            _random_init(W, H),
            _random_init(W, H)
        ]
        
        for x0 in inits:
            if _time_left() < 5.0:
                break
                
            # Quick optimization
            x_res, score = _optimize(x0, W, H, max_iter=50, ftol=1e-6)
            
            if _is_valid(x_res, W, H):
                candidates.append((score, x_res, W, H))
    
    if not candidates:
        # Fallback: Simple grid
        W, H = 1.0, 1.0
        side = 5
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= N: break
                x = (i+0.5)*W/side
                y = (j+0.5)*H/side
                r = min(x,y,W-x,H-y)*0.9
                pts.append([x,y,r])
        return np.array(pts[:N])
        
    # Sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Fine refinement on top candidates
    top_k = min(5, len(candidates))
    
    for i in range(top_k):
        if _time_left() < 10.0:
            break
        score, x_res, W, H = candidates[i]
        
        # Refine
        x_final, score_final = _optimize(x_res, W, H, max_iter=200, ftol=1e-10)
        
        if _is_valid(x_final, W, H) and score_final > best_score:
            best_score = score_final
            best_x = x_final
            best_W = W
            best_H = H

    if best_x is None:
        # Use the best candidate from phase 1
        score, x_res, W, H = candidates[0]
        best_x = x_res
        best_W = W
        best_H = H

    # Final cleanup: ensure strict validity
    cx = best_x[0::3]
    cy = best_x[1::3]
    r = best_x[2::3]
    
    # Clamp radii to be positive
    r = np.maximum(r, 1e-6)
    
    # Clamp positions to be within bounds considering radius
    cx = np.clip(cx, r, best_W - r)
    cy = np.clip(cy, r, best_H - r)
    
    # Note: Clamping might cause overlaps. 
    # A simple repair step: iteratively shrink radii if overlap detected?
    # For safety, if clamping caused issues, we might need to shrink.
    # However, SLSQP should have provided a feasible point. 
    # The clamping is just to handle numerical noise.
    
    # Let's do a final check and shrink if necessary
    for _ in range(10):
        valid = True
        for i in range(N):
            if cx[i] < r[i] or cy[i] < r[i] or cx[i] > best_W - r[i] or cy[i] > best_H - r[i]:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                    if dist < r[i] + r[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
        # If invalid, shrink all radii slightly
        r *= 0.95
        
    result = np.column_stack([cx, cy, r])
    return result

