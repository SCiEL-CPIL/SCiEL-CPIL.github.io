"""Circle packing in perimeter-4 rectangle: Vectorized L-BFGS-B with Penalty Method."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # safety margin

def _get_initial_guess(W, H, layout_type='hex'):
    """Generate initial positions and radii."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    if layout_type == 'hex':
        # Hexagonal packing approximation
        # Estimate base radius
        base_r = min(W, H) / 6.0
        # Generate points in a hex grid
        pts = []
        row = 0
        while len(pts) < N:
            y_offset = row * base_r * np.sqrt(3)
            if y_offset > H: break
            col = 0
            offset = base_r * 1.5 if row % 2 else 0
            while len(pts) < N:
                x_offset = col * base_r * 1.5 + offset
                if x_offset > W: break
                pts.append([x_offset, y_offset])
                col += 1
            row += 1
        
        for i in range(N):
            if i < len(pts):
                x[i] = pts[i][0]
                y[i] = pts[i][1]
            else:
                # Fallback random if hex grid is too small
                x[i] = np.random.uniform(0, W)
                y[i] = np.random.uniform(0, H)
            # Initial radius
            r[i] = min(x[i], y[i], W-x[i], H-y[i]) * 0.5
            r[i] = max(r[i], 0.01)
            
    elif layout_type == 'grid':
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * W
            y[i] = (row + 0.5) / side * H
            r[i] = min(W/side, H/side) * 0.4
            r[i] = max(r[i], 0.01)
            
    else: # random
        x = np.random.uniform(0.1, 0.9, N) * W
        y = np.random.uniform(0.1, 0.9, N) * H
        r = np.random.uniform(0.05, 0.2, N) * min(W, H)
        
    return x, y, r

def _objective_and_grad(z, W, H, penalty_weight=100.0):
    """
    z: [x1, y1, r1, ..., xN, yN, rN, W]
    Returns objective value and gradient.
    Objective: -sum(r) + penalty(overlaps) + penalty(walls)
    """
    n = N
    x = z[0:n]
    y = z[n:2*n]
    r = z[2*n:3*n]
    W = z[-1]
    H = 2.0 - W
    
    # Ensure W and H are positive
    if W < 0.5 or H < 0.5:
        return 1e6, np.zeros_like(z)
        
    obj = -np.sum(r)
    grad = np.zeros_like(z)
    
    # 1. Wall Penalties
    # x < r -> (r-x)^2
    # x > W-r -> (x-(W-r))^2
    # y < r -> (r-y)^2
    # y > H-r -> (y-(H-r))^2
    
    # Left/Right walls
    left_viol = np.maximum(0, r - x)
    right_viol = np.maximum(0, x - (W - r))
    
    # Top/Bottom walls
    bottom_viol = np.maximum(0, r - y)
    top_viol = np.maximum(0, y - (H - r))
    
    wall_pen = penalty_weight * (np.sum(left_viol**2) + np.sum(right_viol**2) + 
                                  np.sum(bottom_viol**2) + np.sum(top_viol**2))
    obj += wall_pen
    
    # Gradients for walls
    # d/dx (r-x)^2 = -2(r-x) if x<r
    # d/dx (x-(W-r))^2 = 2(x-(W-r)) if x>W-r
    grad_x_wall = np.zeros(n)
    mask_l = left_viol > 0
    grad_x_wall[mask_l] = -2 * penalty_weight * left_viol[mask_l]
    mask_r = right_viol > 0
    grad_x_wall[mask_r] += 2 * penalty_weight * right_viol[mask_r]
    
    grad_y_wall = np.zeros(n)
    mask_b = bottom_viol > 0
    grad_y_wall[mask_b] = -2 * penalty_weight * bottom_viol[mask_b]
    mask_t = top_viol > 0
    grad_y_wall[mask_t] += 2 * penalty_weight * top_viol[mask_t]
    
    # d/dr for walls
    # d/dr (r-x)^2 = 2(r-x)
    # d/dr (x-(W-r))^2 = -2(x-(W-r)) (since d(W-r)/dr = -1, so d(x-W+r)/dr = 1? No.
    # Let u = x - (W-r) = x - W + r. du/dr = 1. d(u^2)/dr = 2u.
    # Wait. Violation is max(0, x - (W-r)).
    # If x > W-r, term is (x - W + r)^2. d/dr = 2(x - W + r).
    # If x < r, term is (r-x)^2. d/dr = 2(r-x).
    
    grad_r_wall = np.zeros(n)
    grad_r_wall[mask_l] += 2 * penalty_weight * left_viol[mask_l]
    grad_r_wall[mask_r] += 2 * penalty_weight * right_viol[mask_r]
    grad_r_wall[mask_b] += 2 * penalty_weight * bottom_viol[mask_b]
    grad_r_wall[mask_t] += 2 * penalty_weight * top_viol[mask_t]
    
    # 2. Overlap Penalties
    # d = dist(i,j), req = r_i + r_j
    # viol = max(0, req - d)
    # pen = penalty_weight * viol^2
    
    # Vectorized pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0) # Avoid division by zero
    
    req = r[:, None] + r[None, :]
    viol = np.maximum(0, req - d)
    
    # Symmetric, so we can sum upper triangle
    # But for gradient, it's easier to sum all and divide by 2?
    # Let's just sum all i!=j.
    overlap_pen = 0.5 * penalty_weight * np.sum(viol**2) # 0.5 to avoid double counting
    obj += overlap_pen
    
    # Gradients
    # d(viol^2)/dx_i = 2 * viol * d(req-d)/dx_i
    # d(req-d)/dx_i = - dx_i / d (where dx_i = x_i - x_j)
    # Note: dx in matrix is x_i - x_j.
    # d(d)/dx_i = dx / d
    # So d(-d)/dx_i = -dx/d
    # Term: 2 * viol * (-dx/d) * penalty_weight
    # Sum over j
    
    # grad_x_overlap[i] = sum_j 2 * viol[i,j] * (-dx[i,j]/d[i,j]) * penalty_weight
    # We can use matrix multiplication or sum axis
    
    # viol is (N,N). dx is (N,N). d is (N,N).
    # factor = 2 * penalty_weight * viol / d
    # factor[i,j] = 2 * penalty_weight * viol[i,j] / d[i,j]
    # grad_x[i] = sum_j (-factor[i,j] * dx[i,j])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(d > 1e-8, 2 * penalty_weight * viol / d, 0)
        
    # grad_x = - sum(factor * dx, axis=1)
    grad_x_overlap = -np.sum(factor * dx, axis=1)
    grad_y_overlap = -np.sum(factor * dy, axis=1)
    
    # d/dr_i
    # d(req)/dr_i = 1 (for all j)
    # d(viol)/dr_i = 1 if viol>0
    # d(viol^2)/dr_i = 2 * viol
    # Sum over j: 2 * penalty_weight * sum_j viol[i,j]
    grad_r_overlap = 2 * penalty_weight * np.sum(viol, axis=1)
    
    # 3. Gradient for W
    # H = 2 - W
    # Wall penalties depend on W and H.
    # Right wall: (x - (W-r))^2. d/dW = 2(x - W + r) * (-1) = -2 * right_viol
    # Top wall: (y - (H-r))^2. H = 2-W. dH/dW = -1.
    # Term: (y - (2-W-r))^2 = (y - 2 + W + r)^2.
    # d/dW = 2(y - 2 + W + r) * 1 = 2 * top_viol
    
    grad_W = 0
    # From Right Wall
    grad_W += -2 * penalty_weight * np.sum(right_viol)
    # From Top Wall
    grad_W += 2 * penalty_weight * np.sum(top_viol)
    
    # Combine Gradients
    grad[0:n] = grad_x_wall + grad_x_overlap - 1.0 # -1 from -sum(r)
    grad[n:2*n] = grad_y_wall + grad_y_overlap
    grad[2*n:3*n] = grad_r_wall + grad_r_overlap - 1.0
    grad[-1] = grad_W
    
    return obj, grad

def _evaluate(x, y, r, W, H):
    """Return sum of radii if valid, else None."""
    cx = np.clip(x, 0.0, W)
    cy = np.clip(y, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    
    # Check overlaps
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.sqrt(dx**2 + dy**2)
    req = r[:, None] + r[None, :]
    # Check if any d < req - eps
    if np.any(d < req - 1e-7):
        return None
    return float(np.sum(r))

def _optimize_single(W_init, H_init, layout_type, time_limit=10.0):
    """Run L-BFGS-B for a single initial guess."""
    start_time = time.time()
    x, y, r = _get_initial_guess(W_init, H_init, layout_type)
    
    z0 = np.concatenate([x, y, r, [W_init]])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, 2.0)) # x
        bounds.append((0, 2.0)) # y
        bounds.append((1e-6, 2.0)) # r
    bounds.append((0.5, 1.5)) # W (H = 2-W, so H in [0.5, 1.5])
    
    try:
        res = minimize(
            _objective_and_grad,
            z0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
    except Exception:
        return None, 0.0
        
    if time.time() - start_time > time_limit:
        return None, 0.0
        
    z_opt = res.x
    x_opt = z_opt[0:N]
    y_opt = z_opt[N:2*N]
    r_opt = z_opt[2*N:3*N]
    W_opt = z_opt[-1]
    H_opt = 2.0 - W_opt
    
    # Clamp and Validate
    x_opt = np.clip(x_opt, 0, W_opt)
    y_opt = np.clip(y_opt, 0, H_opt)
    r_opt = np.maximum(r_opt, 1e-6)
    lo = np.minimum(np.minimum(x_opt, y_opt), np.minimum(W_opt - x_opt, H_opt - y_opt))
    r_opt = np.minimum(r_opt, lo)
    
    s = _evaluate(x_opt, y_opt, r_opt, W_opt, H_opt)
    if s is not None:
        return np.column_stack([x_opt, y_opt, r_opt]), s
    return None, 0.0

def _polish(cx, cy, r, W, H, time_budget=5.0):
    """Simple force-directed polish to fix small numerical errors."""
    start = time.time()
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    best_sum = _evaluate(np.column_stack([cx, cy, r]), W, H) or 0.0
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = r.copy()
    
    dt = 0.001
    for _ in range(500):
        if time.time() - start > time_budget:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Vectorized forces
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        req = r[:, None] + r[None, :]
        overlap = np.maximum(0, req - d)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            force_factor = np.where(d > 1e-8, overlap / d, 0)
            
        fx += -np.sum(force_factor * dx, axis=1) * 10
        fy += -np.sum(force_factor * dy, axis=1) * 10
        
        # Walls
        fx += np.where(cx < r, (r - cx) * 20, 0)
        fx += np.where(cx > W - r, -(cx - (W - r)) * 20, 0)
        fy += np.where(cy < r, (r - cy) * 20, 0)
        fy += np.where(cy > H - r, -(cy - (H - r)) * 20, 0)
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp
        cx = np.clip(cx, 0, W)
        cy = np.clip(cy, 0, H)
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r, lo)
        r = np.maximum(r, 1e-6)
        
        s_new = _evaluate(np.column_stack([cx, cy, r]), W, H)
        if s_new is not None and s_new > best_sum:
            best_sum = s_new
            best_cx = cx.copy()
            best_cy = cy.copy()
            best_r = r.copy()
            
    return best_cx, best_cy, best_r, best_sum

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE - 3
        
    random.seed(42)
    np.random.seed(42)
    
    # Try different aspect ratios and layouts
    aspects = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    layouts = ['hex', 'grid', 'random']
    
    candidates = []
    
    for W_init in aspects:
        H_init = 2.0 - W_init
        if H_init < 0.5 or H_init > 1.5:
            continue
            
        for layout in layouts:
            if not time_ok():
                break
            res, s = _optimize_single(W_init, H_init, layout, time_limit=15.0)
            if res is not None and s > best_sum:
                best_sum = s
                best_arr = res
                best_W = W_init
                best_H = H_init
                # Store for potential polishing
                candidates.append((s, res, W_init, H_init))
                
    # Sort candidates by score
    candidates.sort(key=lambda x: -x[0])
    
    # Polish the top candidates
    for s, arr, W, H in candidates[:3]:
        if not time_ok():
            break
        cx_p, cy_p, r_p, s_p = _polish(arr[:, 0], arr[:, 1], arr[:, 2], W, H, time_budget=10.0)
        if s_p > best_sum:
            best_sum = s_p
            best_arr = np.column_stack([cx_p, cy_p, r_p])
            best_W = W
            best_H = H
            
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        x, y, r = _get_initial_guess(W, H, 'grid')
        best_arr = np.column_stack([x, y, r])
        best_W, best_H = W, H
        
    # Final clamp
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Return a valid grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return out

