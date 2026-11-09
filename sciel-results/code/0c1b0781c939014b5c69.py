import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _smooth_objective(params, n=10, tol=1e-7, beta=100.0):
    """Smooth objective using soft penalties for better gradient information"""
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    # Clip sides to positive values for smoothness
    sides_clipped = np.maximum(sides, 1e-6)
    
    pen = 0.0
    
    # Boundary penalties (smooth)
    for i in range(n):
        cx, cy = centers[i]
        s = sides_clipped[i]
        # Left boundary violation
        v_left = -tol - (cx - s/2)
        if v_left > 0:
            pen += beta * v_left * v_left
        # Right boundary violation
        v_right = (cx + s/2) - (1 + tol)
        if v_right > 0:
            pen += beta * v_right * v_right
        # Bottom boundary violation
        v_bottom = -tol - (cy - s/2)
        if v_bottom > 0:
            pen += beta * v_bottom * v_bottom
        # Top boundary violation
        v_top = (cy + s/2) - (1 + tol)
        if v_top > 0:
            pen += beta * v_top * v_top
    
    # Overlap penalties (smooth using log-sum-exp style)
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides_clipped[i] + sides_clipped[j]) / 2.0 - tol
            
            # Soft overlap penalty
            vx = np.maximum(0, t - abs(dx))
            vy = np.maximum(0, t - abs(dy))
            # Use smooth approximation of min(vx, vy) or product
            pen += beta * vx * vy
    
    return -np.sum(sides_clipped) + pen

def _objective(params, n=10, tol=1e-7):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += 1e6 * (1.0 - s)
            continue
        if cx - s/2 < -tol:
            pen += 1e6 * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += 1e6 * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += 1e6 * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += 1e6 * ((cy + s/2) - (1 + tol))
            
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0 - tol
            vx = t - abs(dx)
            vy = t - abs(dy)
            sx = np.maximum(0, vx)
            sy = np.maximum(0, vy)
            pen += 1e5 * (sx * sy)
            
    return -np.sum(sides) + pen

def _polish(centers, sides, n=10, tol=1e-7, max_iter=500):
    """High-precision polish pass with tighter tolerances"""
    params = np.concatenate([centers.flatten(), sides])
    
    # Use L-BFGS-B with bounds for better convergence
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0))  # sides
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
    
    # Reorder: [cx0, cy0, ..., cx9, cy9, s0, ..., s9]
    # Current: [cx0, cy0, ..., cx9, cy9, s0, ..., s9] - already correct
    
    try:
        res = minimize(_objective, params, args=(n,), method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        return res.x
    except Exception:
        return params

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # Construct the specific 3.0 configuration
    # 4 Large (0.375) at corners
    # 6 Small (0.25) in center strip and side gaps
    
    s_large = 0.375
    s_small = 0.25
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 4 Large
    # 0: BL
    centers[0] = [s_large/2, s_large/2]
    sides[0] = s_large
    # 1: BR
    centers[1] = [1 - s_large/2, s_large/2]
    sides[1] = s_large
    # 2: TL
    centers[2] = [s_large/2, 1 - s_large/2]
    sides[2] = s_large
    # 3: TR
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[3] = s_large
    
    # 4 Small in central strip (x: 0.375 to 0.625, width 0.25)
    # y positions: 0.125, 0.375, 0.625, 0.875
    centers[4] = [0.5, 0.125]
    sides[4] = s_small
    centers[5] = [0.5, 0.375]
    sides[5] = s_small
    centers[6] = [0.5, 0.625]
    sides[6] = s_small
    centers[7] = [0.5, 0.875]
    sides[7] = s_small
    
    # 2 Small in side gaps
    # Left gap: x [0, 0.375], y [0.375, 0.625]
    # Center x = 0.1875, Center y = 0.5
    centers[8] = [0.1875, 0.5]
    sides[8] = s_small
    
    # Right gap: x [0.625, 1.0], y [0.375, 0.625]
    # Center x = 0.8125, Center y = 0.5
    centers[9] = [0.8125, 0.5]
    sides[9] = s_small
    
    update_best(centers, sides)
    
    # Try multiple optimization strategies from different starting points
    # Strategy 1: Powell from initial config
    if time.time() < deadline - 60:
        params_init = np.concatenate([centers.flatten(), sides])
        
        try:
            res = minimize(_objective, params_init, args=(n,), method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Strategy 2: Smooth objective with Nelder-Mead
    if time.time() < deadline - 60:
        params_init = np.concatenate([centers.flatten(), sides])
        
        try:
            res = minimize(_smooth_objective, params_init, args=(n,), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Strategy 3: Try different initial configurations
    # Configuration A: Larger corner squares
    if time.time() < deadline - 60:
        s_large_a = 0.4
        s_small_a = 0.2333  # (1 - 0.4) / 2.55 approximately
        
        centers_a = np.zeros((n, 2))
        sides_a = np.zeros(n)
        
        # 4 Large at corners
        centers_a[0] = [s_large_a/2, s_large_a/2]
        sides_a[0] = s_large_a
        centers_a[1] = [1 - s_large_a/2, s_large_a/2]
        sides_a[1] = s_large_a
        centers_a[2] = [s_large_a/2, 1 - s_large_a/2]
        sides_a[2] = s_large_a
        centers_a[3] = [1 - s_large_a/2, 1 - s_large_a/2]
        sides_a[3] = s_large_a
        
        # 6 Small in remaining space
        # Central vertical strip
        centers_a[4] = [0.5, 0.2]
        sides_a[4] = s_small_a
        centers_a[5] = [0.5, 0.5]
        sides_a[5] = s_small_a
        centers_a[6] = [0.5, 0.8]
        sides_a[6] = s_small_a
        
        # Side gaps
        centers_a[7] = [0.2, 0.35]
        sides_a[7] = s_small_a
        centers_a[8] = [0.8, 0.35]
        sides_a[8] = s_small_a
        centers_a[9] = [0.5, 0.5]  # This will overlap, let optimizer fix
        sides_a[9] = s_small_a
        
        update_best(centers_a, sides_a)
        
        if time.time() < deadline - 30:
            params_init_a = np.concatenate([centers_a.flatten(), sides_a])
            try:
                res = minimize(_objective, params_init_a, args=(n,), method='Powell',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                update_best(c_opt, s_opt)
            except Exception:
                pass
    
    # Strategy 4: Configuration with 5 medium + 5 small
    if time.time() < deadline - 60:
        s_med = 0.3
        s_sml = 0.2
        
        centers_b = np.zeros((n, 2))
        sides_b = np.zeros(n)
        
        # 5 Medium squares in a cross pattern
        centers_b[0] = [0.5, 0.15]
        sides_b[0] = s_med
        centers_b[1] = [0.15, 0.5]
        sides_b[1] = s_med
        centers_b[2] = [0.85, 0.5]
        sides_b[2] = s_med
        centers_b[3] = [0.5, 0.85]
        sides_b[3] = s_med
        centers_b[4] = [0.5, 0.5]
        sides_b[4] = s_med
        
        # 5 Small in corners
        centers_b[5] = [0.1, 0.1]
        sides_b[5] = s_sml
        centers_b[6] = [0.9, 0.1]
        sides_b[6] = s_sml
        centers_b[7] = [0.1, 0.9]
        sides_b[7] = s_sml
        centers_b[8] = [0.9, 0.9]
        sides_b[8] = s_sml
        centers_b[9] = [0.5, 0.5]  # Overlaps, optimizer will fix
        sides_b[9] = s_sml
        
        update_best(centers_b, sides_b)
        
        if time.time() < deadline - 30:
            params_init_b = np.concatenate([centers_b.flatten(), sides_b])
            try:
                res = minimize(_objective, params_init_b, args=(n,), method='Powell',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                update_best(c_opt, s_opt)
            except Exception:
                pass
    
    # Final polish on the best solution found
    if best_centers is not None and time.time() < deadline - 10:
        polished_params = _polish(best_centers, best_sides, n, tol=1e-7, max_iter=300)
        c_polished = polished_params[:2*n].reshape(n, 2)
        s_polished = np.maximum(polished_params[2*n:], 1e-6)
        update_best(c_polished, s_polished)
    
    if best_centers is None:
        # Fallback to the static configuration if something went wrong
        best_centers = centers
        best_sides = sides

    return best_centers, best_sides
