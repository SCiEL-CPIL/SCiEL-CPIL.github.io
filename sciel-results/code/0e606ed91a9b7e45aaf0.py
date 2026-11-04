"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of their radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-6):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _make_hexagonal_initial(w, h, rows=None):
    """
    Create a hexagonal-like arrangement for 21 circles.
    """
    if rows is None:
        rows = [5, 4, 5, 4, 3]
    n_rows = len(rows)
    circles = []
    
    r_est = np.sqrt(w * h / (42 * np.sqrt(3)))
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / n_rows
        for col_j in range(count):
            x = (col_j + 0.5) * w / count
            r = r_est
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _make_grid_initial(w, h, rows=None, cols=None):
    """
    Create a grid arrangement for 21 circles.
    """
    if rows is None or cols is None:
        best_r, best_c = 3, 7
        best_diff = abs(w/h - 7/3)
        for r in range(1, 22):
            for c in range(1, 22):
                if r * c >= 21:
                    diff = abs(w/h - c/r)
                    if diff < best_diff:
                        best_diff = diff
                        best_r, best_c = r, c
        rows, cols = best_r, best_c
        
    circles = []
    r_est = min(w / cols, h / rows) / 2
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            circles.append([x, y, r_est])
            
    return np.array(circles[:N])

def _relax_circles_safe(circles, w, h, iterations=1000, tol=1e-9):
    """
    Relaxation to ensure validity.
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for it in range(iterations):
        moved = False
        
        for i in range(n):
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] > w - rad[i]:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] > h - rad[i]:
                pos[i, 1] = h - rad[i]
                moved = True
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True
        
        if not moved:
            break
    
    new_circles = np.column_stack((pos, rad))
    
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    return circles

def _objective_and_grad(params, W, H, beta, eps=1e-4):
    """
    Objective and analytical gradient for L-BFGS-B.
    params: [x0..xN, y0..yN, r0..rN]
    """
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2*n:] = -1.0
    
    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # Boundary penalties
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += beta * np.sum(smax ** 2)
    grad[:n] += -2.0 * beta * smax * gsmax
    grad[2*n:] += 2.0 * beta * smax * gsmax

    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += beta * np.sum(smax ** 2)
    grad[n:2*n] += -2.0 * beta * smax * gsmax
    grad[2*n:] += 2.0 * beta * smax * gsmax

    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += beta * np.sum(smax ** 2)
    grad[:n] += 2.0 * beta * smax * gsmax
    grad[2*n:] += 2.0 * beta * smax * gsmax

    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += beta * np.sum(smax ** 2)
    grad[n:2*n] += 2.0 * beta * smax * gsmax
    grad[2*n:] += 2.0 * beta * smax * gsmax

    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += beta * np.sum(smax ** 2)
    grad[2*n:] += -2.0 * beta * smax * gsmax

    # Overlap penalties
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)

    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]

    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
    obj += beta * np.sum(smax_vec ** 2)

    factor_vec = 2.0 * beta * smax_vec * gsmax_vec
    nx_ij = -dx_ij / dist_ij
    ny_ij = -dy_ij / dist_ij

    grad_x_i = factor_vec * nx_ij
    grad_y_i = factor_vec * ny_ij
    grad_x_j = -factor_vec * nx_ij
    grad_y_j = -factor_vec * ny_ij

    grad[iu] += grad_x_i
    grad[n + iu] += grad_y_i
    grad[2*n + iu] += factor_vec
    grad[ju] += grad_x_j
    grad[n + ju] += grad_y_j
    grad[2*n + ju] += factor_vec

    return obj, grad

def _solve_penalty(circles, w, h, deadline, initial_beta=100.0):
    """
    Optimize using L-BFGS-B with penalty method and analytical gradients.
    """
    if time.time() > deadline - 10.0:
        return None

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))
        
    best_circles = circles.copy()
    best_val = _total_radius(circles)
    
    betas = [initial_beta, initial_beta * 5, initial_beta * 25, initial_beta * 125]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(w, h, beta),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_x = res.x
        except Exception:
            continue
            
        new_circles = np.column_stack((current_x[:n], current_x[n:2*n], current_x[2*n:]))
        
        new_circles = _relax_circles_safe(new_circles, w, h, iterations=200)
        
        if _validate(new_circles, w, h, tol=1e-5):
            val = _total_radius(new_circles)
            if val > best_val:
                best_val = val
                best_circles = new_circles.copy()
                current_x = best_circles.flatten()
            else:
                current_x = best_circles.flatten()
                
    return best_circles

def _try_aspect(ar, deadline, rows=None, use_grid=False):
    """Try a single aspect ratio and return best result."""
    w = ar / (1 + ar) * 2
    h = 1 / (1 + ar) * 2
    
    best_val = 0.0
    best_circles = None
    
    for (ww, hh) in [(w, h), (h, w)]:
        if time.time() > deadline - 10.0:
            break
        
        try:
            if use_grid:
                circles = _make_grid_initial(ww, hh)
            else:
                circles = _make_hexagonal_initial(ww, hh, rows)
            
            circles = _relax_circles_safe(circles, ww, hh, iterations=500)
            
            if not _validate(circles, ww, hh):
                continue
            
            val = _total_radius(circles)
            
            if time.time() < deadline - 15.0:
                opt = _solve_penalty(circles, ww, hh, deadline)
                if opt is not None:
                    circles = opt
                    val = _total_radius(circles)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
                
        except Exception:
            continue
    
    return best_circles, best_val

def _search_aspect_ratio(circles, w, h, deadline):
    """
    Search for better aspect ratio using Nelder-Mead on log(w) and log(h) 
    while maintaining perimeter constraint 2(w+h)=4 => w+h=2.
    We optimize w (h = 2-w).
    """
    if time.time() > deadline - 15.0:
        return circles, w, h, _total_radius(circles)
    
    best_val = _total_radius(circles)
    best_circles = circles.copy()
    best_w = w
    best_h = h
    
    def aspect_obj(w_new):
        h_new = 2.0 - w_new
        if h_new <= 0 or w_new <= 0:
            return 1e6
        
        # Scale circles to new rectangle
        scale_x = w_new / w
        scale_y = h_new / h
        
        new_circles = circles.copy()
        new_circles[:, 0] *= scale_x
        new_circles[:, 1] *= scale_y
        new_circles[:, 2] *= min(scale_x, scale_y) # Scale radii by min to ensure fit? 
        # Actually, scaling radii by min is safe but might be suboptimal. 
        # Let's just scale radii by the geometric mean or just re-relax.
        # Better: scale radii by min(scale_x, scale_y) to keep them inside initially.
        
        # Relax to fit new boundaries
        new_circles = _relax_circles_safe(new_circles, w_new, h_new, iterations=200)
        
        if not _validate(new_circles, w_new, h_new):
            return 1e6
            
        # Quick local optimization
        if time.time() > deadline - 10.0:
            return -_total_radius(new_circles)
            
        opt = _solve_penalty(new_circles, w_new, h_new, deadline, initial_beta=50.0)
        if opt is None:
            return -_total_radius(new_circles)
            
        return -_total_radius(opt)

    # Nelder-Mead search for w in [0.5, 1.5] (since w+h=2, w in [0.5, 1.5] covers square and reasonable aspect ratios)
    # Start at current w
    x0 = np.array([w])
    
    try:
        res = minimize(
            aspect_obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        w_opt = res.x[0]
        h_opt = 2.0 - w_opt
        
        if w_opt > 0.1 and h_opt > 0.1:
            # Re-run with optimized aspect ratio to get the actual best circles
            # We need to re-generate or scale the best circles found during search.
            # The aspect_obj returned the value, but we need the circles.
            # Let's just re-run _try_aspect with the new aspect ratio?
            # No, that's expensive. 
            # Let's assume the best circles from the last successful call in aspect_obj are good.
            # But aspect_obj is a closure, we can't easily get the circles out.
            
            # Alternative: Just run _try_aspect with the new aspect ratio.
            ar_new = w_opt / h_opt
            new_circles, new_val = _try_aspect(ar_new, deadline, rows=[5, 4, 5, 4, 3])
            
            if new_circles is not None and new_val > best_val:
                return new_circles, w_opt, h_opt, new_val
                
    except Exception:
        pass
        
    return best_circles, best_w, best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Targeted aspect ratios
    ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.8, 1.4, 0.75]
    
    row_configs = [
        [5, 4, 5, 4, 3],
        [6, 5, 5, 5],
        [7, 7, 7]
    ]
    
    for ar in ratios:
        if time.time() > deadline - 15.0:
            break
        
        for rows in row_configs:
            if time.time() > deadline - 15.0:
                break
            circles, val = _try_aspect(ar, deadline, rows, use_grid=False)
            if circles is not None and val > best_value:
                best_value = val
                best_result = circles
                
        if ar in [1.0, 1.2, 0.8]:
             if time.time() > deadline - 15.0:
                 break
             circles, val = _try_aspect(ar, deadline, use_grid=True)
             if circles is not None and val > best_value:
                 best_value = val
                 best_result = circles

    # Aspect ratio refinement on the best result found so far
    if best_result is not None and time.time() < deadline - 30.0:
        # We need w and h. We can infer from the aspect ratio used or just try a few.
        # Let's assume the best result came from one of the ratios.
        # We can try to refine the aspect ratio of the best result.
        # To do this, we need to know which w/h was used. 
        # Since _try_aspect returns circles but not w/h, we can just pick a representative w/h 
        # based on the best ratio or just try refining around 1.0.
        
        # Let's pick the ratio that gave the best result. 
        # We didn't track it. Let's just try refining around the best few ratios.
        # Actually, let's just run _search_aspect_ratio on the best result with a guessed w/h.
        # Since perimeter is 4, if we assume aspect ratio 1, w=h=1.
        # But the best might be non-square.
        
        # Let's try refining the aspect ratio for the best few configurations.
        # We'll just pick the best result and try to optimize its aspect ratio.
        # We need to recover w and h. 
        # We can estimate w and h by looking at the bounding box of the circles?
        # No, the circles are inside the rectangle, but might not touch all sides.
        # However, the optimization pushes them to the boundaries.
        # Let's assume the rectangle is tight around the circles.
        # x_min = min(x-r), x_max = max(x+r), etc.
        
        x_min = np.min(best_result[:, 0] - best_result[:, 2])
        x_max = np.max(best_result[:, 0] + best_result[:, 2])
        y_min = np.min(best_result[:, 1] - best_result[:, 2])
        y_max = np.max(best_result[:, 1] + best_result[:, 2])
        
        w_est = x_max - x_min
        h_est = y_max - y_min
        
        # If the rectangle is not tight, this is an underestimate.
        # But for packing, it's usually tight.
        # Let's use the perimeter constraint to scale if needed.
        # If w_est + h_est != 2, we can scale.
        # But _search_aspect_ratio assumes w+h=2.
        
        # Let's just try refining around the aspect ratio of the estimated box.
        ar_est = w_est / h_est if h_est > 0 else 1.0
        
        new_circles, w_new, h_new, val_new = _search_aspect_ratio(best_result, w_est, h_est, deadline)
        
        if new_circles is not None and val_new > best_value:
            best_value = val_new
            best_result = new_circles

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Best Value: {_total_radius(res)}")
