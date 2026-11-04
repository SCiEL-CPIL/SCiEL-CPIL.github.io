import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    pos_r: (n, 3) array of [x, y, r]
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    if np.any(rs < 1e-9):
        return False
    
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _generate_grid_init(w, h, n_circles, cols, rows):
    """
    Generate initial circle positions in a grid pattern.
    """
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
    return circles

def _smooth_penalty(circles, w, h, eps):
    """
    Compute a smooth penalty for constraint violations.
    Uses soft-min to create a differentiable penalty.
    """
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    penalty = 0.0
    
    # Boundary violations: r - dist_to_boundary
    for i in range(n):
        d_left = rs[i] - xs[i]
        d_right = rs[i] - (w - xs[i])
        d_bottom = rs[i] - ys[i]
        d_top = rs[i] - (h - ys[i])
        
        # Soft max of violations (positive = violation)
        vals = np.array([d_left, d_right, d_bottom, d_top])
        m = np.max(vals)
        if m > 0:
            logsumexp = m + np.log(np.sum(np.exp(vals - m)))
            penalty += logsumexp / eps * eps  # scale by eps for smoothness
        else:
            penalty += 0.0  # no violation
    
    # Overlap violations: (r_i + r_j) - dist
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            violation = (rs[i] + rs[j]) - dist
            if violation > 0:
                penalty += violation / eps * eps
    
    return penalty

def _slsqp_optimize(circles, w, h, deadline, maxiter=150):
    """
    Optimize using SLSQP with smooth surrogate objective.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        r = x[2::3]
        # Smooth penalty for constraint violations
        circles_arr = x.reshape(-1, 3)
        
        # Compute penalty with decreasing eps for smoothness
        eps = 0.001
        pen = 0.0
        
        xs = circles_arr[:, 0]
        ys = circles_arr[:, 1]
        rs = circles_arr[:, 2]
        
        # Boundary penalties (soft max)
        for i in range(n):
            d_left = rs[i] - xs[i]
            d_right = rs[i] - (w - xs[i])
            d_bottom = rs[i] - ys[i]
            d_top = rs[i] - (h - ys[i])
            vals = np.array([d_left, d_right, d_bottom, d_top])
            m = np.max(vals)
            if m > 0:
                pen += (m + eps * np.log(np.sum(np.exp((vals - m) / eps)))) / eps
            
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                viol = (rs[i] + rs[j]) - dist
                if viol > 0:
                    pen += viol / eps
        
        return -np.sum(r) + 1000.0 * pen

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )

        if res.success:
            new_circles = res.x.reshape(-1, 3)
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass

    return None

def _polish_pass(circles, w, h, deadline):
    """
    High-precision polish using L-BFGS-B with smooth penalty and progressive eps.
    """
    if time.time() > deadline - 8.0:
        return None
    
    n = len(circles)
    x0 = circles.flatten()
    
    def objective_smooth(x, eps):
        circles_arr = x.reshape(-1, 3)
        xs = circles_arr[:, 0]
        ys = circles_arr[:, 1]
        rs = circles_arr[:, 2]
        
        pen = 0.0
        
        # Boundary penalties
        for i in range(n):
            d_left = rs[i] - xs[i]
            d_right = rs[i] - (w - xs[i])
            d_bottom = rs[i] - ys[i]
            d_top = rs[i] - (h - ys[i])
            vals = np.array([d_left, d_right, d_bottom, d_top])
            m = np.max(vals)
            if m > 0:
                pen += (m + eps * np.log(np.sum(np.exp((vals - m) / eps)))) / eps
        
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                viol = (rs[i] + rs[j]) - dist
                if viol > 0:
                    pen += viol / eps
        
        return -np.sum(rs) + 5000.0 * pen
    
    # Progressive eps schedule
    eps_schedule = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    
    current_x = x0.copy()
    best_x = x0.copy()
    best_score = np.sum(x0[2::3])
    
    for eps in eps_schedule:
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(
                lambda x: objective_smooth(x, eps),
                current_x,
                method='L-BFGS-B',
                bounds=[(1e-6, w), (1e-6, h), (1e-6, min(w,h)/2)] * n,
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            
            if res.success:
                candidate = res.x
                circles_arr = candidate.reshape(-1, 3)
                if _validate_solution(circles_arr, w, h, N):
                    score = np.sum(circles_arr[:, 2])
                    if score > best_score:
                        best_score = score
                        best_x = candidate.copy()
                        current_x = candidate.copy()
        except Exception:
            pass
    
    if best_x is not None and best_score > np.sum(x0[2::3]):
        final_circles = best_x.reshape(-1, 3)
        if _validate_solution(final_circles, w, h, N):
            return final_cicles if False else final_circles
    
    return None

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    """
    best_init = None
    best_init_score = 0.0
    
    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4)]
        
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=100)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed

    if best_init is None:
        best_init = _generate_grid_init(w, h, N, 5, 5)
        best_init = _resolve_collisions(best_init, w, h, max_iters=100)
        if not _validate_solution(best_init, w, h, N):
            return None

    if time.time() < deadline - 10.0:
        optimized = _slsqp_optimize(best_init, w, h, deadline)
        if optimized is not None and _validate_solution(optimized, w, h, N):
            best_init = optimized

    # Final polish pass
    if time.time() < deadline - 5.0:
        polished = _polish_pass(best_init, w, h, deadline)
        if polished is not None and _validate_solution(polished, w, h, N):
            if np.sum(polished[:, 2]) > np.sum(best_init[:, 2]):
                best_init = polished

    return best_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin under 150s limit
    
    best_result = None
    best_score = 0.0
    
    aspect_ratios = [1.0, 1.25, 0.8, 1.5, 0.67, 1.1, 0.9, 1.4, 0.7, 1.6, 0.6, 1.8, 0.5, 2.0, 0.55, 1.3, 0.75]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 12.0:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        result = _solve_aspect(w, h, deadline)
        
        if result is not None and _validate_solution(result, w, h, N):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _generate_grid_init(w, h, N, 5, 5)
        best_result = _resolve_collisions(best_result, w, h, max_iters=100)
        if not _validate_solution(best_result, w, h, N):
            for i in range(N):
                x, y, r = best_result[i]
                min_d = min(x, w-x, y, h-y)
                best_result[i, 2] = min(r, min_d - 1e-4)
            best_result = _resolve_collisions(best_result, w, h, max_iters=100)

    return best_result
