import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _generate_hex_init(w, h, n_circles):
    """
    Generate initial circle positions in a hexagonal pattern.
    """
    circles = np.zeros((n_circles, 3))
    # Estimate radius from area
    r_est = np.sqrt(w * h / (n_circles * np.pi)) * 0.95
    # Hex spacing
    dx = 2.0 * r_est
    dy = np.sqrt(3) * r_est
    
    count = 0
    row = 0
    while count < n_circles:
        offset = (dx / 2) if (row % 2 == 1) else 0
        col = 0
        while count < n_circles:
            x = r_est + offset + col * dx
            y = r_est + row * dy
            if x + r_est <= w and y + r_est <= h:
                circles[count, 0] = x
                circles[count, 1] = y
                circles[count, 2] = r_est
                count += 1
            else:
                break
            col += 1
        row += 1
        if r_est + row * dy > h + r_est:
            break
    
    # Fallback for any remaining circles
    if count < n_circles:
        for i in range(count, n_circles):
            circles[i, 0] = w / 2
            circles[i, 1] = h / 2
            circles[i, 2] = r_est * 0.8
    
    return circles

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
            # Initial radius: half the min cell size, slightly smaller
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    return circles

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
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

def _resolve_collisions(circles, w, h, max_iters=5000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints
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
        
        # 2. Pairwise collisions
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = rad[i] + rad[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
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
            
    # Final clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _objective_and_grad(params, W, H, penalty_factor):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    # Smooth penalty function: soft_max(0, v)^2
    # v = violation
    eps = 1e-6
    boundary_weight = penalty_factor
    overlap_weight = penalty_factor

    def _smooth_max_grad(v):
        # soft_max(0, v) = 0.5 * (v + sqrt(v^2 + eps^2))
        # This is a smooth approximation of max(0, v)
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        # d/dv soft_max(0, v) = 0.5 * (1 + v / sqrt(v^2 + eps^2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # Boundary penalties
    # 1. x - r >= 0 => violation = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # 2. y - r >= 0 => violation = r - y
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # 3. x + r <= W => violation = x + r - W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # 4. y + r <= H => violation = y + r - H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # 5. r >= 0 => violation = -r
    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties
    # dist >= r_i + r_j => violation = (r_i + r_j) - dist
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
    obj += overlap_weight * np.sum(smax_vec ** 2)
    
    # Gradient of penalty w.r.t. x_i, y_i, r_i
    # d/dx_i penalty = 2 * w * smax * gsmax * d/dx_i (r_sum - dist)
    # d/dx_i dist = (x_i - x_j) / dist = -dx_ij / dist_ij
    # So d/dx_i violation = -(-dx_ij / dist_ij) = dx_ij / dist_ij
    # Wait, v = r_sum - dist. d(v)/d(x_i) = - d(dist)/d(x_i) = -(x_i - x_j)/dist = -dx_ij/dist.
    # Let's re-verify.
    # dist = sqrt((x_i-x_j)^2 + ...). d(dist)/d(x_i) = (x_i-x_j)/dist.
    # v = r_sum - dist. d(v)/d(x_i) = - (x_i-x_j)/dist.
    # dx_ij = x_i - x_j. So d(v)/d(x_i) = -dx_ij / dist.
    
    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
    # d/dx_i: factor * (-dx_ij / dist_ij)
    # d/dy_i: factor * (-dy_ij / dist_ij)
    # d/dr_i: factor * 1
    # d/dx_j: factor * (dx_ij / dist_ij)  (since d(v)/d(x_j) = -d(dist)/d(x_j) = -(-(x_i-x_j)/dist) = dx_ij/dist)
    # d/dy_j: factor * (dy_ij / dist_ij)
    # d/dr_j: factor * 1
    
    grad_x_i = factor_vec * (-dx_ij / dist_ij)
    grad_y_i = factor_vec * (-dy_ij / dist_ij)
    grad_x_j = factor_vec * (dx_ij / dist_ij)
    grad_y_j = factor_vec * (dy_ij / dist_ij)
    
    np.add.at(grad, 3 * iu, grad_x_i)
    np.add.at(grad, 3 * iu + 1, grad_y_i)
    np.add.at(grad, 3 * iu + 2, factor_vec)
    np.add.at(grad, 3 * ju, grad_x_j)
    np.add.at(grad, 3 * ju + 1, grad_y_j)
    np.add.at(grad, 3 * ju + 2, factor_vec)

    return obj, grad

def _local_optimize(circles, W, H, deadline, penalty_factor=100.0):
    """
    Use L-BFGS-B with analytical gradients to locally optimize.
    """
    if time.time() > deadline - 2.0:
        return circles

    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    try:
        result = minimize(
            _objective_and_grad, params, args=(W, H, penalty_factor),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200,
                     'callback': lambda x: time.time() > deadline - 1.0}
        )
        new_circles = result.x.reshape(N, 3)
        
        # Validate
        if _validate_solution(new_circles, W, H, N):
            return new_circles
        else:
            # Try to fix
            fixed = _resolve_collisions(new_circles, W, H, max_iters=2000)
            if _validate_solution(fixed, W, H, N):
                return fixed
    except Exception:
        pass
    return circles

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    """
    best_init = None
    best_init_score = 0.0
    
    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6), (5, 4)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4), (4, 5)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (3, 7)]
        
    # Try grid inits
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=500)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed
    
    # Try hex init
    if time.time() < deadline - 5:
        hex_circles = _generate_hex_init(w, h, N)
        hex_relaxed = _resolve_collisions(hex_circles, w, h, max_iters=500)
        if _validate_solution(hex_relaxed, w, h, N):
            score = np.sum(hex_relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = hex_relaxed

    if best_init is None:
        # Fallback
        best_init = _generate_grid_init(w, h, N, 5, 5)
        best_init = _resolve_collisions(best_init, w, h, max_iters=500)
        if not _validate_solution(best_init, w, h, N):
            return None

    # Optimize with increasing penalty
    current = best_init
    current_score = np.sum(current[:, 2])
    
    # Pass 1: Lower penalty to find feasible region
    if time.time() < deadline - 2:
        opt1 = _local_optimize(current, w, h, deadline, penalty_factor=50.0)
        if _validate_solution(opt1, w, h, N) and np.sum(opt1[:, 2]) > current_score - 1e-6:
            current = opt1
            current_score = np.sum(current[:, 2])
            
    # Pass 2: Higher penalty for tighter constraints
    if time.time() < deadline - 2:
        opt2 = _local_optimize(current, w, h, deadline, penalty_factor=500.0)
        if _validate_solution(opt2, w, h, N) and np.sum(opt2[:, 2]) > current_score - 1e-6:
            current = opt2
            current_score = np.sum(current[:, 2])
            
    # Pass 3: Very high penalty for final polish
    if time.time() < deadline - 2:
        opt3 = _local_optimize(current, w, h, deadline, penalty_factor=2000.0)
        if _validate_solution(opt3, w, h, N) and np.sum(opt3[:, 2]) > current_score - 1e-6:
            current = opt3
            current_score = np.sum(current[:, 2])
            
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    # Perimeter 4 => w + h = 2
    # We test a range of widths
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15, 0.75, 1.25, 0.7, 1.3, 0.6, 1.4]
    
    for w in aspect_ratios:
        if time.time() > deadline - 5.0:
            break
            
        h = 2.0 - w
        
        # Time check
        if deadline - time.time() < 15.0:
            # If little time left, only try the most promising aspect (square) if not done
            if w != 1.0:
                continue
            elif best_result is not None:
                break

        try:
            result = _solve_aspect(w, h, deadline)
            if result is not None:
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
        except Exception:
            pass

    if best_result is None:
        # Fallback: Simple grid
        w, h = 1.0, 1.0
        positions = []
        radii = []
        cols = 5
        rows = 5
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.9
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
