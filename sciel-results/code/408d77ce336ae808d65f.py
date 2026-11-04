"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    This ensures the output is strictly valid.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]

                if dist < min_dist:
                    if dist < 1e-9:
                        # Random push if coincident
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

    # Final boundary clamp to ensure validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    # Final validation check
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    # If relaxation failed (shouldn't happen with small radii), return original
    return circles

def _make_grid(n, w, h):
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.5 * min(w, h) / side * 0.9
            cells.append([cx, cy, r])
    return np.array(cells[:n])

def _make_hexagonal_grid(n, w, h):
    """Create a hexagonal packing initial layout."""
    circles = []
    r = min(w, h) / 6.0 * 0.85
    dx = 2 * r * 0.866
    dy = 2 * r * 0.75
    
    row = 0
    while len(circles) < n:
        offset = 0 if row % 2 == 0 else dx / 2
        col = 0
        while len(circles) < n:
            cx = r + offset + col * dx
            cy = r + row * dy
            if cx + r <= w and cy + r <= h:
                circles.append([cx, cy, r])
            col += 1
        row += 1
    
    if len(circles) < n:
        # Fill remaining with grid
        remaining = n - len(circles)
        for i in range(remaining):
            cx = w * (i + 0.5) / remaining
            cy = h * 0.5
            circles.append([cx, cy, r * 0.5])
    
    return np.array(circles[:n])

def _log_sum_exp_penalty(circles, w, h, eps=0.01):
    """
    Compute smooth penalty using log-sum-exp for overlap and boundary constraints.
    Returns (penalty_value, gradient_wrt_circles).
    """
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    penalty = 0.0
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)
    
    # Overlap penalties: for each pair, penalty if dist < r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            
            # Smooth max(0, min_dist - dist) using log-sum-exp approximation
            z = (np.sqrt(dist_sq + 1e-12) - min_dist) / eps
            if z > 50:
                penalty_term = 0.0
                dp = 0.0
            elif z < -50:
                penalty_term = -z * eps
                dp = 1.0
            else:
                penalty_term = -eps * np.log(1.0 + np.exp(-z))
                dp = 1.0 / (1.0 + np.exp(z))
            
            penalty += penalty_term
            
            if penalty_term > 0:
                dist = np.sqrt(dist_sq + 1e-12)
                if dist > 1e-9:
                    ux = dx / dist
                    uy = dy / dist
                    grad_x[i] += dp * ux
                    grad_x[j] -= dp * ux
                    grad_y[i] += dp * uy
                    grad_y[j] -= dp * uy
                    grad_r[i] += dp
                    grad_r[j] += dp
    
    # Boundary penalties
    for i in range(n):
        # Left boundary: x_i - r_i >= 0
        z_left = (x[i] - r[i]) / eps
        if z_left > 50:
            p_left = 0.0
            dp_left = 0.0
        elif z_left < -50:
            p_left = -z_left * eps
            dp_left = 1.0
        else:
            p_left = -eps * np.log(1.0 + np.exp(-z_left))
            dp_left = 1.0 / (1.0 + np.exp(z_left))
        penalty += p_left
        if p_left > 0:
            grad_x[i] += dp_left
            grad_r[i] -= dp_left
        
        # Right boundary: w - x_i - r_i >= 0
        z_right = (w - x[i] - r[i]) / eps
        if z_right > 50:
            p_right = 0.0
            dp_right = 0.0
        elif z_right < -50:
            p_right = -z_right * eps
            dp_right = 1.0
        else:
            p_right = -eps * np.log(1.0 + np.exp(-z_right))
            dp_right = 1.0 / (1.0 + np.exp(z_right))
        penalty += p_right
        if p_right > 0:
            grad_x[i] -= dp_right
            grad_r[i] -= dp_right
        
        # Bottom boundary: y_i - r_i >= 0
        z_bottom = (y[i] - r[i]) / eps
        if z_bottom > 50:
            p_bottom = 0.0
            dp_bottom = 0.0
        elif z_bottom < -50:
            p_bottom = -z_bottom * eps
            dp_bottom = 1.0
        else:
            p_bottom = -eps * np.log(1.0 + np.exp(-z_bottom))
            dp_bottom = 1.0 / (1.0 + np.exp(z_bottom))
        penalty += p_bottom
        if p_bottom > 0:
            grad_y[i] += dp_bottom
            grad_r[i] -= dp_bottom
        
        # Top boundary: h - y_i - r_i >= 0
        z_top = (h - y[i] - r[i]) / eps
        if z_top > 50:
            p_top = 0.0
            dp_top = 0.0
        elif z_top < -50:
            p_top = -z_top * eps
            dp_top = 1.0
        else:
            p_top = -eps * np.log(1.0 + np.exp(-z_top))
            dp_top = 1.0 / (1.0 + np.exp(z_top))
        penalty += p_top
        if p_top > 0:
            grad_y[i] -= dp_top
            grad_r[i] -= dp_top
    
    grad = np.column_stack((grad_x, grad_y, grad_r))
    return penalty, grad

def _penalty_optimize(circles, w, h, deadline, penalty_weight=100.0, eps=0.01, maxiter=100):
    """
    Optimize using L-BFGS-B with penalty method and log-sum-exp smooth constraints.
    """
    if time.time() > deadline - 5.0:
        return None
    
    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))  # x
        bounds.append((1e-6, h))  # y
        bounds.append((1e-6, min(w, h) / 2))  # r
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        cx = x[:n]
        cy = x[n:2*n]
        cr = np.maximum(x[2*n:], 1e-6)
        circles_arr = np.column_stack((cx, cy, cr))
        
        # Base objective: negative sum of radii
        base_obj = -np.sum(cr)
        
        # Penalty
        pen, _ = _log_sum_exp_penalty(circles_arr, w, h, eps=eps)
        
        return base_obj + penalty_weight * pen
    
    def gradient(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        cx = x[:n]
        cy = x[n:2*n]
        cr = np.maximum(x[2*n:], 1e-6)
        circles_arr = np.column_stack((cx, cy, cr))
        
        # Base gradient
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Penalty gradient
        _, pen_grad = _log_sum_exp_penalty(circles_arr, w, h, eps=eps)
        grad[:n] += penalty_weight * pen_grad[:, 0]
        grad[n:2*n] += penalty_weight * pen_grad[:, 1]
        grad[2*n:] += penalty_weight * pen_grad[:, 2]
        
        return grad
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            jac=gradient,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.fun < 1e9:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], np.maximum(res.x[2*n:], 1e-6)))
            # Relax to ensure validity
            new_circles = _relax_circles(new_circles, w, h, iterations=500)
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass
    
    return None

def _high_precision_polish(circles, w, h, deadline):
    """
    Final high-precision polish with tighter tolerances and increasing penalty weights.
    """
    if time.time() > deadline - 10.0:
        return None
    
    best = circles.copy()
    best_val = _total_radius(best)
    
    # Progressive penalty weights and decreasing epsilon
    stages = [
        {'weight': 500.0, 'eps': 0.005, 'maxiter': 50},
        {'weight': 1000.0, 'eps': 0.002, 'maxiter': 50},
        {'weight': 2000.0, 'eps': 0.001, 'maxiter': 50},
        {'weight': 5000.0, 'eps': 0.0005, 'maxiter': 50},
    ]
    
    for stage in stages:
        if time.time() > deadline - 5.0:
            break
        
        result = _penalty_optimize(
            best, w, h, deadline,
            penalty_weight=stage['weight'],
            eps=stage['eps'],
            maxiter=stage['maxiter']
        )
        
        if result is not None:
            val = _total_radius(result)
            if val > best_val:
                best = result
                best_val = val
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None

    # Try a range of aspect ratios
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
    
    for ar in ratios:
        if time.time() > deadline - 15.0:
            break
        
        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2
        
        # Also try swapped orientation
        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 15.0:
                break
            
            try:
                # Initialize with grid
                circles = _make_grid(N, ww, hh)
                
                # Also try hexagonal initialization
                hex_circles = _make_hexagonal_grid(N, ww, hh)
                
                # Relax to ensure validity
                circles = _relax_circles(circles, ww, hh, iterations=500)
                hex_circles = _relax_circles(hex_circles, ww, hh, iterations=500)
                
                # Choose better initial
                if _total_radius(hex_circles) > _total_radius(circles):
                    circles = hex_circles
                
                if not _validate(circles, ww, hh):
                    continue
                
                current_value = _total_radius(circles)
                
                # Try penalty optimization with progressive refinement
                if time.time() < deadline - 20.0:
                    optimized = _penalty_optimize(circles, ww, hh, deadline, penalty_weight=100.0, eps=0.01, maxiter=100)
                    if optimized is not None:
                        circles = optimized
                        current_value = _total_radius(circles)
                
                # High precision polish
                if time.time() < deadline - 15.0:
                    polished = _high_precision_polish(circles, ww, hh, deadline)
                    if polished is not None:
                        circles = polished
                        current_value = _total_radius(circles)
                
                if current_value > best_value:
                    best_value = current_value
                    best_result = circles.copy()
                    
            except Exception:
                continue

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
