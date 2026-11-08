"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIMETER = 4.0
TIME_LIMIT = 145.0  # Safety margin before 150s limit


def _is_valid(circles, tol=1e-9):
    """Check if the configuration is valid."""
    if len(circles) == 0:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2 * (W + H) > PERIMETER + tol:
        return False
    
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - tol) ** 2:
                return False
    return True


def _compute_value(circles):
    """Compute the sum of radii for a valid configuration, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0


def _seed_grid(W, H):
    """Create a grid-based seed configuration within a box of size W x H."""
    # Try to fit a grid
    # Estimate number of circles per row/col
    # Area of box = W*H. Area per circle ~ pi*r^2.
    # Let's just try a few grid sizes.
    best_val = -1
    best_circles = None
    
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
            
        # Check if grid fits
        s_x = W / cols
        s_y = H / rows
        s = min(s_x, s_y)
        
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * s_x
                cy = (j + 0.5) * s_y
                r = 0.5 * s - 1e-4
                circles.append([cx, cy, r])
        circles = np.array(circles[:N])
        
        if _is_valid(circles):
            val = np.sum(circles[:, 2])
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
    
    return best_circles


def _optimize(W, H, initial_circles, time_budget):
    """Optimize circle positions and radii in a fixed W x H box."""
    t_start = time.time()
    best_circles = initial_circles.copy()
    best_val = _compute_value(best_circles)
    
    def objective_and_grad(params):
        if time.time() - t_start > time_budget:
            return 1e10, np.zeros_like(params)
        
        n = len(params) // 3
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0  # d(-sum r)/dr = -1
        
        # Boundary penalties
        # We want circles to stay inside [0, W] x [0, H]
        # Penalty: max(0, r - x)^2 + max(0, r - (W-x))^2 + ...
        lam_bound = 1000.0
        
        for i in range(n):
            # X boundaries
            if x[i] < r[i]:
                pen = (r[i] - x[i]) ** 2
                obj += lam_bound * pen
                # d/dx: 2(r-x)(-1) = -2(r-x) = 2(x-r)
                grad[3*i] += lam_bound * 2 * (x[i] - r[i])
                # d/dr: 2(r-x)(1) = 2(r-x)
                grad[3*i+2] += lam_bound * 2 * (r[i] - x[i])
                
            if W - x[i] < r[i]:
                pen = (r[i] - (W - x[i])) ** 2
                obj += lam_bound * pen
                # d/dx: 2(r - (W-x))(-1) = -2(r-W+x) = 2(W-x-r)
                grad[3*i] += lam_bound * 2 * (W - x[i] - r[i])
                # d/dr: 2(r - (W-x))(1) = 2(r-W+x)
                grad[3*i+2] += lam_bound * 2 * (r[i] - (W - x[i]))
                
            # Y boundaries
            if y[i] < r[i]:
                pen = (r[i] - y[i]) ** 2
                obj += lam_bound * pen
                grad[3*i+1] += lam_bound * 2 * (y[i] - r[i])
                grad[3*i+2] += lam_bound * 2 * (r[i] - y[i])
                
            if H - y[i] < r[i]:
                pen = (r[i] - (H - y[i])) ** 2
                obj += lam_bound * pen
                grad[3*i+1] += lam_bound * 2 * (H - y[i] - r[i])
                grad[3*i+2] += lam_bound * 2 * (r[i] - (H - y[i]))
                
        # Overlap penalties
        lam_ov = 1000.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                
                if dist < sum_r:
                    overlap = sum_r - dist
                    pen = overlap ** 2
                    obj += lam_ov * pen
                    
                    # Gradient of (sum_r - dist)^2
                    # d/dx_i: 2(overlap) * (-d(dist)/dx_i) = 2(overlap) * (-(x_i-x_j)/dist)
                    # Note: if dist is 0, gradient is undefined. We handle dist>0.
                    if dist > 1e-6:
                        g_x = 2 * overlap * (-(x[i] - x[j]) / dist)
                        g_y = 2 * overlap * (-(y[i] - y[j]) / dist)
                        g_r = 2 * overlap * 1  # d(sum_r)/dr = 1, d(dist)/dr = 0. So d(overlap)/dr = 1.
                        
                        grad[3*i] += lam_ov * g_x
                        grad[3*i+1] += lam_ov * g_y
                        grad[3*i+2] += lam_ov * g_r
                        
                        grad[3*j] += lam_ov * -g_x
                        grad[3*j+1] += lam_ov * -g_y
                        grad[3*j+2] += lam_ov * g_r
                    else:
                        # If dist is 0, push them apart arbitrarily
                        grad[3*i] += lam_ov
                        grad[3*j] -= lam_ov
        
        return obj, grad

    def callback(xk):
        nonlocal best_circles, best_val
        current_circles = xk.reshape(-1, 3)
        current_circles[:, 2] = np.maximum(current_circles[:, 2], 1e-6)
        val = _compute_value(current_circles)
        if val > best_val:
            best_val = val
            best_circles = current_circles.copy()
        return False # Continue

    try:
        # Use L-BFGS-B for speed
        # We don't set bounds on r, x, y strictly to allow the penalty to work, 
        # but we can set loose bounds to help the optimizer.
        bounds = [
            (0, W) for _ in range(N)  # x
        ] + [
            (0, H) for _ in range(N)  # y
        ] + [
            (1e-6, min(W, H)/2) for _ in range(N)  # r
        ]
        
        res = minimize(objective_and_grad, initial_circles.flatten(), method='L-BFGS-B', 
                       jac=True, bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8}, callback=callback)
    except Exception:
        pass
    
    return best_circles


def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle maximizing sum of radii."""
    t_start = time.time()
    best_overall = None
    best_overall_val = 0.0
    
    # Try different aspect ratios
    # W + H = 2. W can range from 0.5 to 1.5.
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
    
    for ar in aspect_ratios:
        if time.time() - t_start > TIME_LIMIT - 10.0:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        # Seed
        circles = _seed_grid(W, H)
        if circles is None:
            # Fallback seed: small circles in a grid
            cols = 5
            rows = 5
            s = min(W/cols, H/rows)
            circles = []
            for i in range(cols):
                for j in range(rows):
                    if len(circles) >= N: break
                    circles.append([(i+0.5)*s, (j+0.5)*s, 0.4*s])
            circles = np.array(circles[:N])
        
        # Optimize
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 2.0:
            optimized = _optimize(W, H, circles, time_remaining)
            val = _compute_value(optimized)
            if val > best_overall_val:
                best_overall_val = val
                best_overall = optimized
        
        # Also try a random perturbation of the seed for diversity
        if time.time() - t_start > TIME_LIMIT - 20.0:
            break
            
        circles_pert = circles.copy()
        circles_pert[:, :2] += np.random.normal(0, 0.05, circles_pert[:, :2].shape)
        circles_pert[:, :2] = np.clip(circles_pert[:, :2], 0, np.array([W, H]))
        
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 2.0:
            optimized = _optimize(W, H, circles_pert, time_remaining)
            val = _compute_value(optimized)
            if val > best_overall_val:
                best_overall_val = val
                best_overall = optimized

    if best_overall is None:
        # Fallback to a trivial valid config
        W, H = 1.0, 1.0
        circles = _seed_grid(W, H)
        if circles is None or not _is_valid(circles):
             s = 2.0 / (7 + 3)
             circles = []
             for i in range(7):
                 for j in range(3):
                     if len(circles) >= N: break
                     circles.append([(i+0.5)*s, (j+0.5)*s, 0.4*s])
             circles = np.array(circles[:N])
        best_overall = circles

    return best_overall
