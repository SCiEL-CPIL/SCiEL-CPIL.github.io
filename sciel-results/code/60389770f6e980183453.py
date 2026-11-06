import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 120.0  # Safety margin
N = 200

def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    
    # Distance to walls
    wall_dist = np.min(0.5 - np.abs(centers), axis=None)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return float(wall_dist)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(float(wall_dist), float(min_pair_dist / 2.0))

def _hex_init():
    """Generate hexagonal grid initialization."""
    # Approximate number of points per side for hex grid
    # Area = 1. Hex cell area = s^2 * sqrt(3)/2.
    # N cells ~ 200. s^2 ~ 200 * 2 / sqrt(3) ~ 230. s ~ 15.
    # But we need to fit in [-0.5, 0.5].
    # Let's just generate a grid of hex points and pick 200.
    
    # Generate points
    points = []
    s = 0.06 # spacing
    # Range to cover [-0.5, 0.5]
    range_x = int(1.0 / s) + 2
    range_y = int(1.0 / (s * np.sqrt(3)/2)) + 2
    
    for i in range(-range_x, range_x + 1):
        for j in range(-range_y, range_y + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have more than N, we need to select N.
    # We want to maximize the minimum distance.
    # A simple heuristic: pick the points that are most "central" or use a greedy farthest-point sampling.
    # For now, let's just take the first N if sorted, or random.
    # Better: Sort by distance to center? No, that might cluster.
    # Let's just take a random subset? Or the first N from a sorted list?
    # Actually, a regular hex grid is already good. If we have > N, we can just drop the ones closest to the boundary?
    # Or just take the first N.
    
    if len(points) > N:
        # Random selection might be bad. Let's just take the first N.
        # Or, better: we can optimize the hex grid spacing s to get exactly N or close.
        # For now, just take first N.
        points = points[:N]
    else:
        # Pad with random points if too few
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
            
    return points

def _violation_and_grad(x, R):
    """
    Calculate violation and gradient for target radius R.
    x: flattened array of centers (2N).
    R: target radius.
    """
    c = x.reshape(-1, 2)
    n = len(c)
    
    # Wall violations
    # Constraints: x >= -0.5 + R, x <= 0.5 - R, y >= -0.5 + R, y <= 0.5 - R
    # Violation terms: max(0, -0.5 + R - x)^2 + max(0, x - (0.5 - R))^2 + ...
    
    viol = 0.0
    grad = np.zeros_like(x)
    
    # Walls
    # Left: -0.5 + R - x
    v_l = -0.5 + R - c[:, 0]
    mask_l = v_l > 0
    if np.any(mask_l):
        viol += np.sum(v_l[mask_l] ** 2)
        grad[2*mask_l] -= 2 * v_l[mask_l] # d/dx max(0, v)^2 = 2v * 1. Here v = -0.5+R-x. d/dx = -1. So -2v.
        
    # Right: x - (0.5 - R)
    v_r = c[:, 0] - (0.5 - R)
    mask_r = v_r > 0
    if np.any(mask_r):
        viol += np.sum(v_r[mask_r] ** 2)
        grad[2*mask_r] += 2 * v_r[mask_r]
        
    # Bottom: -0.5 + R - y
    v_b = -0.5 + R - c[:, 1]
    mask_b = v_b > 0
    if np.any(mask_b):
        viol += np.sum(v_b[mask_b] ** 2)
        grad[2*mask_b + 1] -= 2 * v_b[mask_b]
        
    # Top: y - (0.5 - R)
    v_t = c[:, 1] - (0.5 - R)
    mask_t = v_t > 0
    if np.any(mask_t):
        viol += np.sum(v_t[mask_t] ** 2)
        grad[2*mask_t + 1] += 2 * v_t[mask_t]
        
    # Pairwise
    # d_ij < 2R
    # T_ij = max(0, 2R - d_ij)^2
    # Grad w.r.t c_i: -2(2R - d) * (c_i - c_j)/d
    
    # Vectorized calculation is hard for gradient accumulation.
    # Loop is O(N^2). N=200, N^2=40000. Fast enough in Python? Maybe.
    # Let's try to vectorize the distance part, then loop for grad accumulation?
    # Or just loop. 40k iterations per gradient eval. L-BFGS-B might call grad 100-1000 times.
    # 40M ops. Should be fine in 120s?
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d < 2 * R:
                # Violation
                v = 2 * R - d
                viol += v ** 2
                
                # Gradient
                # dT/dc_i = -2v * diff/d
                # dT/dc_j = -2v * (-diff)/d = 2v * diff/d
                factor = -2 * v / d if d > 1e-12 else 0
                grad_i = factor * diff
                grad_j = -factor * diff
                
                grad[2*i] += grad_i[0]
                grad[2*i+1] += grad_i[1]
                grad[2*j] += grad_j[0]
                grad[2*j+1] += grad_j[1]
                
    return viol, grad

def _optimize_for_radius(centers, R, max_iter=500):
    """Optimize centers to satisfy radius R."""
    x0 = centers.flatten()
    
    def fun_and_grad(x):
        v, g = _violation_and_grad(x, R)
        return v, g
        
    # L-BFGS-B
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    return res.x.reshape(-1, 2), res.fun

def construct():
    centers = _hex_init()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    current_R = best_radius
    step = 0.0001
    
    while _remaining_time() > 10.0:
        # Try to increase R
        target_R = current_R + step
        
        # If target_R is too big (e.g. > 0.5), break
        if target_R > 0.5:
            break
            
        # Optimize
        try:
            new_centers, viol = _optimize_for_radius(centers, target_R, max_iter=200)
            
            # Check if valid
            # If viol is small, accept
            if viol < 1e-6:
                current_R = target_R
                centers = new_centers
                best_radius = current_R
                best_centers = centers.copy()
                # Maybe increase step slightly?
                # step *= 1.05
            else:
                # If failed, maybe try smaller step or break
                # Try a smaller step
                step *= 0.5
                if step < 1e-6:
                    break
        except Exception:
            break
            
    # Final polish: Optimize for current best_radius to minimize violations (should be 0)
    # and ensure we are in a good local minimum
    if _remaining_time() > 5.0:
        try:
            polished, _ = _optimize_for_radius(best_centers, best_radius, max_iter=500)
            r_polished = _compute_radius(polished)
            if r_polished >= best_radius - 1e-9:
                best_centers = polished
                best_radius = r_polished
        except Exception:
            pass
            
    return best_centers
