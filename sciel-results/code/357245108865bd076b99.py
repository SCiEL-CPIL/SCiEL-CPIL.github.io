"""Circle packing in perimeter-4 rectangle using pebble growth + local refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s limit
    
    best_value = 0.0
    best_result = None
    
    # Try multiple aspect ratios and initial configurations
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.45, 1.6, 0.4, 1.7, 0.35, 1.8, 0.3, 2.0, 0.25]
    
    for ar_idx, w0 in enumerate(aspect_ratios):
        if time.time() > deadline:
            break
        h0 = 2.0 - w0
        if h0 < 0.2:
            continue
        
        # Try multiple random seeds for each aspect ratio
        for seed_offset in range(5):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(ar_idx * 100 + seed_offset * 17 + 3)
            
            # Generate initial centers using a jittered grid
            side = int(np.ceil(np.sqrt(N)))
            centers = []
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                # Add jitter
                cx += rng.uniform(-0.02, 0.02)
                cy += rng.uniform(-0.02, 0.02)
                centers.append([cx, cy])
            
            centers = np.array(centers)
            
            # Pebble growth: start with tiny radii and grow
            result = _pebble_growth(centers, w0, h0, deadline - (time.time() - start_time))
            
            if result is not None:
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result
            
            # Local refinement using SLSQP
            if result is not None and time.time() < deadline - 5:
                refined = _local_refine(result, deadline - (time.time() - start_time))
                if refined is not None:
                    ref_val = np.sum(refined[:, 2])
                    if ref_val > best_value:
                        best_value = ref_val
                        best_result = refined
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        w0, h0 = 1.0, 1.0
        cells = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r = min(w0 / side, h0 / side) * 0.4
            cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    return best_result


def _pebble_growth(centers, w, h, time_budget):
    """
    Pebble growth algorithm: grow circles from initial centers until they
    touch neighbors or boundaries, then optimize aspect ratio.
    """
    start_time = time.time()
    n = len(centers)
    centers = centers.copy()
    radii = np.full(n, 0.01)  # Start with tiny radii
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > start_time + time_budget:
            break
        
        # For each circle, compute the max radius it can grow to
        # without overlapping any other circle or going out of bounds
        grow_amounts = np.zeros(n)
        
        for i in range(n):
            # Distance to boundaries
            dist_left = centers[i, 0]
            dist_right = w - centers[i, 0]
            dist_bottom = centers[i, 1]
            dist_top = h - centers[i, 1]
            max_r_boundary = min(dist_left, dist_right, dist_bottom, dist_top)
            
            # Distance to other circles
            max_r_neighbors = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Max radius such that circles don't overlap: r_i + r_j <= dist
                # So r_i <= dist - r_j
                available = dist - radii[j]
                if available < max_r_neighbors:
                    max_r_neighbors = available
            
            grow_amounts[i] = max(0.0, max_r_neighbors - radii[i])
        
        # Grow all circles by the minimum grow amount (uniform growth step)
        min_grow = np.min(grow_amounts)
        if min_grow < 1e-8:
            # At least one circle can't grow, so we need to move centers
            # Use force-directed relaxation to resolve
            centers = _force_relax_positions(centers, radii, w, h, max_iter=100)
            # After relaxation, try growing again
            continue
        
        # Grow uniformly
        radii += min_grow
        
        # Check if we've reached a stable state (all circles touching something)
        if min_grow < 1e-6:
            break
    
    # Final force relaxation to ensure validity
    centers = _force_relax_positions(centers, radii, w, h, max_iter=200)
    
    # Verify and clip
    result = _validate_and_clip(centers, radii, w, h)
    return result


def _force_relax_positions(centers, radii, w, h, max_iter=200):
    """
    Force-directed relaxation to resolve overlaps and boundary violations.
    Moves centers to eliminate overlaps while keeping circles inside the rectangle.
    """
    centers = centers.copy()
    n = len(centers)
    dt = 0.005
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall forces
        for i in range(n):
            if centers[i, 0] - radii[i] < 0:
                fx[i] += (centers[i, 0] - radii[i]) * 50
            if centers[i, 0] + radii[i] > w:
                fx[i] -= (centers[i, 0] + radii[i] - w) * 50
            if centers[i, 1] - radii[i] < 0:
                fy[i] += (centers[i, 1] - radii[i]) * 50
            if centers[i, 1] + radii[i] > h:
                fy[i] -= (centers[i, 1] + radii[i] - h) * 50
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d_sq = dx*dx + dy*dy
                min_d = radii[i] + radii[j]
                if d_sq < min_d * min_d and d_sq > 1e-12:
                    d = np.sqrt(d_sq)
                    overlap = min_d - d
                    f = overlap * 20
                    fx[i] += (dx / d) * f
                    fy[i] += (dy / d) * f
                    fx[j] -= (dx / d) * f
                    fy[j] -= (dy / d) * f
        
        # Update
        centers += dt * np.column_stack([fx, fy])
        
        # Clamp to bounds
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
    
    return centers


def _validate_and_clip(centers, radii, w, h):
    """Validate the configuration and clip any small violations."""
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()
    
    # Clip radii to fit within bounds
    for i in range(n):
        max_r = min(centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max(0.001, max_r * 0.99)
    
    # Final overlap check and resolution
    for _ in range(50):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    valid = False
                    # Shrink the larger circle slightly
                    if radii[i] >= radii[j]:
                        radii[i] *= 0.99
                    else:
                        radii[j] *= 0.99
                    break
            if not valid:
                break
        
        if valid:
            break
    
    # Final boundary check
    for i in range(n):
        max_r = min(centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
        if radii[i] > max_r:
            radii[i] = max(0.001, max_r * 0.99)
    
    return np.column_stack([centers, radii])


def _local_refine(result, time_budget):
    """
    Local refinement using SLSQP to maximize sum of radii with hard constraints.
    """
    if time_budget < 3:
        return result
    
    n = N
    # Parameter vector: [x1, y1, r1, ..., xn, yn, rn, w]
    # h = 2 - w
    x0 = result[:, 0].copy()
    y0 = result[:, 1].copy()
    r0 = result[:, 2].copy()
    w0 = 1.0
    h0 = 1.0
    
    # Estimate w and h from the current bounding box
    x_min = np.min(x0 - r0)
    x_max = np.max(x0 + r0)
    y_min = np.min(y0 - r0)
    y_max = np.max(y0 + r0)
    
    # Fit w and h to contain all circles
    w_est = max(0.5, x_max - x_min + 0.1)
    h_est = max(0.5, y_max - y_min + 0.1)
    
    # Scale to perimeter 4
    scale = 2.0 / (w_est + h_est)
    w0 = w_est * scale
    h0 = h_est * scale
    
    # Scale centers and radii
    x0 = x0 * scale
    y0 = y0 * scale
    r0 = r0 * scale
    
    z0 = np.concatenate([x0, y0, r0, [w0]])
    
    def objective(z):
        x = z[0:n]
        y = z[n:2*n]
        r = z[2*n:3*n]
        w = z[3*n]
        h = 2.0 - w
        
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6
        
        obj = -np.sum(r)
        
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                dist_req = r[i] + r[j]
                if d_sq < dist_req * dist_req:
                    d = np.sqrt(d_sq)
                    overlap = dist_req - d
                    obj += 1000.0 * overlap * overlap
        
        # Boundary penalty
        for i in range(n):
            if x[i] < r[i]:
                obj += 1000.0 * (r[i] - x[i]) ** 2
            elif x[i] > w - r[i]:
                obj += 1000.0 * (x[i] - (w - r[i])) ** 2
            if y[i] < r[i]:
                obj += 1000.0 * (r[i] - y[i]) ** 2
            elif y[i] > h - r[i]:
                obj += 1000.0 * (y[i] - (h - r[i])) ** 2
        
        return obj
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.01, 2.0))  # x
        bounds.append((0.01, 2.0))  # y
        bounds.append((0.005, 0.5))  # r
    bounds.append((0.3, 1.7))  # w
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.fun < 1e4:
            x_opt = res.x[0:n]
            y_opt = res.x[n:2*n]
            r_opt = res.x[2*n:3*n]
            w_opt = res.x[3*n]
            h_opt = 2.0 - w_opt
            
            if w_opt > 0.2 and h_opt > 0.2 and np.all(r_opt > 0):
                # Validate
                valid = True
                for i in range(n):
                    if x_opt[i] < r_opt[i] - 1e-6 or x_opt[i] > w_opt - r_opt[i] + 1e-6:
                        valid = False
                        break
                    if y_opt[i] < r_opt[i] - 1e-6 or y_opt[i] > h_opt - r_opt[i] + 1e-6:
                        valid = False
                        break
                    if r_opt[i] <= 0:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = x_opt[i] - x_opt[j]
                            dy = y_opt[i] - y_opt[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r_opt[i] + r_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    return np.column_stack([x_opt, y_opt, r_opt])
    
    except Exception:
        pass
    
    return None
