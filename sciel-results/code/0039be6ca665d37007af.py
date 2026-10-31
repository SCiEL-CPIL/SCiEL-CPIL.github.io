"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _make_hex_grid(n, w, h, seed=0):
    """Create a hexagonal lattice clipped to rectangle [0,w]x[0,h]."""
    rng = np.random.RandomState(seed)
    # Hexagonal spacing
    # Try to fit n circles
    # Start with a reasonable spacing
    area = w * h
    # Approximate radius
    r_est = np.sqrt(area / (n * 2 * np.sqrt(3)))  # hex packing density
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    # Generate grid
    pts = []
    row = 0
    while len(pts) < n + 10:
        y = dy * row
        if y > h:
            break
        x_offset = dx / 2 if row % 2 else 0
        col = 0
        while True:
            x = dx * col + x_offset
            if x > w:
                break
            if 0 <= x <= w and 0 <= y <= h:
                pts.append((x, y))
            col += 1
        row += 1
    
    if len(pts) < n:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(n)))
        pts = [((i + 0.5) / side * w, (j + 0.5) / side * h) for i in range(side) for j in range(side)]
    
    # Shuffle and take first n
    rng.shuffle(pts)
    return np.array(pts[:n])


def _objective(params, n, w, h):
    """Objective: maximize sum of radii subject to constraints.
    params: [x1, y1, r1, x2, y2, r2, ...]
    We minimize -sum(r) + penalty for violations.
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Bounds penalty
    pen = 0.0
    for i in range(n):
        if xs[i] < rs[i]:
            pen += (rs[i] - xs[i]) ** 2
        if ys[i] < rs[i]:
            pen += (rs[i] - ys[i]) ** 2
        if xs[i] + rs[i] > w:
            pen += (xs[i] + rs[i] - w) ** 2
        if ys[i] + rs[i] > h:
            pen += (ys[i] + rs[i] - h) ** 2
    
    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    
    return -np.sum(rs) + 100.0 * pen


def _grad(params, n, w, h):
    """Gradient of the objective."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    grad = np.zeros_like(params)
    
    # Bounds gradient
    for i in range(n):
        if xs[i] < rs[i]:
            grad[i * 3] += 200.0 * (rs[i] - xs[i])
            grad[i * 3 + 2] += -200.0 * (rs[i] - xs[i])
        if ys[i] < rs[i]:
            grad[i * 3 + 1] += 200.0 * (rs[i] - ys[i])
            grad[i * 3 + 2] += -200.0 * (rs[i] - ys[i])
        if xs[i] + rs[i] > w:
            grad[i * 3] += 200.0 * (xs[i] + rs[i] - w)
            grad[i * 3 + 2] += 200.0 * (xs[i] + rs[i] - w)
        if ys[i] + rs[i] > h:
            grad[i * 3 + 1] += 200.0 * (ys[i] + rs[i] - h)
            grad[i * 3 + 2] += 200.0 * (ys[i] + rs[i] - h)
    
    # Overlap gradient
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                diff = min_dist - dist
                if dist > 1e-10:
                    unit_x = dx / dist
                    unit_y = dy / dist
                    grad[i * 3] += 200.0 * diff * unit_x
                    grad[i * 3 + 1] += 200.0 * diff * unit_y
                    grad[j * 3] -= 200.0 * diff * unit_x
                    grad[j * 3 + 1] -= 200.0 * diff * unit_y
                    grad[i * 3 + 2] += 200.0 * diff
                    grad[j * 3 + 2] += 200.0 * diff
    
    # Radius gradient (negative since we minimize -sum(r))
    for i in range(n):
        grad[i * 3 + 2] -= 1.0
    
    return grad


def _soft_min(a, b, k=10.0):
    """Soft minimum for smooth optimization."""
    return -np.logaddexp(-k * a, -k * b) / k


def _smooth_objective(params, n, w, h, k=5.0):
    """Smooth objective using soft constraints."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Bounds: use soft max for violations
    viol = 0.0
    for i in range(n):
        # x < r -> violation
        viol += np.maximum(0, rs[i] - xs[i]) ** 2
        viol += np.maximum(0, rs[i] - ys[i]) ** 2
        viol += np.maximum(0, xs[i] + rs[i] - w) ** 2
        viol += np.maximum(0, ys[i] + rs[i] - h) ** 2
    
    # Overlap: soft version
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = rs[i] + rs[j]
            viol += np.maximum(0, min_dist - dist) ** 2
    
    return -np.sum(rs) + 50.0 * viol


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = -1.0
    best_result = None
    
    # Rectangle with perimeter 4: 2*(w+h) = 4, so w+h = 2
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 2.5, 0.33, 3.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        w = ar
        h = 2.0 - ar
        if w <= 0 or h <= 0:
            continue
        
        # Multiple initializations
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize with hex grid
            pts = _make_hex_grid(N, w, h, seed)
            
            # Initial radius estimate
            r_init = 0.02
            params = np.zeros(N * 3)
            for i in range(N):
                params[i * 3] = pts[i, 0]
                params[i * 3 + 1] = pts[i, 1]
                params[i * 3 + 2] = r_init
            
            # Gradient descent with adaptive step size
            step_size = 0.001
            for it in range(500):
                if time.time() > deadline - 5.0:
                    break
                
                obj = _smooth_objective(params, N, w, h)
                grad = _grad(params, N, w, h)
                
                # Adaptive step size
                params_new = params - step_size * grad
                
                obj_new = _smooth_objective(params_new, N, w, h)
                
                if obj_new < obj:
                    params = params_new
                    step_size = min(step_size * 1.05, 0.01)
                else:
                    step_size *= 0.7
                
                # Check convergence
                if np.max(np.abs(grad)) < 1e-6:
                    break
            
            # Extract result
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            # Validate and clamp
            valid = True
            for i in range(N):
                if rs[i] < 1e-6:
                    valid = False
                    break
                if xs[i] < rs[i] - 1e-6 or ys[i] < rs[i] - 1e-6:
                    valid = False
                    break
                if xs[i] + rs[i] > w + 1e-6 or ys[i] + rs[i] > h + 1e-6:
                    valid = False
                    break
            
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                    if dist < rs[i] + rs[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([xs, ys, rs])
            
            # Try to increase radii greedily
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(N):
                    old_r = rs[i]
                    # Try increasing radius
                    for dr in [0.005, 0.002, 0.001]:
                        new_r = old_r + dr
                        # Check if still valid
                        ok = True
                        if xs[i] < new_r or ys[i] < new_r:
                            ok = False
                        if xs[i] + new_r > w or ys[i] + new_r > h:
                            ok = False
                        if ok:
                            for j in range(N):
                                if i == j:
                                    continue
                                dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                                if dist < new_r + rs[j]:
                                    ok = False
                                    break
                        if ok:
                            rs[i] = new_r
                            improved = True
                            break
                if not improved:
                    break
            
            # Re-validate after radius expansion
            valid = True
            for i in range(N):
                if rs[i] < 1e-6:
                    valid = False
                    break
                if xs[i] < rs[i] - 1e-6 or ys[i] < rs[i] - 1e-6:
                    valid = False
                    break
                if xs[i] + rs[i] > w + 1e-6 or ys[i] + rs[i] > h + 1e-6:
                    valid = False
                    break
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                    if dist < rs[i] + rs[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = np.sum(rs)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([xs, ys, rs])
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
