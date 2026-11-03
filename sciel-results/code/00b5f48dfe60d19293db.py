"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Combines hexagonal structured initialization with softplus gradient optimization.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 145.0  # Safety margin

def _softplus(x, beta=100.0):
    """Stable softplus function."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus (sigmoid scaled by beta)."""
    xb = beta * x
    return np.where(xb > 0,
                    1.0 / (1.0 + np.exp(-xb)),
                    np.exp(xb) / (1.0 + np.exp(xb)))

def _objective_and_grad(x, w, h, beta=100.0):
    """
    Objective: -sum(r) + penalties for boundary violations and overlaps.
    x: array of shape (3*N,) = [x1, y1, r1, x2, y2, r2, ...]
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Boundary penalties
    # Left: r - x > 0  => violation if x - r < 0
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    # Right: x + r - w > 0 => violation if x + r > w
    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    # Bottom: r - y > 0 => violation if y - r < 0
    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    # Top: y + r - h > 0 => violation if y + r > h
    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    # Radius positivity penalty (weak)
    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties
    # We vectorize the inner loop for speed
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        # Only penalize if overlapping (v > 0)
        # Softplus handles this smoothly
        sp = _softplus(v, beta)
        g = _softplus_grad(v, beta)
        
        obj += np.sum(sp)
        
        # Gradients for circle i
        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)
        
        # Gradients for circles j > i
        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    # Gradient for maximizing radii (minimizing -sum(r))
    grad[2::3] += -1.0

    return obj, grad

def _strictly_valid(circles, w, h, tol=1e-8):
    """Check if circles are within bounds and disjoint."""
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True

def _hex_init(w, h, seed=0):
    """
    Generate hex-packed initial configurations with various row layouts.
    This is the key structural improvement.
    """
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [4, 4, 4, 4, 4, 1],
        [1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4],
        [3, 4, 4, 5, 5],
        [5, 5, 4, 4, 3],
        [4, 4, 4, 5, 4],
        [5, 4, 4, 4, 4],
        [4, 4, 5, 4, 4],
        [3, 4, 4, 4, 6],
        [6, 4, 4, 4, 3],
    ]
    
    best = None
    best_val = -np.inf
    
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        
        # Estimate radius from horizontal and vertical constraints
        # Horizontal: max_cols * 2r <= w  => r <= w / (2 * max_cols)
        # Vertical: n_rows * r * sqrt(3) <= h => r <= h / (n_rows * sqrt(3))
        # Use a slightly conservative estimate to start valid
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        
        if r_est < 0.01:
            continue
        
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            # Vertical position
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            
            # Horizontal offset for hex packing
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            
            # Center the row horizontally
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
                
        circles = np.array(circles)
        
        # Check validity
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
                
    if best is None:
        # Fallback: simple grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
        
    return best

def _grid_init(w, h):
    """Simple grid initialization."""
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)

def _random_init(w, h, seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.2
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles

def _scale_to_perim(circles, target_perim=4.0):
    """Scale circles to fit within a bounding box of target perimeter."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    width = max_x - min_x
    height = max_y - min_y
    perim = 2.0 * (width + height)
    
    if perim <= target_perim + 1e-9:
        return circles
    
    scale = target_perim / perim
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    xs_new = cx + (xs - cx) * scale
    ys_new = cy + (ys - cy) * scale
    rs_new = rs * scale
    
    return np.column_stack([xs_new, ys_new, rs_new])

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_value = -np.inf
    best_result = None
    
    # Aspect ratios to try: w/h
    # Perimeter 2(w+h)=4 => w+h=2
    # If ar = w/h, then w = 2*ar/(1+ar), h = 2/(1+ar)
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.9, 1.1, 0.75, 1.25, 0.85, 1.15]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Generate initializations
        inits = []
        
        # Hex inits are most important
        for seed in range(5):
            circ = _hex_init(w, h, seed)
            inits.append(circ)
            
        # Grid inits
        circ = _grid_init(w, h)
        inits.append(circ)
        
        # Random inits
        for seed in [1, 2, 3]:
            circ = _random_init(w, h, seed)
            inits.append(circ)
            
        for init_circles in inits:
            if time.time() > deadline - 2:
                break
                
            x0 = init_circles.flatten()
            
            # Define bounds
            bounds = []
            for i in range(N):
                bounds.append((0, w)) # x
                bounds.append((0, h)) # y
                bounds.append((1e-6, 1.0)) # r
                
            try:
                # Use SLSQP with gradient
                res = minimize(
                    lambda p: _objective_and_grad(p, w, h)[0],
                    x0,
                    jac=lambda p: _objective_and_grad(p, w, h)[1],
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                
                if res.success or res.fun < 1e6:
                    x_opt = res.x
                    circles_opt = x_opt.reshape(-1, 3)
                    
                    # Check validity
                    if _strictly_valid(circles_opt, w, h):
                        val = np.sum(circles_opt[:, 2])
                        if val > best_value:
                            best_value = val
                            best_result = circles_opt.copy()
                            
            except Exception:
                pass
                
    # Final scaling to ensure perimeter <= 4
    if best_result is not None:
        best_result = _scale_to_perim(best_result, 4.0)
    else:
        # Fallback
        best_result = _grid_init(1.0, 1.0)
        best_result = _scale_to_perim(best_result, 4.0)
        
    return best_result
