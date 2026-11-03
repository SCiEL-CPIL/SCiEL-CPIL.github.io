"""Circle packing in perimeter-4 rectangle: SLSQP with explicit constraints."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def _strictly_valid(circles, w, h, tol=1e-8):
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

def _make_constraints(w, h):
    constraints = []
    
    # Boundary constraints: r <= x, r <= w-x, r <= y, r <= h-y
    # x - r >= 0
    # w - x - r >= 0
    # y - r >= 0
    # h - y - r >= 0
    
    # We define constraints as functions that return >= 0
    # Constraint: x_i - r_i >= 0
    def c_left(x):
        c = x.reshape(-1, 3)
        return c[:, 0] - c[:, 2]
    def c_right(x):
        c = x.reshape(-1, 3)
        return w - c[:, 0] - c[:, 2]
    def c_bottom(x):
        c = x.reshape(-1, 3)
        return c[:, 1] - c[:, 2]
    def c_top(x):
        c = x.reshape(-1, 3)
        return h - c[:, 1] - c[:, 2]
        
    constraints.append({'type': 'ineq', 'fun': c_left})
    constraints.append({'type': 'ineq', 'fun': c_right})
    constraints.append({'type': 'ineq', 'fun': c_bottom})
    constraints.append({'type': 'ineq', 'fun': c_top})
    
    # Pairwise non-overlap: dist_ij - (r_i + r_j) >= 0
    # dist_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    # We need to define these for all pairs i < j
    indices = []
    for i in range(N):
        for j in range(i+1, N):
            indices.append((i, j))
            
    for i, j in indices:
        def make_pair_constraint(i, j):
            def c_pair(x):
                c = x.reshape(-1, 3)
                xi, yi, ri = c[i]
                xj, yj, rj = c[j]
                d = np.sqrt((xi - xj)**2 + (yi - yj)**2 + 1e-12)
                return d - ri - rj
            return c_pair
        constraints.append({'type': 'ineq', 'fun': make_pair_constraint(i, j)})
        
    return constraints

def _hex_init(w, h, seed):
    """
    Generate a hexagonal packing initialization.
    Tries to fit a hexagonal grid of 21 circles.
    """
    # 21 circles can be arranged in rows like 4-5-4-5-3 or similar.
    # Or a hexagon shape: 1, 3, 5, 3, 1 is 13. 1, 3, 5, 7, 5, 3, 1 is 25.
    # 21 is 1+3+5+3+5+3+1? No.
    # Common arrangements for 21:
    # 4-5-4-5-3 (21)
    # 5-4-5-4-3 (21)
    # 3-4-5-4-5 (21)
    # Let's try a few row configurations
    
    configs = [
        [4, 5, 4, 5, 3],
        [5, 4, 5, 4, 3],
        [3, 4, 5, 4, 5],
        [4, 4, 5, 4, 4],
        [5, 5, 5, 3, 3],
        [3, 3, 5, 5, 5],
        [4, 5, 5, 4, 3],
        [3, 4, 5, 5, 4],
    ]
    
    best_circles = None
    best_val = -np.inf
    
    for row_counts in configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        # Estimate radius based on width
        max_cols = max(row_counts)
        # In hex packing, horizontal spacing is 2r, vertical is sqrt(3)r
        # Total width approx max_cols * 2r
        # Total height approx (n_rows - 1) * sqrt(3)r + 2r
        est_r_w = w / (max_cols * 2.0)
        est_r_h = h / ((n_rows - 1) * np.sqrt(3) + 2.0)
        r_est = min(est_r_w, est_r_h) * 0.95
        
        if r_est < 0.05:
            continue
            
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = r_est + row_idx * r_est * np.sqrt(3)
            # Offset alternating rows
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            # Center the row horizontally
            row_width = n_cols * 2.0 * r_est
            start_x = (w - row_width) / 2.0 + r_est + x_offset
            # Adjust start_x so the first circle is at start_x
            # Actually, if row_width is n_cols * 2r, the centers are at start_x, start_x+2r, ...
            # Let's just distribute them evenly within a centered block
            # Block width = (n_cols - 1) * 2r
            block_width = (n_cols - 1) * 2.0 * r_est
            center_x = w / 2.0
            for col_idx in range(n_cols):
                x = center_x - block_width / 2.0 + col_idx * 2.0 * r_est + x_offset
                circles.append([x, y, r_est])
                
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
                
    if best_circles is None:
        # Fallback grid
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
        best_circles = np.array(circles)
        
    return best_circles

def _grid_init(w, h, seed):
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
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.25
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles

def _optimize_slsqp(circles0, w, h, deadline, max_iter=200):
    x0 = circles0.flatten()
    r_max = min(w, h) / 2.0
    bounds = []
    for _ in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((1e-6, r_max))
        
    constraints = _make_constraints(w, h)
    
    # Objective: maximize sum of radii -> minimize -sum(radii)
    def objective(x):
        c = x.reshape(-1, 3)
        return -np.sum(c[:, 2])
        
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
        )
    except Exception:
        return None, -np.inf
        
    if time.time() > deadline:
        return None, -np.inf
        
    circles = res.x.reshape(-1, 3)
    if _strictly_valid(circles, w, h):
        return circles, float(np.sum(circles[:, 2]))
    return None, -np.inf

def _polish(circles, w, h, deadline):
    """
    Try to increase radii slightly while maintaining validity.
    """
    circles = circles.copy()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            old_r = circles[i, 2]
            # Try increasing radius by 1%
            new_r = old_r * 1.01
            circles[i, 2] = new_r
            if _strictly_valid(circles, w, h):
                improved = True
            else:
                circles[i, 2] = old_r
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_circles = None
    best_value = -np.inf
    best_w = best_h = None
    
    # Aspect ratios: w + h = 2
    aspect_ratios = [1.0, 1.2, 1.4, 1.6, 1.8, 1.3, 1.5, 1.7,
                     0.8, 0.9, 1.1, 2.0, 0.7, 2.2, 0.65, 2.35, 0.6]
                     
    seed_counter = 0
    
    for w in aspect_ratios:
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue
        if time.time() > deadline:
            break
            
        inits = [
            _hex_init(w, h, seed_counter),
            _grid_init(w, h, seed_counter + 1000),
            _random_init(w, h, seed_counter + 2000),
        ]
        seed_counter += 50
        
        for circles0 in inits:
            if time.time() > deadline:
                break
            circles, val = _optimize_slsqp(circles0, w, h, deadline, max_iter=150)
            if circles is not None:
                circles = _polish(circles, w, h, deadline)
                if _strictly_valid(circles, w, h):
                    val = float(np.sum(circles[:, 2]))
                    if val > best_value:
                        best_value = val
                        best_circles = circles.copy()
                        best_w, best_h = w, h
                        
    # Final polish on the best found
    if best_circles is not None and time.time() < deadline - 5:
        circles = _polish(best_circles, best_w, best_h, deadline)
        if _strictly_valid(circles, best_w, best_h):
            val = float(np.sum(circles[:, 2]))
            if val > best_value:
                best_value = val
                best_circles = circles.copy()
                
    if best_circles is None:
        # Fallback
        w, h = 1.4, 0.6
        a, b = 7, 3
        circles = np.zeros((N, 3))
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / a, h / b) * 0.45
            circles[i, 0] = (col + 0.5) * w / a
            circles[i, 1] = (row + 0.5) * h / b
            circles[i, 2] = r
        best_circles = circles
        
    return best_circles
