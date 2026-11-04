import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
_DEADLINE = 140.0  # Safety margin for 150s limit

def _elapsed():
    return time.time() - _START

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _penalty_and_grad(x, n=N, mu=10000.0):
    """
    x: (3n,) vector [x0, y0, r0, x1, y1, r1, ...]
    Minimize -sum(r) + mu * (boundary_violations + overlap_violations)
    """
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    obj = -np.sum(rs)
    
    # Boundary Violations
    # r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
    # Violation is max(0, r_i - x_i), etc.
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)
    
    # Overlap Violations
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)
    
    total_penalty = bpen + open_
    energy = obj + mu * total_penalty
    
    # Gradient
    grad = np.zeros_like(x)
    
    # d(-sum r)/dr = -1
    grad[2::3] = -1.0
    
    # Boundary Gradient
    # d(bv)/dx_i = -1 if b_left > 0, +1 if b_right > 0
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))
    
    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri
    
    # Overlap Gradient
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist
        
        # d(ov^2)/dx_i = 2*ov * d(ov)/dx_i
        # ov = r_i + r_j - dist
        # d(ov)/dx_i = -ux
        # d(ov)/dx_j = +ux
        # d(ov)/dr_i = 1
        # d(ov)/dr_j = 1
        
        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)
        
    return energy, grad

def _is_valid(x, n=N, tol=1e-7):
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Check boundaries
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
        
    # Check overlaps
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True

def _fix_feasibility(x, n=N, tol=1e-8):
    """
    Aggressively shrink radii and adjust positions to ensure validity.
    This is a fallback to guarantee a valid output.
    """
    pts = x.reshape(n, 3).copy()
    for _ in range(100):
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # 1. Boundary fix: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
        max_r_bound = np.minimum(np.minimum(xs, ys), np.minimum(1.0 - xs, 1.0 - ys))
        # Add small epsilon to ensure strict inequality for evaluator safety
        max_r_bound = max_r_bound * (1 - 1e-6)
        rs = np.minimum(rs, max_r_bound)
        
        # 2. Overlap fix: If r_i + r_j > dist, shrink both
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)
        
        if not np.any(ov > tol):
            break
            
        # Reduce radii proportionally to overlap
        # Simple approach: reduce all radii slightly if any overlap exists
        # Or better: reduce the specific circles involved
        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)
        
        # Reduce radius by half the overlap amount + small factor
        reduction = overlap_per_circle * 0.5 + 1e-6
        rs = np.maximum(0.0, rs - reduction)
        
        pts[:, 2] = rs
        
    return pts.ravel()

def construct():
    global _START
    _START = time.time()
    
    best_score = -np.inf
    best_x = None
    
    # Bounds for L-BFGS-B
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Generate diverse initializations
    inits = []
    
    # 1. Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    xs, ys, rs = [], [], []
    for i in range(side):
        for j in range(side):
            if len(xs) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            xs.append(x)
            ys.append(y)
            rs.append(r)
    # Pad if necessary
    while len(xs) < N:
        xs.append(0.5)
        ys.append(0.5)
        rs.append(0.05)
    inits.append(np.column_stack((xs, ys, rs)).ravel())
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(20): # 20 random starts
        xs = rng.random(N)
        ys = rng.random(N)
        rs = rng.random(N) * 0.05 + 0.01
        inits.append(np.column_stack((xs, ys, rs)).ravel())
        
    # 3. Hexagonal-like initialization
    # Approximate positions for 26 circles
    # 5 rows of 5, 5, 5, 5, 4, 2? No, let's just use a dense grid perturbed
    # A better hex init:
    r_est = 0.1
    x_spacing = 2 * r_est
    y_spacing = r_est * np.sqrt(3)
    xs, ys, rs = [], [], []
    for row in range(6):
        count = 5 if row < 5 else 4
        if len(xs) + count > N:
            count = N - len(xs)
        y = (row + 0.5) * y_spacing
        for col in range(count):
            x = (col + 0.5) * x_spacing + (0.5 * x_spacing if row % 2 else 0)
            # Center the whole block
            # Rough centering
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    if len(xs) < N:
        while len(xs) < N:
            xs.append(0.5)
            ys.append(0.5)
            rs.append(0.05)
    # Normalize to fit in [0,1]
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    # Shift to center
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9) * 0.9 + 0.05
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9) * 0.9 + 0.05
    inits.append(np.column_stack((xs, ys, rs[:N])).ravel())
    
    # Optimize each initialization
    for i, x0 in enumerate(inits):
        if _remaining() < 5.0:
            break
            
        try:
            # Run L-BFGS-B
            # We use a high penalty mu to enforce constraints strictly during optimization
            res = minimize(
                _penalty_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6, 'disp': False}
            )
            x_opt = res.x
            
            # Ensure feasibility with a final projection
            x_valid = _fix_feasibility(x_opt)
            
            # Check validity
            if _is_valid(x_valid):
                score = np.sum(x_valid.reshape(N, 3)[:, 2])
                if score > best_score:
                    best_score = score
                    best_x = x_valid.copy()
        except Exception:
            continue
            
    # If no valid solution found, return a simple valid grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells)
        
    return best_x.reshape(N, 3)
