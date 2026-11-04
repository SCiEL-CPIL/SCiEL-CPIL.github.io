import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(pos_r, w, h, n_circles, tol=1e-7):
    """Check validity: disjoint interiors, within bounds."""
    if pos_r.shape[0] != n_circles:
        return False
    xs, ys, rs = pos_r[:, 0], pos_r[:, 1], pos_r[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    # Vectorized overlap check
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = rs[:, None] + rs[None, :]
    if np.any(dist < min_dist - tol):
        return False
    return True

def _total_radius(pos_r):
    return np.sum(pos_r[:, 2])

def _smooth_overlap_penalty(pos_r, w, h, eps=1e-4):
    """Smooth penalty for overlaps and boundary violations."""
    xs, ys, rs = pos_r[:, 0], pos_r[:, 1], pos_r[:, 2]
    
    # Boundary penalty
    pen = 0.0
    pen += np.sum(np.maximum(0, rs - xs)**2)
    pen += np.sum(np.maximum(0, xs + rs - w)**2)
    pen += np.sum(np.maximum(0, rs - ys)**2)
    pen += np.sum(np.maximum(0, ys + rs - h)**2)
    
    # Overlap penalty (smooth)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    min_dist = rs[:, None] + rs[None, :]
    overlap = np.maximum(0, min_dist - dist)
    pen += np.sum(overlap**2)
    
    return pen

def _relax_circles(circles, w, h, max_iters=500):
    """Relaxation to resolve overlaps and boundary violations."""
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        # Boundary
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r; moved = True
            if x + r > w:
                pos[i, 0] = w - r; moved = True
            if y - r < 0:
                pos[i, 1] = r; moved = True
            if y + r > h:
                pos[i, 1] = h - r; moved = True
        # Pairwise
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
                    else:
                        overlap = min_dist - dist
                        nx, ny = dx/dist, dy/dist
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

def _make_hexagonal_grid(w, h, n_circles):
    """Generate hexagonal grid initialization."""
    circles = []
    # Try different hexagonal densities
    for spacing in [0.15, 0.18, 0.20, 0.22, 0.25]:
        r = spacing / 2 * 0.9
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        count = 0
        for row in range(int(h / dy) + 2):
            y = row * dy
            if y + r > h or y - r < 0:
                continue
            offset = (row % 2) * dx / 2
            for col in range(int(w / dx) + 2):
                x = col * dx + offset
                if x + r > w or x - r < 0:
                    continue
                if count >= n_circles:
                    break
                circles.append([x, y, r])
                count += 1
            if count >= n_circles:
                break
        if len(circles) >= n_circles:
            circles = circles[:n_circles]
            arr = np.array(circles)
            relaxed = _relax_circles(arr, w, h, max_iters=200)
            if _validate(relaxed, w, h, n_circles):
                return relaxed
    # Fallback to simple grid
    return _make_grid(w, h, n_circles)

def _make_grid(w, h, n_circles, cols=5, rows=5):
    """Generate grid initialization."""
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            r = 0.45 * min(w/cols, h/rows)
            circles[count] = [x, y, r]
            count += 1
    relaxed = _relax_circles(circles, w, h, max_iters=200)
    return relaxed

def _penalty_optimize(circles, w, h, deadline, max_iter=2000, ftol=1e-10):
    """Penalty-based L-BFGS-B optimization with vectorized computations."""
    if time.time() > deadline - 5.0:
        return None
    
    n = len(circles)
    x0 = circles.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        pos_r = x.reshape(-1, 3)
        # Adaptive epsilon for smooth penalty
        eps = max(1e-6, 0.01 * (1 - (time.time() - start) / (deadline - start)))
        return -_total_radius(pos_r) + 1000.0 * _smooth_overlap_penalty(pos_r, w, h, eps)
    
    def gradient(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        pos_r = x.reshape(-1, 3)
        xs, ys, rs = pos_r[:, 0], pos_r[:, 1], pos_r[:, 2]
        grad = np.zeros_like(x)
        
        # Gradient of -sum(r)
        grad[2::3] = -1.0
        
        # Gradient of boundary penalty
        # d/dx: 2*max(0, r-x)*(-1) for x-r<0, 2*max(0, x+r-w)*1 for x+r>w
        grad[0::3] += -2 * np.maximum(0, rs - xs)
        grad[0::3] += 2 * np.maximum(0, xs + rs - w)
        grad[1::3] += -2 * np.maximum(0, rs - ys)
        grad[1::3] += 2 * np.maximum(0, ys + rs - h)
        # d/dr for boundary
        grad[2::3] += 2 * np.maximum(0, rs - xs)
        grad[2::3] += 2 * np.maximum(0, xs + rs - w)
        grad[2::3] += 2 * np.maximum(0, rs - ys)
        grad[2::3] += 2 * np.maximum(0, ys + rs - h)
        
        # Gradient of overlap penalty
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = rs[:, None] + rs[None, :]
        overlap = np.maximum(0, min_dist - dist)
        
        # d/dx_i: sum_j 2*overlap_ij * (x_i - x_j)/dist_ij
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if overlap[i, j] > 0:
                    grad[i*3] += 2 * overlap[i, j] * (xs[i] - xs[j]) / dist[i, j]
                    grad[i*3+1] += 2 * overlap[i, j] * (ys[i] - ys[j]) / dist[i, j]
                    grad[i*3+2] += 2 * overlap[i, j]
        
        return grad
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))
    
    start = time.time()
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8}
        )
        new_circles = res.x.reshape(-1, 3)
        if _validate(new_circles, w, h, n_circles):
            return new_circles
        fixed = _relax_circles(new_circles, w, h, max_iters=100)
        if _validate(fixed, w, h, n_circles):
            return fixed
    except Exception:
        pass
    return None

def _high_precision_polish(circles, w, h, deadline):
    """High-precision polish with tighter tolerances."""
    if time.time() > deadline - 3.0:
        return circles
    
    n = len(circles)
    x0 = circles.flatten()
    
    def objective(x):
        pos_r = x.reshape(-1, 3)
        return -_total_radius(pos_r) + 1e6 * _smooth_overlap_penalty(pos_r, w, h, 1e-8)
    
    def gradient(x):
        pos_r = x.reshape(-1, 3)
        xs, ys, rs = pos_r[:, 0], pos_r[:, 1], pos_r[:, 2]
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        grad[0::3] += -2 * np.maximum(0, rs - xs)
        grad[0::3] += 2 * np.maximum(0, xs + rs - w)
        grad[1::3] += -2 * np.maximum(0, rs - ys)
        grad[1::3] += 2 * np.maximum(0, ys + rs - h)
        grad[2::3] += 2 * np.maximum(0, rs - xs)
        grad[2::3] += 2 * np.maximum(0, xs + rs - w)
        grad[2::3] += 2 * np.maximum(0, rs - ys)
        grad[2::3] += 2 * np.maximum(0, ys + rs - h)
        
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = rs[:, None] + rs[None, :]
        overlap = np.maximum(0, min_dist - dist)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if overlap[i, j] > 0:
                    grad[i*3] += 2 * overlap[i, j] * (xs[i] - xs[j]) / dist[i, j]
                    grad[i*3+1] += 2 * overlap[i, j] * (ys[i] - ys[j]) / dist[i, j]
                    grad[i*3+2] += 2 * overlap[i, j]
        return grad
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h)/2))
    
    try:
        res = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        new_circles = res.x.reshape(-1, 3)
        if _validate(new_circles, w, h, n_circles):
            return new_circles
    except Exception:
        pass
    return circles

def _solve_aspect(w, h, deadline):
    """Solve for a specific aspect ratio."""
    # Try hexagonal grid first
    init = _make_hexagonal_grid(w, h, N)
    
    if not _validate(init, w, h, N):
        init = _make_grid(w, h, N)
        if not _validate(init, w, h, N):
            return None
    
    # Penalty optimization
    if time.time() < deadline - 15.0:
        optimized = _penalty_optimize(init, w, h, deadline, max_iter=1500)
        if optimized is not None:
            init = optimized
    
    # High-precision polish
    if time.time() < deadline - 5.0:
        polished = _high_precision_polish(init, w, h, deadline)
        if _validate(polished, w, h, N) and _total_radius(polished) >= _total_radius(init):
            init = polished
    
    return init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_result = None
    best_score = 0.0
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 0.75, 1.33, 0.67]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        result = _solve_aspect(w, h, deadline)
        
        if result is not None and _validate(result, w, h, N):
            score = _total_radius(result)
            if score > best_score:
                best_score = score
                best_result = result
    
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _make_grid(w, h, N)
        if not _validate(best_result, w, h, N):
            for i in range(N):
                x, y, r = best_result[i]
                min_d = min(x, w-x, y, h-y)
                best_result[i, 2] = min(r, min_d - 1e-4)
            best_result = _relax_circles(best_result, w, h, max_iters=100)
    
    return best_result
