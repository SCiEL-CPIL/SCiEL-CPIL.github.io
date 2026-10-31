"""Circle packing in perimeter-4 rectangle: force-directed with global optimization."""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 21
_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < _DEADLINE - 5.0

def _validate(circles, tol=1e-9):
    """
    Check if circles are valid: inside rectangle and non-overlapping.
    circles: (n, 3) array of [cx, cy, r]
    Returns (is_valid, sum_radii, W, H)
    """
    if circles is None or len(circles) != N:
        return False, 0.0, 0.0, 0.0
    
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Compute bounding box
    x_min = np.min(cx - r)
    x_max = np.max(cx + r)
    y_min = np.min(cy - r)
    y_max = np.max(cy + r)
    
    W = x_max - x_min
    H = y_max - y_min
    perimeter = 2.0 * (W + H)
    
    if perimeter > 4.0 + tol:
        return False, 0.0, W, H
    
    # Check all radii positive
    if np.any(r < -tol):
        return False, 0.0, W, H
    
    # Check non-overlapping
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d < r[i] + r[j] - tol:
                return False, 0.0, W, H
    
    return True, float(np.sum(r)), W, H

def _scale_to_perimeter(circles):
    """
    Scale circles so that their bounding box has perimeter exactly 4.
    Returns scaled circles.
    """
    cx = circles[:, 0].copy()
    cy = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    x_min = np.min(cx - r)
    x_max = np.max(cx + r)
    y_min = np.min(cy - r)
    y_max = np.max(cy + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if W <= 0 or H <= 0:
        return circles
    
    perimeter = 2.0 * (W + H)
    if perimeter <= 0:
        return circles
    
    scale = 4.0 / perimeter
    
    # Scale positions and radii
    cx = (cx - x_min) * scale
    cy = (cy - y_min) * scale
    r = r * scale
    
    return np.column_stack([cx, cy, r])

def _force_directed_relax(circles, W, H, n_iter=200, dt=0.001, tol=1e-10):
    """
    Force-directed relaxation to maximize sum of radii while maintaining
    non-overlapping and boundary constraints.
    
    circles: (n, 3) array
    W, H: rectangle dimensions
    """
    cx = circles[:, 0].copy()
    cy = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    for it in range(n_iter):
        if not _time_ok():
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    dx = random.uniform(-1e-3, 1e-3)
                    dy = random.uniform(-1e-3, 1e-3)
                    d = np.hypot(dx, dy)
                
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    # Push apart
                    inv_d = 1.0 / d
                    fx[i] += overlap * dx * inv_d
                    fy[i] += overlap * dy * inv_d
                    fx[j] -= overlap * dx * inv_d
                    fy[j] -= overlap * dy * inv_d
                    # Grow radii
                    fr[i] += overlap * 0.5
                    fr[j] += overlap * 0.5
        
        # Boundary forces: push circles away from walls
        for i in range(N):
            # Left wall
            dist_left = cx[i] - r[i]
            if dist_left < 0:
                fx[i] += -dist_left * 5.0
                fr[i] += -dist_left * 0.5
            # Right wall
            dist_right = (W - cx[i]) - r[i]
            if dist_right < 0:
                fx[i] += -dist_right * 5.0
                fr[i] += -dist_right * 0.5
            # Bottom wall
            dist_bottom = cy[i] - r[i]
            if dist_bottom < 0:
                fy[i] += -dist_bottom * 5.0
                fr[i] += -dist_bottom * 0.5
            # Top wall
            dist_top = (H - cy[i]) - r[i]
            if dist_top < 0:
                fy[i] += -dist_top * 5.0
                fr[i] += -dist_top * 0.5
        
        # Update positions and radii
        cx += dt * fx
        cy += dt * fy
        r += dt * fr
        
        # Clamp to valid region
        cx = np.clip(cx, 0, W)
        cy = np.clip(cy, 0, H)
        r = np.maximum(r, 1e-8)
        
        # Enforce boundary constraints on radii
        for i in range(N):
            r_max = min(cx[i], W - cx[i], cy[i], H - cy[i])
            r[i] = min(r[i], r_max)
        
        # Enforce non-overlapping by shrinking if needed
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < r[i] + r[j]:
                    # Scale down both radii proportionally
                    total_r = r[i] + r[j]
                    if total_r > 1e-12:
                        scale = d / total_r
                        r[i] *= scale
                        r[j] *= scale
    
    return np.column_stack([cx, cy, r])

def _penalty_objective(params, W, H):
    """
    Objective function for optimization: minimize negative sum of radii
    with penalties for violations.
    """
    n = N
    cx = params[:n]
    cy = params[n:2*n]
    r = params[2*n:3*n]
    
    penalty = 0.0
    
    # Boundary penalties
    for i in range(n):
        if r[i] < 0:
            penalty += abs(r[i]) * 100
        if cx[i] - r[i] < 0:
            penalty += (cx[i] - r[i]) * 50
        if cx[i] + r[i] > W:
            penalty += (cx[i] + r[i] - W) * 50
        if cy[i] - r[i] < 0:
            penalty += (cy[i] - r[i]) * 50
        if cy[i] + r[i] > H:
            penalty += (cy[i] + r[i] - H) * 50
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            overlap = r[i] + r[j] - d
            if overlap > 0:
                penalty += overlap * 100
    
    return -np.sum(r) + penalty

def _optimize_for_aspect(W, H, x0, n_relax=100):
    """
    Optimize a single layout for given W, H.
    """
    cx = x0[:, 0].copy()
    cy = x0[:, 1].copy()
    r = x0[:, 2].copy()
    
    # Ensure initial validity by clamping
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    r = np.maximum(r, 1e-6)
    for i in range(N):
        r_bound = min(cx[i], W - cx[i], cy[i], H - cy[i])
        r[i] = min(r[i], r_bound)
    
    circles = np.column_stack([cx, cy, r])
    
    # Force-directed relaxation
    circles = _force_directed_relax(circles, W, H, n_iter=n_relax)
    
    # Validate
    valid, score, _, _ = _validate(circles)
    if not valid:
        # Try to fix by scaling
        circles = _scale_to_perimeter(circles)
        valid, score, _, _ = _validate(circles)
        if not valid:
            return None, 0.0
    
    return circles, score

def _hexagonal_layout(W, H):
    """Create a hexagonal packing layout."""
    pts = []
    base_r = min(W, H) / 5.0
    row_h = base_r * np.sqrt(3)
    n_rows = int(H / row_h) + 2
    n_cols = int(W / (2 * base_r)) + 2
    
    for j in range(n_rows):
        y = j * row_h
        if y > H:
            break
        offset = (j % 2) * base_r
        for i in range(n_cols):
            x = i * 2 * base_r + offset
            if x > W:
                break
            r_max = min(x, W - x, y, H - y)
            if r_max > base_r * 0.5:
                r = base_r * 0.95
                r = min(r, r_max)
                if r > 1e-4:
                    pts.append([x, y, r])
    
    if len(pts) == 0:
        # Fallback grid
        kx, ky = 5, 5
        for j in range(ky):
            for i in range(kx):
                cx = (i + 0.5) * W / kx
                cy = (j + 0.5) * H / ky
                r = min(cx, cy, W - cx, H - cy) * 0.9
                r = max(r, 1e-4)
                pts.append([cx, cy, r])
    
    arr = np.array(pts)
    order = np.argsort(-arr[:, 2])[:N]
    arr = arr[order]
    
    if len(arr) < N:
        for _ in range(N - len(arr)):
            arr = np.vstack([arr, [W/2, H/2, 0.01]])
    return arr[:N]

def _make_grid(kx, ky, W, H):
    """Create a kx by ky grid of circles, take first N."""
    pts = []
    for j in range(ky):
        for i in range(kx):
            cx = (i + 0.5) * W / kx
            cy = (j + 0.5) * H / ky
            r = min(cx, cy, W - cx, H - cy) * 0.9
            r = max(r, 1e-4)
            pts.append([cx, cy, r])
    arr = np.array(pts)
    if len(arr) < N:
        for _ in range(N - len(arr)):
            arr = np.vstack([arr, [W/2, H/2, 0.01]])
    return arr[:N]

def construct():
    best_score = 0.0
    best_out = None
    
    if not _time_ok():
        # Fallback: simple grid
        W, H = 1.0, 1.0
        g = _make_grid(5, 5, W, H)
        return g
    
    random.seed(42)
    np.random.seed(42)
    
    # Try various aspect ratios
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        if H <= 0:
            continue
        
        candidates = []
        
        # Grid layouts
        grid_configs = [(4, 5), (5, 4), (3, 7), (7, 3), (5, 5), (4, 6), (6, 4), (3, 6), (6, 3)]
        for kx, ky in grid_configs:
            if kx * ky < N:
                continue
            g = _make_grid(kx, ky, W, H)
            candidates.append(g)
        
        # Hexagonal layout
        candidates.append(_hexagonal_layout(W, H))
        
        # Random layouts
        for _ in range(3):
            cx = np.random.uniform(0.1, W * 0.9, N)
            cy = np.random.uniform(0.1, H * 0.9, N)
            r = np.ones(N) * min(W, H) / 8.0
            r = np.minimum(r, np.minimum(np.minimum(cx, W - cx), np.minimum(cy, H - cy)))
            r = np.maximum(r, 1e-4)
            candidates.append(np.column_stack([cx, cy, r]))
        
        # Optimize each candidate
        for g in candidates:
            if not _time_ok():
                break
            out, score = _optimize_for_aspect(W, H, g)
            if out is not None and score > best_score:
                best_score = score
                best_out = out
    
    # If we found a good solution, do perturbation restarts on best
    if best_out is not None and _time_ok():
        # Determine W, H from best_out
        cx = best_out[:, 0]
        cy = best_out[:, 1]
        r = best_out[:, 2]
        W = np.max(cx + r) - np.min(cx - r)
        H = np.max(cy + r) - np.min(cy - r)
        
        for _ in range(5):
            if not _time_ok():
                break
            noise = np.random.randn(*best_out.shape) * 0.03
            perturbed = best_out + noise
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, W)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, H)
            perturbed[:, 2] = np.maximum(perturbed[:, 2], 1e-4)
            out, score = _optimize_for_aspect(W, H, perturbed)
            if out is not None and score > best_score:
                best_score = score
                best_out = out
    
    # Final fallback
    if best_out is None:
        W, H = 1.0, 1.0
        g = _make_grid(5, 5, W, H)
        best_out = g
    
    return best_out
