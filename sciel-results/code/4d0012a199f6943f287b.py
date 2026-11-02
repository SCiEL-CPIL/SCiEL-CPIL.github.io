import numpy as np
import time

N = 300

def _hexagonal_initial(n_points):
    """Generate a hexagonal lattice initial configuration."""
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    k = int(1.0 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append((x, y))
    points = np.array(points)
    if len(points) < n_points:
        return np.random.randn(n_points, 2) * 0.5
    else:
        radius = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radius, kind='stable')[:n_points]
        return points[idx]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    pts = _hexagonal_initial(N)
    pts = pts + np.random.randn(N, 2) * 0.005
    
    beta = 50.0
    lr = 0.005
    best_r = -np.inf
    best_pts = pts.copy()
    
    prev_S = -np.inf
    
    while time.time() - start_time < deadline - 5.0:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        D = np.linalg.norm(diff, axis=2)
        norms = np.linalg.norm(pts, axis=1)
        B = 1.0 - norms
        
        R_pairs = D / 2.0
        np.fill_diagonal(R_pairs, np.inf)
        R_bounds = B
        
        all_R_flat = np.concatenate([R_pairs.ravel(), R_bounds])
        all_R_flat = all_R_flat[all_R_flat < np.inf]
        if len(all_R_flat) == 0:
            break
        r_min = all_R_flat.min()
        
        # Shift for stability
        exp_pairs = np.exp(-beta * (R_pairs - r_min))
        np.fill_diagonal(exp_pairs, 0.0)
        exp_bounds = np.exp(-beta * (R_bounds - r_min))
        
        sum_exp = exp_pairs.sum() + exp_bounds.sum()
        if sum_exp == 0 or not np.isfinite(sum_exp):
            break
            
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        S_val = r_min - np.log(sum_exp) / beta
        
        # Gradient
        D_safe = np.where(D > 1e-12, D, 1.0)
        unit_diff = diff / D_safe[:, :, np.newaxis]
        
        grad_pairs = np.sum(w_pairs[:, :, np.newaxis] * unit_diff / 2.0, axis=1)
        
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        grad_bounds = -w_bounds[:, np.newaxis] * (pts / norms_safe[:, np.newaxis])
        
        grad = grad_pairs + grad_bounds
        
        # Normalize gradient to control step size?
        # Or use fixed lr.
        
        pts = pts + lr * grad
        
        # Check validity and best
        min_pair = D.min() / 2.0
        min_bound = B.min()
        current_r = min(min_pair, min_bound)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
            
        # Adapt beta
        # If progress is slow, increase beta
        if S_val < prev_S:
             # This shouldn't happen often with small lr, but if it does, maybe reduce lr?
             pass
             
        beta *= 1.05
        lr *= 0.99
        
        prev_S = S_val
        
    return best_pts
