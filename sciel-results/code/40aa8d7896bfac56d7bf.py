# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _relax_packing(x, y, r, w, h, n_iters=2000, k_repel=100.0, k_spring=10.0, k_wall=50.0):
    """
    Relax the packing to resolve overlaps and enforce contact constraints.
    Returns updated x, y.
    """
    N = len(x)
    x = x.copy()
    y = y.copy()
    
    for it in range(n_iters):
        # Adaptive learning rate
        lr = 0.01 * (0.995 ** it)
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Overlapping: Strong repulsion
                    overlap = min_dist - dist
                    force = k_repel * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist * 1.02:
                    # Near contact: Spring force to maintain distance
                    # This helps rigidify the structure
                    diff = dist - min_dist
                    force = k_spring * diff
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Wall forces
        for i in range(N):
            if x[i] < r[i]:
                pen = r[i] - x[i]
                fx[i] += k_wall * pen
            elif x[i] > w - r[i]:
                pen = x[i] - (w - r[i])
                fx[i] -= k_wall * pen
            
            if y[i] < r[i]:
                pen = r[i] - y[i]
                fy[i] += k_wall * pen
            elif y[i] > h - r[i]:
                pen = y[i] - (h - r[i])
                fy[i] -= k_wall * pen
        
        # Update positions
        x += lr * fx
        y += lr * fy
        
        # Hard clamp to ensure we don't go too far outside
        for i in range(N):
            x[i] = np.clip(x[i], 0, w)
            y[i] = np.clip(y[i], 0, h)
            
    return x, y

def _is_valid(x, y, r, w, h, tol=1e-6):
    N = len(x)
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol * 10:
                return False
    return True

def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    """
    N = 21
    best_val = -1
    best_result = None
    
    # Perimeter 4 => w + h = 2
    # Try a range of aspect ratios
    w_values = np.linspace(0.8, 1.2, 10)
    
    for w in w_values:
        h = 2.0 - w
        
        # Initialize positions in a grid
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            col = i % cols
            row = i // cols
            x[i] = (col + 0.5) * w / cols
            y[i] = (row + 0.5) * h / rows
            # Initial radius estimate
            r[i] = min(w / cols, h / rows) * 0.45
            
        # Grow radii iteratively
        growth_steps = 100
        base_r = r.copy()
        
        for step in range(growth_steps):
            # Scale radii up
            scale = 1.0 + 0.05 * (1.0 - step / growth_steps) # Larger steps early
            r_new = base_r * (1.0 + 0.02) # Fixed small growth
            
            # Relax positions with new radii
            x, y = _relax_packing(x, y, r_new, w, h, n_iters=500)
            
            # Check validity
            if _is_valid(x, y, r_new, w, h):
                base_r = r_new
                val = np.sum(r_new)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x, y, r_new])
            else:
                # Try to recover by reducing radius slightly and relaxing harder
                r_try = base_r * 0.99
                x, y = _relax_packing(x, y, r_try, w, h, n_iters=1000)
                if _is_valid(x, y, r_try, w, h):
                    base_r = r_try
                else:
                    break # Stuck, try next aspect ratio
                    
    if best_result is not None:
        return best_result
    
    # Fallback
    side = 5
    cells = []
    for i in range(N):
        col = i % side
        row = i // side
        w_cell = 1.0 / side
        h_cell = 1.0 / side
        cells.append([
            (col + 0.5) * w_cell,
            (row + 0.5) * h_cell,
            0.4 * min(w_cell, h_cell)
        ])
    return np.array(cells)
# EVOLVE-BLOCK-END
