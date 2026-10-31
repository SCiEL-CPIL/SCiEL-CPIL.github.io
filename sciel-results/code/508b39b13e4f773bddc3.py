"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    best_value = 0.0
    best_circles = None
    
    def compute_value(xs, ys, rs, w):
        """Compute sum of radii if valid, else -inf."""
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return -np.inf
        if np.any(rs <= 0):
            return -np.inf
        
        # Check boundary constraints
        for i in range(len(rs)):
            if xs[i] < rs[i] - 1e-9 or xs[i] > w - rs[i] + 1e-9:
                return -np.inf
            if ys[i] < rs[i] - 1e-9 or ys[i] > h - rs[i] + 1e-9:
                return -np.inf
        
        # Check pairwise non-overlap
        for i in range(len(rs)):
            for j in range(i+1, len(rs)):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist*min_dist - 1e-12:
                    return -np.inf
        
        return np.sum(rs)
    
    def energy(params, w_scale=1.0):
        """
        params: [x1,y1,r1, x2,y2,r2, ..., x21,y21,r21, w]
        Returns negative value (to minimize) with penalties.
        """
        n = len(params) // 3
        circles = params[:n*3].reshape(n, 3)
        w = params[-1]
        h = 2.0 - w
        
        if w <= 0.1 or h <= 0.1:
            return 1e10
        
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Penalty for negative radii
        r_pen = np.sum(np.maximum(0, -rs) ** 2)
        
        # Boundary penalties (soft)
        b_pen = 0.0
        for i in range(n):
            if xs[i] < rs[i]:
                b_pen += (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]:
                b_pen += (xs[i] - (w - rs[i])) ** 2
            if ys[i] < rs[i]:
                b_pen += (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]:
                b_pen += (ys[i] - (h - rs[i])) ** 2
        
        # Overlap penalties (soft)
        o_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    o_pen += (min_dist - dist) ** 2
        
        total_penalty = r_pen + 100*b_pen + 100*o_pen
        
        if total_penalty > 1e-4:
            return 1e6 + total_penalty
        
        val = np.sum(rs)
        return -val
    
    # Try multiple aspect ratios and optimization strategies
    np.random.seed(12345)
    
    # Strategy 1: Physics-based relaxation with varied aspect ratios
    for trial in range(30):
        # Vary aspect ratio
        if trial < 10:
            w_init = 1.0 + np.random.uniform(-0.2, 0.2)
        elif trial < 20:
            w_init = 1.2 + np.random.uniform(-0.15, 0.15)
        else:
            w_init = 0.8 + np.random.uniform(-0.15, 0.15)
        
        h_init = 2.0 - w_init
        
        # Initialize with hexagonal-like pattern
        params = np.zeros(64)
        
        # Use different grid layouts
        if trial % 3 == 0:
            # 5x5 grid, take 21
            cols, rows = 5, 5
            for i in range(21):
                col = i % cols
                row = i // cols
                x = (col + 0.5) * w_init / cols
                y = (row + 0.5) * h_init / rows
                r = min(w_init / cols, h_init / rows) * 0.48
                params[3*i] = x
                params[3*i+1] = y
                params[3*i+2] = r
        elif trial % 3 == 1:
            # 6x4 grid, take 21
            cols, rows = 6, 4
            for i in range(21):
                col = i % cols
                row = i // cols
                x = (col + 0.5) * w_init / cols
                y = (row + 0.5) * h_init / rows
                r = min(w_init / cols, h_init / rows) * 0.48
                params[3*i] = x
                params[3*i+1] = y
                params[3*i+2] = r
        else:
            # Random initialization
            for i in range(21):
                x = np.random.uniform(0.1, w_init - 0.1)
                y = np.random.uniform(0.1, h_init - 0.1)
                r = np.random.uniform(0.05, min(w_init, h_init) * 0.2)
                params[3*i] = x
                params[3*i+1] = y
                params[3*i+2] = r
        
        params[-1] = w_init
        
        # Optimize
        try:
            result = minimize(energy, params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7,
                                      'adaptive': True})
            val, sol = None, None
            if result.fun < 1e5:
                # Extract and validate
                circles = result.x[:63].reshape(21, 3)
                w_opt = result.x[-1]
                val = compute_value(circles[:, 0], circles[:, 1], circles[:, 2], w_opt)
                if np.isfinite(val) and val > best_value:
                    best_value = val
                    best_circles = circles.copy()
        except Exception:
            pass
    
    # Strategy 2: Start from known good patterns and scale
    # Try a few hand-crafted configurations
    for config_trial in range(10):
        # Square-ish rectangle
        w = 1.0 + 0.1 * (config_trial - 5) / 5.0
        h = 2.0 - w
        
        # Place circles in a pattern that leaves room
        circles = np.zeros((21, 3))
        
        # Use a 5x5 grid but with adjusted radii
        cols, rows = 5, 5
        cell_w = w / cols
        cell_h = h / rows
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= 21:
                    break
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                # Radius limited by both cell dimensions and neighbors
                r = min(cell_w, cell_h) * 0.49
                circles[idx, 0] = x
                circles[idx, 1] = y
                circles[idx, 2] = r
                idx += 1
        
        # Optimize positions and radii
        params = np.zeros(64)
        params[:63] = circles.flatten()
        params[-1] = w
        
        try:
            result = minimize(energy, params, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-10})
            if result.fun < 1e5:
                circles_opt = result.x[:63].reshape(21, 3)
                w_opt = result.x[-1]
                val = compute_value(circles_opt[:, 0], circles_opt[:, 1], circles_opt[:, 2], w_opt)
                if np.isfinite(val) and val > best_value:
                    best_value = val
                    best_circles = circles_opt.copy()
        except Exception:
            pass
    
    # If no valid solution, return fallback
    if best_circles is None:
        # Fallback: 5x5 grid in square
        side = 5
        cells = []
        for i in range(21):
            col = i % side
            row = i // side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.95
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_circles
# EVOLVE-BLOCK-END
