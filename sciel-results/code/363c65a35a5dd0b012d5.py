"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

# Deadline guard: kill after 140s (10s safety margin)
_DEADLINE = time.time() + 140


def _check_deadline():
    return time.time() > _DEADLINE


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (i.e., 2*(W+H)=4 => W+H=2).
    Maximize sum of radii.
    
    Strategy:
    1. Start with a good initial grid-like placement.
    2. Use iterative local optimization:
       - Fix rectangle dimensions (W, H) with W+H=2.
       - Optimize circle centers and radii via SLSQP with:
         * Non-overlap constraints: dist(i,j) >= r_i + r_j
         * Wall constraints: r_i <= x_i, r_i <= W-x_i, r_i <= y_i, r_i <= H-y_i
         * All radii > 0
    3. Also optimize W (and H=2-W) as part of the variables.
    4. Use multiple restarts with different initial configurations.
    """
    best_value = -np.inf
    best_result = None
    
    # Try a range of aspect ratios (W from 0.8 to 1.2, since W+H=2)
    # Symmetric around W=1 (square)
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial configurations to try
    configs = []
    
    # Config 1: 5x5 grid, take 21
    for aspect in [0.9, 1.0, 1.1]:
        W = aspect
        H = 2.0 - aspect
        # 5x5 grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = 0.5 * min(W, H) / side - 0.001
                cells.append([cx, cy, r])
        configs.append(cells[:N])
    
    # Config 2: 4x6 grid (24 cells, take 21)
    for aspect in [0.9, 1.0, 1.1]:
        W = aspect
        H = 2.0 - aspect
        nx, ny = 4, 6
        cells = []
        for i in range(nx):
            for j in range(ny):
                cx = (i + 0.5) * W / nx
                cy = (j + 0.5) * H / ny
                r = 0.5 * min(W/nx, H/ny) - 0.001
                cells.append([cx, cy, r])
        configs.append(cells[:N])
    
    # Config 3: 3x7 grid (21 cells exactly)
    for aspect in [0.8, 0.9, 1.0, 1.1, 1.2]:
        W = aspect
        H = 2.0 - aspect
        nx, ny = 3, 7
        cells = []
        for i in range(nx):
            for j in range(ny):
                cx = (i + 0.5) * W / nx
                cy = (j + 0.5) * H / ny
                r = 0.5 * min(W/nx, H/ny) - 0.001
                cells.append([cx, cy, r])
        configs.append(cells[:N])
    
    # Config 4: Random perturbation of 5x5
    for _ in range(3):
        W = 1.0
        H = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * W / side + np.random.normal(0, 0.02)
                cy = (j + 0.5) * H / side + np.random.normal(0, 0.02)
                r = 0.5 * min(W, H) / side - 0.002
                cells.append([cx, cy, r])
        configs.append(cells[:N])
    
    def _evaluate(circles, W, H):
        """Check validity and compute sum of radii."""
        circles = np.array(circles)
        # Check wall constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if r <= 0:
                return -np.inf
            if x - r < -1e-9 or x + r > W + 1e-9 or y - r < -1e-9 or y + r > H + 1e-9:
                return -np.inf
        # Check overlap
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                    return -np.inf
        return np.sum(circles[:, 2])
    
    def _optimize_single(circles_init, W_init, H_init):
        """Optimize circles and rectangle width for a single starting config."""
        circles_init = np.array(circles_init, dtype=float)
        
        # Variables: W, then for each circle: x, y, r
        # Total: 1 + 3*N
        x0 = np.concatenate([[W_init], circles_init.flatten()])
        
        def _unflatten(x):
            W = x[0]
            H = 2.0 - W
            circles = x[1:].reshape(N, 3)
            return W, H, circles
        
        def _objective(x):
            W, H, circles = _unflatten(x)
            # Penalize invalid configurations heavily
            penalty = 0.0
            for i in range(N):
                cx, cy, r = circles[i]
                if r <= 0:
                    penalty += 100.0
                if cx - r < 0:
                    penalty += 100.0 * (cx - r)**2
                if cx + r > W:
                    penalty += 100.0 * (cx + r - W)**2
                if cy - r < 0:
                    penalty += 100.0 * (cy - r)**2
                if cy + r > H:
                    penalty += 100.0 * (cy + r - H)**2
            for i in range(N):
                for j in range(i+1, N):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i, 2] + circles[j, 2]
                    if dist < min_dist:
                        penalty += 100.0 * (min_dist - dist)**2
            return -np.sum(circles[:, 2]) + penalty
        
        # Bounds
        bounds = [(0.5, 1.5)]  # W
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((1e-4, 0.5))  # r
        
        # Run SLSQP
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-8})
            if res.success or res.fun < 0:  # Accept if it improved
                W, H, circles = _unflatten(res.x)
                val = _evaluate(circles, W, H)
                if val > 0:
                    return circles, W, H, val
        except Exception:
            pass
        
        return None
    
    # Try each config
    for idx, cfg in enumerate(configs):
        if _check_deadline():
            break
        
        W_init = 1.0
        H_init = 1.0
        
        result = _optimize_single(cfg, W_init, H_init)
        if result is not None:
            circles, W, H, val = result
            if val > best_value:
                best_value = val
                best_result = (circles, W, H)
    
    # If no valid result, fall back to a simple grid
    if best_result is None:
        W, H = 1.0, 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = 0.5 * min(W, H) / side - 0.001
                cells.append([cx, cy, r])
        best_result = (np.array(cells[:N]), W, H)
        best_value = _evaluate(best_result[0], W, H)
    
    circles, W, H = best_result
    
    # Final verification and formatting
    # Ensure all circles are valid
    result = []
    for i in range(N):
        x, y, r = circles[i]
        # Clamp to be safe
        r = max(r, 1e-6)
        x = min(max(x, r), W - r)
        y = min(max(y, r), H - r)
        result.append([x, y, r])
    
    result = np.array(result)
    
    # Final check: if invalid, return a known-good grid
    val = _evaluate(result, W, H)
    if val <= 0:
        W, H = 1.0, 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = 0.5 * min(W, H) / side - 0.001
                cells.append([cx, cy, r])
        result = np.array(cells[:N])
    
    return result
# EVOLVE-BLOCK-END
