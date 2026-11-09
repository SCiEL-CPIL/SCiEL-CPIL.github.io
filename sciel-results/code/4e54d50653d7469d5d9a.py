# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    half = sides / 2.0
    if np.any(centers[:, 0] - half < -tol) or np.any(centers[:, 0] + half > 1 + tol):
        return False
    if np.any(centers[:, 1] - half < -tol) or np.any(centers[:, 1] + half > 1 + tol):
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 axis-aligned squares in the unit square.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # 20-second safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    tol = 1e-7
    
    def objective(params):
        """
        params: [s_1,...,s_10, cx_1,...,cx_10, cy_1,...,cy_10]
        Returns: -sum(s) + penalty
        """
        sides = params[:10]
        cx = params[10:20]
        cy = params[20:30]
        
        if np.any(sides <= 1e-8):
            return 1e12
        
        penalty = 0.0
        half = sides / 2.0
        
        # Boundary violations
        lo_x = cx - half
        hi_x = cx + half
        lo_y = cy - half
        hi_y = cy + half
        
        penalty += np.sum(np.maximum(0, -tol - lo_x)) * 500
        penalty += np.sum(np.maximum(0, hi_x - (1 + tol))) * 500
        penalty += np.sum(np.maximum(0, -tol - lo_y)) * 500
        penalty += np.sum(np.maximum(0, hi_y - (1 + tol))) * 500
        
        # Overlap penalties
        for i in range(10):
            for j in range(i + 1, 10):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                ox = np.maximum(0, t - dx)
                oy = np.maximum(0, t - dy)
                if ox > 0 and oy > 0:
                    penalty += ox * oy * 500
        
        return -np.sum(sides) + penalty
    
    def try_optimize(params_init, label=""):
        nonlocal best_centers, best_sides, best_score
        if time.time() > deadline:
            return
        try:
            result = minimize(
                objective,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            params_opt = result.x
            sides_opt = np.maximum(params_opt[:10], 1e-6)
            cx_opt = np.clip(params_opt[10:20], 0, 1)
            cy_opt = np.clip(params_opt[20:30], 0, 1)
            centers_opt = np.stack([cx_opt, cy_opt], axis=1)
            
            if _is_valid(centers_opt, sides_opt):
                score = np.sum(sides_opt)
                if score > best_score:
                    best_score = score
                    best_centers = centers_opt.copy()
                    best_sides = sides_opt.copy()
        except Exception:
            pass
    
    # Initial configuration 1: 4x4 grid (6 filled)
    k = 4
    side = 1.0 / k
    idx = np.arange(10)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    sides0 = np.full(10, side)
    params0 = np.concatenate([sides0, cx0, cy0])
    try_optimize(params0, "grid4x4")
    
    # Initial configuration 2: 3x3 grid with one extra
    # 3x3 = 9 squares, add 10th
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx1 = (cols + 0.5) * side
    cy1 = (rows + 0.5) * side
    sides1 = np.full(9, side)
    # 10th square: small, in a corner
    cx1 = np.append(cx1, 0.5)
    cy1 = np.append(cy1, 0.5)
    sides1 = np.append(sides1, 0.1)
    params1 = np.concatenate([sides1, cx1, cy1])
    try_optimize(params1, "grid3x3+1")
    
    # Initial configuration 3: Two large + eight small
    # Two large squares of side 0.45, eight small of side ~0.13
    sides2 = np.array([0.45, 0.45] + [0.13]*8)
    cx2 = np.array([0.225, 0.775] + [0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7])
    cy2 = np.array([0.225, 0.775] + [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    params2 = np.concatenate([sides2, cx2, cy2])
    try_optimize(params2, "2large+8small")
    
    # Initial configuration 4: One very large + nine small
    sides3 = np.array([0.5] + [0.12]*9)
    cx3 = np.array([0.25] + [0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
    cy3 = np.array([0.75] + [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    params3 = np.concatenate([sides3, cx3, cy3])
    try_optimize(params3, "1large+9small")
    
    # Initial configuration 5: Random with seed
    rng = np.random.RandomState(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        sides_r = rng.uniform(0.1, 0.35, size=10)
        cx_r = rng.uniform(0.15, 0.85, size=10)
        cy_r = rng.uniform(0.15, 0.85, size=10)
        params_r = np.concatenate([sides_r, cx_r, cy_r])
        try_optimize(params_r, f"random{restart}")
    
    # If no valid solution found, return the grid baseline
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
