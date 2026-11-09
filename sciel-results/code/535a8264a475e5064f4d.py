# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize total side length.
    Uses multi-start optimization with diverse initial configurations.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 580  # safety margin before 630s limit
    
    def objective(params):
        """Negative sum of sides (to minimize)."""
        cx = params[:n]
        cy = params[n:2*n]
        s = params[2*n:]
        
        # Boundary violations
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return 1e6
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return 1e6
        
        # Overlap violations
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return 1e6
        
        return -np.sum(s)
    
    def objective_smooth(params):
        """Smooth objective with soft constraints."""
        cx = params[:n]
        cy = params[n:2*n]
        s = np.maximum(params[2*n:], 1e-10)
        
        penalty = 0.0
        # Boundary penalties
        for i in range(n):
            penalty += max(0, -(cx[i] - s[i]/2) - tol)**2
            penalty += max(0, (cx[i] + s[i]/2) - 1 - tol)**2
            penalty += max(0, -(cy[i] - s[i]/2) - tol)**2
            penalty += max(0, (cy[i] + s[i]/2) - 1 - tol)**2
        
        # Overlap penalties (smooth)
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                dx = abs(cx[i] - cx[j]) - t
                dy = abs(cy[i] - cy[j]) - t
                # Penalize when both dx < 0 and dy < 0 (overlap)
                overlap_x = max(0, -dx)
                overlap_y = max(0, -dy)
                penalty += overlap_x * overlap_y
        
        return -np.sum(s) + 1000.0 * penalty
    
    def is_valid(params):
        """Check if configuration is valid."""
        cx = params[:n]
        cy = params[n:2*n]
        s = params[2*n:]
        
        if not np.all(np.isfinite(params)):
            return False
        if np.any(s <= 0):
            return False
        
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True
    
    def grid_init():
        """Simple grid initialization."""
        k = 4  # 4x4 grid, use 10 cells
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        return np.concatenate([cx, cy, s])
    
    def big_small_init(n_big=2, n_small=8):
        """Two big squares and 8 small ones."""
        # Two big squares in top half
        big_side = 0.4
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        # Place 2 big squares
        cx[0] = 0.25
        cy[0] = 0.75
        s[0] = big_side
        cx[1] = 0.75
        cy[1] = 0.75
        s[1] = big_side
        
        # Place 8 small squares in bottom half
        small_side = 0.25
        for i in range(n_small):
            col = i % 4
            row = i // 4
            cx[2+i] = (col + 0.5) * small_side + 0.0
            cy[2+i] = (row + 0.5) * small_side + 0.0
            s[2+i] = small_side
        
        return np.concatenate([cx, cy, s])
    
    def three_big_init():
        """Three big squares and 7 small ones."""
        big_side = 0.35
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        
        cx[0] = 0.2
        cy[0] = 0.7
        s[0] = big_side
        cx[1] = 0.8
        cy[1] = 0.7
        s[1] = big_side
        cx[2] = 0.5
        cy[2] = 0.3
        s[2] = big_side
        
        small_side = 0.2
        for i in range(7):
            col = i % 4
            row = i // 4
            cx[3+i] = (col + 0.5) * small_side
            cy[3+i] = (row + 0.5) * small_side + 0.55
            s[3+i] = small_side
        
        return np.concatenate([cx, cy, s])
    
    def random_init():
        """Random valid-ish initialization."""
        cx = np.random.uniform(0.1, 0.9, n)
        cy = np.random.uniform(0.1, 0.9, n)
        s = np.random.uniform(0.1, 0.3, n)
        return np.concatenate([cx, cy, s])
    
    def polish(params):
        """High-precision polish using L-BFGS-B."""
        bounds = []
        for i in range(n):
            bounds.append((1e-4, 0.999))  # cx
        for i in range(n):
            bounds.append((1e-4, 0.999))  # cy
        for i in range(n):
            bounds.append((1e-4, 0.999))  # s
        
        result = minimize(
            objective_smooth,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success and is_valid(result.x):
            return result.x
        return params
    
    best_params = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    inits = [
        grid_init(),
        big_small_init(2, 8),
        big_small_init(3, 7),
        three_big_init(),
    ]
    
    # Add random inits
    for _ in range(20):
        inits.append(random_init())
    
    # Shuffle and try each
    np.random.seed(42)
    np.random.shuffle(inits)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Try Nelder-Mead
        try:
            result = minimize(
                objective_smooth,
                init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.success and is_valid(result.x):
                score = np.sum(result.x[2*n:])
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except:
            pass
        
        # Try L-BFGS-B polish
        if time.time() > deadline:
            break
        try:
            polished = polish(init)
            if is_valid(polished):
                score = np.sum(polished[2*n:])
                if score > best_score:
                    best_score = score
                    best_params = polished.copy()
        except:
            pass
    
    # If we have a good result, do a final high-precision polish
    if best_params is not None and time.time() < deadline - 60:
        try:
            final = polish(best_params)
            if is_valid(final):
                score = np.sum(final[2*n:])
                if score > best_score:
                    best_score = score
                    best_params = final.copy()
        except:
            pass
    
    # Fallback to grid if nothing found
    if best_params is None:
        best_params = grid_init()
    
    cx = best_params[:n]
    cy = best_params[n:2*n]
    s = best_params[2*n:]
    
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
