# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Optimizes the packing of 10 axis-aligned squares in the unit square to maximize
    the sum of their side lengths.
    
    Uses a multi-start local optimization strategy with a penalty method to handle
    non-overlap and boundary constraints.
    """
    n = 10
    start_time = time.time()
    deadline = 500.0  # Safety margin before 600s limit
    
    # Baseline: 2x5 grid
    # 2 rows, 5 columns
    side_base = 0.5
    centers_base = np.zeros((n, 2))
    sides_base = np.full(n, side_base)
    idx = np.arange(n)
    cols = idx % 5
    rows = idx // 5
    centers_base[:, 0] = (cols + 0.5) * side_base
    centers_base[:, 1] = (rows + 0.5) * side_base
    best_centers = centers_base.copy()
    best_sides = sides_base.copy()
    best_score = np.sum(sides_base)
    
    def validate(centers, sides):
        """
        Checks if the packing is valid according to the task contract.
        Returns True if valid, False otherwise.
        """
        tol = 1e-7
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if not np.all(sides > 0):
            return False
        
        # Boundary check
        # x_min = cx - s/2, x_max = cx + s/2
        # Must be in [-tol, 1+tol]
        x_mins = centers[:, 0] - sides / 2
        x_maxs = centers[:, 0] + sides / 2
        y_mins = centers[:, 1] - sides / 2
        y_maxs = centers[:, 1] + sides / 2
        
        if np.any(x_mins < -tol) or np.any(x_maxs > 1 + tol):
            return False
        if np.any(y_mins < -tol) or np.any(y_maxs > 1 + tol):
            return False
            
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    return False
        return True

    def objective(params):
        """
        Objective function for optimization.
        params: (n, 3) array of [cx, cy, s]
        Returns: -sum(s) + penalty
        """
        centers = params[:, :2]
        sides = params[:, 2]
        
        # Ensure sides are positive (penalize if not)
        if np.any(sides <= 0):
            return 1e6 - np.sum(sides)
            
        score = -np.sum(sides)
        penalty = 0.0
        
        # Boundary penalties
        x_mins = centers[:, 0] - sides / 2
        x_maxs = centers[:, 0] + sides / 2
        y_mins = centers[:, 1] - sides / 2
        y_maxs = centers[:, 1] + sides / 2
        
        # Push back if out of bounds
        penalty += np.sum(np.maximum(0, -x_mins)**2)
        penalty += np.sum(np.maximum(0, x_maxs - 1)**2)
        penalty += np.sum(np.maximum(0, -y_mins)**2)
        penalty += np.sum(np.maximum(0, y_maxs - 1)**2)
        
        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                
                # Violation if both dx < t and dy < t
                # We want to encourage dx >= t OR dy >= t
                # Penalty term: sum of squared overlaps in both axes
                vx = np.maximum(0, t - dx)
                vy = np.maximum(0, t - dy)
                
                # If they overlap, we add penalty. 
                # Using vx*vy or vx^2+vy^2? 
                # If we use vx^2+vy^2, it pushes them apart in both axes, which is good.
                penalty += vx**2 + vy**2
                
        return score + 100.0 * penalty

    # Multi-start optimization
    num_restarts = 100
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize parameters
        if r == 0:
            # Start with the known good 2x5 grid
            params_init = np.hstack([centers_base, sides_base.reshape(-1, 1)])
        elif r == 1:
            # Try a 1 large + 9 small config
            # Large square side 0.5 at (0.25, 0.25)
            # 9 small squares side 0.25 in the remaining space?
            # This is tricky to construct perfectly, so just random near grid
            params_init = np.hstack([centers_base, sides_base.reshape(-1, 1)])
            params_init += rng.normal(0, 0.1, params_init.shape)
        else:
            # Random initialization
            # Centers in [0.1, 0.9], Sides in [0.1, 0.6]
            centers_init = rng.uniform(0.1, 0.9, (n, 2))
            sides_init = rng.uniform(0.1, 0.6, n)
            params_init = np.hstack([centers_init, sides_init.reshape(-1, 1)])
        
        # Optimize
        try:
            res = minimize(objective, params_init, method='Powell', 
                           options={'maxiter': 500, 'maxfev': 10000, 'disp': False})
            
            if res.success or res.fun < 1e6:
                best_params = res.x
                best_centers = best_params[:, :2]
                best_sides = best_params[:, 2]
                
                # Check validity
                if validate(best_centers, best_sides):
                    current_score = np.sum(best_sides)
                    if current_score > best_score:
                        best_score = current_score
                        best_centers = best_centers.copy()
                        best_sides = best_sides.copy()
                        
        except Exception as e:
            pass

    return best_centers, best_sides

# EVOLVE-BLOCK-END
