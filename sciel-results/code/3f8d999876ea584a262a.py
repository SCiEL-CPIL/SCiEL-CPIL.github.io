"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _penalized(params):
    """Penalized objective: minimize -sum(radii) + penalties for violations."""
    n = len(params) // 3
    xy = params[:2*n].reshape(n, 2)
    r = params[2*n:]
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for circles going outside [0,1]^2
    for i in range(n):
        x, y = xy[i]
        ri = r[i]
        # Distance to boundaries
        d_left = x - ri
        d_right = 1.0 - x - ri
        d_bottom = y - ri
        d_top = 1.0 - y - ri
        # Penalize if any distance is negative
        for d in [d_left, d_right, d_bottom, d_top]:
            if d < 0:
                obj += 100.0 * d**2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                obj += 100.0 * (min_dist - dist)**2
    
    return obj


def _grid_init():
    """Create initial grid placement."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    return np.array(positions)


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple random initializations
    np.random.seed(42)
    n_trials = 5
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        if trial == 0:
            # Use grid initialization
            init = _grid_init()
        else:
            # Random initialization with some structure
            # Place circles in a rough grid with random perturbations
            side = int(np.ceil(np.sqrt(N)))
            init = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                init[i, 0] = (gx + 0.5) / side + np.random.uniform(-0.1, 0.1)
                init[i, 1] = (gy + 0.5) / side + np.random.uniform(-0.1, 0.1)
                init[i, 2] = 0.5 / side - 0.001
            # Clip to valid range
            init[:, 0] = np.clip(init[:, 0], 0.05, 0.95)
            init[:, 1] = np.clip(init[:, 1], 0.05, 0.95)
            init[:, 2] = np.clip(init[:, 2], 0.01, 0.5)
        
        # Flatten parameters
        params = init.flatten()
        
        # Bounds: x, y in [0.01, 0.99], r in [0.001, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))  # x
            bounds.append((0.01, 0.99))  # y
        for i in range(N):
            bounds.append((0.001, 0.5))  # r
        
        # Optimize
        try:
            result = minimize(
                _penalized,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 1000}
            )
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline:
            break
    
    # If no valid result, fall back to grid
    if best_params is None:
        init = _grid_init()
        return init
    
    # Extract and validate
    n = len(best_params) // 3
    xy = best_params[:2*n].reshape(n, 2)
    r = best_params[2*n:]
    
    result = np.column_stack([xy, r])
    
    # Final validation and clipping
    for i in range(n):
        x, y, ri = result[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    return result
# EVOLVE-BLOCK-END
