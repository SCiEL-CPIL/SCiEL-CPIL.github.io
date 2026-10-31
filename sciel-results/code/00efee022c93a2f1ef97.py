import numpy as np
from scipy.optimize import minimize
import time

N = 26

def _check_validity(circles, tol=1e-9):
    """Check if circles are within [0,1]^2 and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary check
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
        
    # Non-overlap check
    # Use vectorized distance calculation for speed
    # dist[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
    # Need dist[i,j] >= ri + rj for i != j
    
    # To save time, we can do a quick check. 
    # For N=26, N^2 is small (676).
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dist = np.sqrt(x_diff**2 + y_diff**2)
    
    r_sum = r[:, None] + r[None, :]
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Check if any dist < r_sum - tol
    if np.any(dist[mask] < r_sum[mask] - tol):
        return False
    return True

def _penalty_and_objective(params):
    """
    Objective function for optimization.
    params: (N, 3) array of [x, y, r]
    Returns: -sum(r) + penalty
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Soften boundaries: push back if out of bounds
    # Actually, for SA, we just want a high penalty.
    # For minimize, we want to minimize -sum(r) + penalty.
    
    obj = -np.sum(r)
    
    # Boundary penalties
    # If x < 0, penalty (x)^2? No, if x < r, it's out of bounds? 
    # The constraint is x in [0,1] and the circle must be inside.
    # So x >= r and x <= 1-r.
    # If not, penalty.
    bound_pen = 0
    for i in range(N):
        if x[i] < r[i]:
            bound_pen += (r[i] - x[i])**2
        if x[i] > 1 - r[i]:
            bound_pen += (x[i] - (1 - r[i]))**2
        if y[i] < r[i]:
            bound_pen += (r[i] - y[i])**2
        if y[i] > 1 - r[i]:
            bound_pen += (y[i] - (1 - r[i]))**2
            
    # Overlap penalties
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    dist = np.sqrt(x_diff**2 + y_diff**2)
    r_sum = r[:, None] + r[None, :]
    mask = ~np.eye(N, dtype=bool)
    
    overlaps = np.maximum(0, r_sum - dist)
    overlap_pen = np.sum(overlaps[mask]**2)
    
    total_penalty = 100.0 * (bound_pen + overlap_pen)
    
    return obj + total_penalty

def _sa_optimize(init_params, time_limit=10.0):
    """
    Simulated Annealing to find a good configuration.
    """
    current_params = init_params.copy()
    current_score = _penalty_and_objective(current_params)
    
    best_params = current_params.copy()
    best_score = current_score
    
    T = 1.0
    T_min = 0.001
    alpha = 0.9995 # Cooling rate
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Perturb
        new_params = current_params.copy()
        
        # Pick random circle
        i = np.random.randint(N)
        
        # Perturb position
        new_params[i, 0] += np.random.normal(0, 0.05)
        new_params[i, 1] += np.random.normal(0, 0.05)
        
        # Perturb radius
        new_params[i, 2] += np.random.normal(0, 0.05)
        
        # Keep radii positive
        new_params[i, 2] = max(new_params[i, 2], 0.01)
        
        # Keep positions roughly in bounds to avoid huge penalties
        new_params[i, 0] = np.clip(new_params[i, 0], -0.5, 1.5)
        new_params[i, 1] = np.clip(new_params[i, 1], -0.5, 1.5)
        
        new_score = _penalty_and_objective(new_params)
        
        # Acceptance probability
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_params = current_params.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            if np.random.rand() < np.exp(-(new_score - current_score) / T):
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_params = current_params.copy()
                    best_score = current_score
                    
        T *= alpha
        
    return best_params

def _polish(params):
    """
    Use scipy minimize to polish the solution.
    """
    def objective(p):
        return _penalty_and_objective(p.reshape(N, 3))
    
    try:
        res = minimize(objective, params.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success:
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return params

def _make_valid(circles, shrink_factor=1e-6):
    """
    Ensure the circles are strictly valid by shrinking radii if necessary.
    """
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2].copy()
    
    # Adjust for boundaries
    for i in range(N):
        min_r = min(x[i], y[i], 1-x[i], 1-y[i])
        if r[i] > min_r:
            r[i] = min_r * (1 - shrink_factor)
            
    # Adjust for overlaps iteratively
    # This is a simple heuristic. If overlaps remain, shrink all radii slightly.
    for _ in range(10):
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist = np.sqrt(x_diff**2 + y_diff**2)
        r_sum = r[:, None] + r[None, :]
        mask = ~np.eye(N, dtype=bool)
        
        # Check for overlaps
        overlaps = r_sum - dist
        max_overlap = np.max(overlaps[mask])
        
        if max_overlap < 1e-8:
            break
            
        # Shrink radii proportional to overlap? Or just global shrink?
        # Global shrink is safer but might reduce score significantly.
        # Better: identify overlapping pairs and shrink one of them.
        # For simplicity and robustness, let's just shrink all radii by a small factor if invalid.
        r *= 0.999
        
    return np.column_stack((x, y, r))

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_solution = None
    best_score = -np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    grid_init = np.array(cells[:N])
    inits.append(grid_init)
    
    # 2. Random
    np.random.seed(42)
    rand_init = np.zeros((N, 3))
    rand_init[:, 0] = np.random.uniform(0.1, 0.9, N)
    rand_init[:, 1] = np.random.uniform(0.1, 0.9, N)
    rand_init[:, 2] = np.random.uniform(0.05, 0.15, N)
    inits.append(rand_init)
    
    # 3. Hexagonal-ish (Perturbed grid)
    hex_init = grid_init.copy()
    hex_init += np.random.normal(0, 0.05, hex_init.shape)
    hex_init[:, 0] = np.clip(hex_init[:, 0], 0.1, 0.9)
    hex_init[:, 1] = np.clip(hex_init[:, 1], 0.1, 0.9)
    inits.append(hex_init)
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Allocate remaining time to this restart
        remaining = deadline - time.time()
        sa_time = min(15.0, remaining / 2.0) # Spend up to 15s on SA, leave time for polish
        
        if sa_time < 1.0:
             break
             
        sa_result = _sa_optimize(init, time_limit=sa_time)
        
        # Polish
        if time.time() < deadline - 5.0:
             polished = _polish(sa_result)
        else:
             polished = sa_result
             
        # Validate and fix
        valid_result = _make_valid(polished)
        
        # Calculate score (sum of radii)
        score = np.sum(valid_result[:, 2])
        
        if _check_validity(valid_result):
            if score > best_score:
                best_score = score
                best_solution = valid_result
        else:
            # If still invalid, try to force validity by shrinking more
            forced = _make_valid(polished, shrink_factor=0.01)
            if _check_validity(forced):
                 score = np.sum(forced[:, 2])
                 if score > best_score:
                     best_score = score
                     best_solution = forced

    if best_solution is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        
    return best_solution
