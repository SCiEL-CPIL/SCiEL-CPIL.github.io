"""
Improved packing of 200 equal circles in a unit square.
Strategy:
1. Generate diverse initial configurations (Hex, Square, Random).
2. Refine using L-BFGS-B on a soft-min surrogate objective.
3. Polish using Simulated Annealing on the exact non-smooth objective.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200

def _compute_radius(points):
    """Compute the max common radius for points in [-0.5, 0.5]^2."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary < 0:
        return 0.0
    
    # Min pairwise distance
    if N < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
        
    return min(min_pair / 2.0, min_boundary)

def _soft_min_radius(points, beta=50.0):
    """
    Smooth surrogate for the min-radius objective using log-sum-exp.
    We want to maximize r = min(min_pair/2, min_boundary).
    Equivalently, minimize -r.
    
    Let Y be the vector of all constraint distances (wall and pair/2).
    r = min(Y).
    softmin(Y) = -1/beta * log(sum(exp(-beta * Y)))
    
    We minimize -softmin(Y).
    """
    if points.shape != (N, 2):
        return 1000.0
    if not np.all(np.isfinite(points)):
        return 1000.0
    
    # Wall distances: 4 per point
    # Left: 0.5 - x, Right: 0.5 + x, Bottom: 0.5 - y, Top: 0.5 + y
    x, y = points[:, 0], points[:, 1]
    walls = np.concatenate([
        0.5 - x,
        0.5 + x,
        0.5 - y,
        0.5 + y
    ])
    
    # Pair distances: N(N-1)/2
    # Using pdist for efficiency, then dividing by 2
    if N > 1:
        pairs = pdist(points) / 2.0
        # Combine
        Y = np.concatenate([walls, pairs])
    else:
        Y = walls
        
    # Softmin calculation
    # Subtract min to avoid underflow
    Y_min = np.min(Y)
    # If Y_min is very negative, exp(-beta*(Y-Y_min)) might be large, but Y-Y_min >= 0
    exp_terms = np.exp(-beta * (Y - Y_min))
    sum_exp = np.sum(exp_terms)
    
    softmin_val = Y_min - np.log(sum_exp) / beta
    return -softmin_val

def _gen_hex_init(n_rows):
    """Generate a hexagonal grid initialization."""
    n_cols = int(np.ceil(N / n_rows))
    # Estimate spacing
    # Fit n_cols in width 1, n_rows in height 1
    # s_x ~ 1/n_cols, s_y ~ s_x * sqrt(3)/2
    # But we need to fit in square.
    # Let's just place them and let optimizer fix it.
    # Standard hex offset
    s_x = 1.0 / n_cols
    s_y = s_x * np.sqrt(3) / 2.0
    
    # Check if it fits vertically
    total_h = (n_rows - 1) * s_y
    if total_h > 1.0:
        scale = 1.0 / total_h
        s_x *= scale
        s_y *= scale
        
    points = []
    for r in range(n_rows):
        for c in range(n_cols):
            if len(points) >= N:
                break
            offset = 0.0 if r % 2 == 0 else s_x / 2.0
            x = -0.5 + s_x/2 + c * s_x + offset
            y = -0.5 + s_y/2 + r * s_y
            # Clamp to ensure inside
            x = np.clip(x, -0.49, 0.49)
            y = np.clip(y, -0.49, 0.49)
            points.append([x, y])
            
    if len(points) < N:
        # Fill remaining randomly
        while len(points) < N:
            points.append([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)])
            
    return np.array(points[:N])

def _gen_square_init():
    """Generate a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _gen_random_init():
    """Generate a random initialization."""
    return np.random.uniform(-0.45, 0.45, size=(N, 2))

def _refine_lbfgs(points, time_limit=5.0):
    """Refine points using L-BFGS-B on soft-min objective."""
    start = time.time()
    
    # Flatten
    x0 = points.flatten()
    
    # Objective function
    def objective(x):
        p = x.reshape(N, 2)
        return _soft_min_radius(p, beta=50.0)
        
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if res.success or res.fun < objective(x0):
            return res.x.reshape(N, 2)
    except Exception:
        pass
        
    return points

def _simulated_annealing(points, time_limit=20.0):
    """
    Simulated Annealing on the exact objective.
    Moves are random perturbations.
    Accepts worse solutions with probability exp(-delta_E / T).
    """
    current_points = points.copy()
    current_r = _compute_radius(current_points)
    best_points = current_points.copy()
    best_r = current_r
    
    T = 0.01 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate a move
        # Perturb a random subset of points
        n_perturb = np.random.randint(1, 5)
        idxs = np.random.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        # Step size scales with T
        step = 0.05 * T + 0.001
        new_points[idxs] += np.random.normal(0, step, size=(n_perturb, 2))
        
        # Clamp to bounds
        new_points = np.clip(new_points, -0.5, 0.5)
        
        new_r = _compute_radius(new_points)
        
        delta_r = new_r - current_r
        
        if delta_r > 0:
            # Accept
            current_points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = current_points.copy()
        else:
            # Metropolis criterion
            if np.random.rand() < np.exp(delta_r / T):
                current_points = new_points
                current_r = new_r
                
        T *= cooling_rate
        
    return best_points

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    best_points = None
    best_r = 0.0
    
    # 1. Generate initial candidates
    inits = []
    
    # Square grid
    inits.append(_gen_square_init())
    
    # Hex grids with varying rows
    for nr in [12, 13, 14, 15, 16]:
        inits.append(_gen_hex_init(nr))
        
    # Random
    inits.append(_gen_random_init())
    
    # 2. Refine each with L-BFGS
    refined_candidates = []
    for init in inits:
        if time.time() > deadline - 30: # Reserve 30s for SA
            break
            
        # Allocate time for L-BFGS
        time_left_for_lbfgs = min(5.0, deadline - time.time() - 30)
        if time_left_for_lbfgs < 1.0:
            break
            
        refined = _refine_lbfgs(init, time_limit=time_left_for_lbfgs)
        r = _compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_points = refined.copy()
            
        refined_candidates.append(refined)
        
    # 3. Polish the best candidate (and maybe others) with SA
    # We have ~30s left. Let's use it for SA on the best one.
    # Or split time among top 2.
    
    if time.time() < deadline - 5:
        # Use the best found so far
        sa_time = min(25.0, deadline - time.time() - 5)
        if sa_time > 1.0:
            polished = _simulated_annealing(best_points, time_limit=sa_time)
            r_polished = _compute_radius(polished)
            
            if r_polished > best_r:
                best_r = r_polished
                best_points = polished.copy()
                
    # Fallback if nothing worked
    if best_points is None:
        best_points = _gen_square_init()
        
    # Final safety check
    if not np.all(np.isfinite(best_points)):
        best_points = _gen_square_init()
        
    return best_points
