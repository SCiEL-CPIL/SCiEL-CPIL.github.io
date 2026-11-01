import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5

def _radius_score(x):
    """Calculate the actual radius (min distance/2 and wall dist)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (N, 2) or not np.all(np.isfinite(x)):
        return 0.0
    # Pairwise distances
    diff = x[:, None, :] - x[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq)) / 2.0
    
    # Wall distances
    min_wall_dist = np.min(HALF - np.abs(x))
    
    return min(min_pair_dist, min_wall_dist)

def _soft_min_radius(x, beta=100.0):
    """
    Soft-min approximation of the radius.
    Higher beta -> sharper minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    # Pairwise
    diff = x[:, None, :] - x[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    pair_dists = 0.5 * np.sqrt(dist_sq)
    
    # Walls
    wall_dists = HALF - np.abs(x)
    
    # Soft min of all these values
    # Combine pair_dists (N*N) and wall_dists (N*2)
    # To avoid huge arrays, we can compute soft min separately and then combine?
    # Or just concatenate. N=50, N*N=2500. 2500+100=2600 floats. This is fine.
    
    all_dists = np.concatenate([pair_dists.flatten(), wall_dists.flatten()])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To prevent underflow, shift by min
    d_min = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - d_min))
    soft_min = d_min - np.log(np.sum(exp_terms)) / beta
    
    return soft_min

def _neg_soft_min(x, beta=100.0):
    """Objective function for minimization."""
    return -_soft_min_radius(x, beta)

def _project(x):
    """Project points into the square."""
    return np.clip(x, -HALF, HALF)

def _nelder_mead_optimize(x0, beta=100.0, max_iter=2000, deadline=None):
    """
    Run Nelder-Mead optimization.
    """
    x0 = np.array(x0, dtype=np.float64)
    
    # Initial simplex size
    # Default is 0.05 * x0 + 0.00025 for zeros
    # We want to explore a bit
    xatol = 1e-8
    fatol = 1e-10
    
    def objective(x):
        if deadline and time.time() > deadline:
            return 1e10 # Large penalty to stop
        return _neg_soft_min(x, beta)
        
    def callback(xk):
        if deadline and time.time() > deadline:
            raise TimeoutError()
            
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True})
        return res.x, -res.fun
    except TimeoutError:
        # If timeout, return current best? We don't have easy access to current best in callback without extra state.
        # Better to check time inside objective and return a large value, but Nelder-Mead might not stop immediately.
        # Let's just catch the exception and return the initial point if it fails.
        return x0, _radius_score(x0)
    except Exception:
        return x0, _radius_score(x0)

def construct():
    deadline = time.time() + 140.0 # Safety margin
    
    best_x = None
    best_r = 0.0
    
    # Generate seeds
    seeds = []
    
    # 1. Grid seed
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        seeds.append(grid[idx])
    else:
        seeds.append(grid)
        
    # 2. Hex seed
    dx = 2 * 0.07 # Approx radius
    dy = np.sqrt(3) * 0.07
    hex_pts = []
    row = 0
    y = -HALF + 0.07
    while y <= HALF - 0.07 + 1e-9:
        x = -HALF + 0.07 + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - 0.07 + 1e-9:
            hex_pts.append([x, y])
            x += dx
        y += dy
        row += 1
    if hex_pts:
        hex_pts = np.array(hex_pts)
        if len(hex_pts) >= N:
            d2 = np.sum(hex_pts * hex_pts, axis=1)
            idx = np.argsort(d2)[:N]
            seeds.append(hex_pts[idx])
        else:
            seeds.append(hex_pts)
            
    # 3. Random seeds
    np.random.seed(42)
    for _ in range(5):
        r_x = np.random.uniform(-HALF, HALF, N)
        r_y = np.random.uniform(-HALF, HALF, N)
        seeds.append(np.column_stack((r_x, r_y)))
        
    # Run Nelder-Mead on each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5:
            break
            
        # Ensure seed is valid shape
        if len(seed) < N:
            # Pad with random
            pad = np.random.uniform(-HALF, HALF, (N - len(seed), 2))
            seed = np.vstack([seed, pad])
        else:
            seed = seed[:N]
            
        seed = _project(seed)
        
        # Use a slightly lower beta for initial search to avoid getting stuck in sharp minima too early?
        # Or use adaptive beta? Nelder-Mead doesn't easily support changing objective params.
        # Let's use a moderate beta.
        beta = 50.0
        
        x_opt, r_opt = _nelder_mead_optimize(seed, beta=beta, max_iter=1000, deadline=deadline - 2)
        
        # Polish with higher beta
        if time.time() < deadline - 2:
            x_opt2, r_opt2 = _nelder_mead_optimize(x_opt, beta=200.0, max_iter=500, deadline=deadline - 1)
            if r_opt2 > r_opt:
                x_opt = x_opt2
                r_opt = r_opt2
                
        # Final check with exact score
        r_exact = _radius_score(x_opt)
        if r_exact > best_r:
            best_r = r_exact
            best_x = x_opt
            
    if best_x is None:
        # Fallback
        best_x = np.random.uniform(-HALF, HALF, (N, 2))
        
    return best_x
