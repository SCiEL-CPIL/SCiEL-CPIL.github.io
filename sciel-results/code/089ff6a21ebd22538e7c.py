from scipy.spatial.distance import cdist
import numpy as np
from scipy.optimize import minimize
import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 150.0 # Seconds
SAFETY_MARGIN = 5.0 # Seconds

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(centers):
    """Compute the maximum possible radius for the given centers."""
    # Distance to boundary
    # Centers must be in [-0.5, 0.5]
    # dist_to_wall = 0.5 - max(|x|, |y|, |z|) for each point? No.
    # dist_to_wall_i = min(0.5 - x_i, 0.5 - y_i, 0.5 - z_i, 0.5 + x_i, 0.5 + y_i, 0.5 + z_i)
    # Actually, min_k (0.5 - |c_ik|) is correct for distance to the cube boundary [-0.5, 0.5]^3.
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    dist_to_wall = 0.5 - max_abs
    r_wall = np.min(dist_to_wall)
    
    # Pairwise distances
    # Use cdist for efficiency
    from scipy.spatial.distance import cdist
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    
    return min(r_wall, r_pair)

def _simulated_annealing(centers, T_start=0.01, T_end=0.0001, steps=20000):
    """
    Simulated annealing to maximize the minimum radius.
    """
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / steps)
    
    for i in range(steps):
        if _check_time():
            break
            
        # Pick a random sphere
        idx = np.random.randint(N)
        
        # Generate a random move
        # Step size decreases with temperature
        step_size = T * 0.5
        move = np.random.uniform(-step_size, step_size, 3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += move
        
        # Check if move keeps it roughly inside? 
        # We can allow it to go out, but the radius will become negative.
        # Better to clamp to [-0.5, 0.5] to keep it valid?
        # If we clamp, the radius calculation handles it.
        # But if it goes out, r_wall becomes negative.
        
        new_radius = _compute_radius(new_centers)
        
        delta = new_radius - current_radius
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_centers = new_centers
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        
        T *= cooling_rate
        
    return best_centers, best_radius

def construct():
    # 1. Initialize with a good grid
    # A 5x5x4 grid is a good start.
    # Or use BCC?
    # Let's use a 5x5x4 grid, it's simple and valid.
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
    # Wait, 5*5*4 = 100.
    # Ticks for x,y: 5 points. Ticks for z: 4 points.
    # This creates a 5x5x4 block.
    
    # Ensure we have exactly N points
    if len(grid) < N:
        # Fallback to cube root grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
    else:
        grid = grid[:N]
    
    # 2. Run Simulated Annealing
    # Time budget: use most of the time
    # SA is stochastic, so run as long as possible
    best_centers, best_radius = _simulated_annealing(grid, T_start=0.05, T_end=0.0001, steps=50000)
    
    # 3. Optional: Polish with local optimization
    # If time remains, try to polish the best center using a smooth objective
    if not _check_time():
        def smooth_objective(params):
            C = params.reshape(N, 3)
            # Compute min distance to wall
            abs_C = np.abs(C)
            max_abs = np.max(abs_C, axis=1)
            r_wall = 0.5 - np.max(max_abs) # This is not right, min wall dist is min(0.5 - max_abs_i)
            r_wall = np.min(0.5 - max_abs)
            
            # Compute min pair dist using log-sum-exp for smoothness?
            # Or just use hard min for now, L-BFGS can handle it if we are close.
            D = cdist(C, C)
            np.fill_diagonal(D, np.inf)
            min_dist = np.min(D)
            r_pair = min_dist / 2.0
            
            r = min(r_wall, r_pair)
            return -r
        
        try:
            res = minimize(smooth_objective, best_centers.flatten(), method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
            polished_centers = res.x.reshape(N, 3)
            polished_radius = _compute_radius(polished_centers)
            
            if polished_radius > best_radius:
                best_centers = polished_centers
                best_radius = polished_radius
        except Exception:
            pass

    return best_centers
