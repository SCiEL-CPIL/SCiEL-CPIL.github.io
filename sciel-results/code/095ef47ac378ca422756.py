import numpy as np
from scipy.optimize import minimize
import time

N = 26
TIME_LIMIT = 110  # seconds, leaving margin for 120s limit

def compute_penalty_and_value(params, lam=100.0):
    """
    params: array of shape (N, 3) with [x, y, r]
    returns: penalty + value (to be minimized)
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Sum of radii (we want to maximize, so we minimize -sum)
    sum_r = np.sum(r)
    
    # Boundary penalties
    # x - r < 0, y - r < 0, 1-x-r < 0, 1-y-r < 0
    viol_x_min = np.maximum(0, r - x)
    viol_y_min = np.maximum(0, r - y)
    viol_x_max = np.maximum(0, r - (1 - x))
    viol_y_max = np.maximum(0, r - (1 - y))
    
    boundary_pen = np.sum(viol_x_min**2) + np.sum(viol_y_min**2) + np.sum(viol_x_max**2) + np.sum(viol_y_max**2)
    
    # Overlap penalties
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            viol = np.maximum(0, min_dist - dist)
            overlap_pen += viol**2
            
    total_pen = boundary_pen + overlap_pen
    return -sum_r + lam * total_pen

def compute_value(params):
    """
    Returns the sum of radii if valid, else -inf or a large negative number.
    We check validity by ensuring all constraints are satisfied within a small tolerance.
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Check boundaries
    if np.any(x < r - 1e-6) or np.any(y < r - 1e-6) or np.any(x > 1 - r + 1e-6) or np.any(y > 1 - r + 1e-6):
        return -np.inf
    if np.any(r < 1e-6):
        return -np.inf
        
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-6:
                return -np.inf
                
    return np.sum(r)

def generate_hexagonal(n):
    """Generate a hexagonal packing initial guess."""
    # Hexagonal packing density is ~0.9069
    # For 26 circles, let's estimate the radius.
    # Area of square = 1. Area of circles = n * pi * r^2.
    # r ~ sqrt(1 / (n * pi * density))
    r_est = np.sqrt(1.0 / (n * np.pi * 0.9069))
    
    # Create a grid of hex points
    # We need to fit n points.
    # Let's create a larger grid and pick the best n or adjust.
    # A simpler approach: place circles in a hex pattern with estimated r, 
    # then let the optimizer adjust positions and radii.
    
    # Hex spacing: dx = 2*r, dy = sqrt(3)*r
    # Let's create points in a box slightly larger than 1x1
    points = []
    r_init = r_est * 0.9 # Start a bit smaller to avoid initial overlaps
    
    # Generate hex grid
    for row in range(-1, 5):
        for col in range(-1, 5):
            x = col * 2 * r_init + (row % 2) * r_init
            y = row * np.sqrt(3) * r_init
            # Center the grid roughly in [0,1]
            x += 0.5 - 2 * r_init
            y += 0.5 - 2 * np.sqrt(3) * r_init
            points.append([x, y, r_init])
            
    # Sort by distance to center? Or just take first n?
    # Taking first n might be biased. Let's take the ones closest to the center of the square.
    points = np.array(points)
    cx, cy = points[:, 0], points[:, 1]
    dist_to_center = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    indices = np.argsort(dist_to_center)[:n]
    return points[indices]

def generate_grid(n):
    """Generate a grid initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / side
    points = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            points.append([x, y, r_est * 0.9])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    return np.array(points[:n])

def generate_random(n):
    """Generate a random initial guess."""
    r_est = np.sqrt(1.0 / (n * np.pi))
    points = []
    for _ in range(n):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = r_est * 0.9
        points.append([x, y, r])
    return np.array(points)

def optimize_single(initial_params, time_budget):
    """Run optimization on a single initial configuration."""
    start_time = time.time()
    
    # Define the objective function for scipy
    def objective(params):
        return compute_penalty_and_value(params, lam=50.0)
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0.01, 0.5)] * N
    
    # Use L-BFGS-B
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    except Exception:
        return None, -np.inf

    if time.time() - start_time > time_budget:
        return None, -np.inf

    best_params = result.x
    best_val = compute_value(best_params)
    
    # If the result is invalid, try to repair it or just return invalid
    # We can also try to increase lambda and re-optimize if invalid
    if best_val == -np.inf:
        # Try with higher penalty
        def objective_high_pen(params):
            return compute_penalty_and_value(params, lam=500.0)
        try:
            result2 = minimize(objective_high_pen, best_params, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if time.time() - start_time > time_budget:
                return None, -np.inf
            best_params = result2.x
            best_val = compute_value(best_params)
        except Exception:
            pass

    return best_params, best_val

def construct():
    start_time = time.time()
    
    # Initial configurations
    inits = []
    inits.append(generate_hexagonal(N))
    inits.append(generate_grid(N))
    inits.append(generate_random(N))
    inits.append(generate_random(N))
    
    best_val = -np.inf
    best_params = None
    
    # We have ~110 seconds. Let's allocate time to each initial.
    # 4 inits, so ~25s each.
    time_per_init = 20  # seconds
    
    for i, init in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        params, val = optimize_single(init, time_per_init)
        
        if val > best_val:
            best_val = val
            best_params = params.copy()
            
        # If we found a very good solution, we can stop early?
        # The theoretical max for 26 circles is not known exactly, but 
        # sum of radii for hexagonal packing of 26 circles:
        # r ~ 0.11, sum ~ 2.8.
        # If we get > 2.8, it's likely optimal or close.
        if best_val > 2.8:
            break

    if best_params is None:
        # Fallback to grid
        return generate_grid(N)
        
    return best_params
