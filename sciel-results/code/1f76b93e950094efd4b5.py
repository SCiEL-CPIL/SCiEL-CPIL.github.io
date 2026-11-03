"""Improved packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200


def _compute_radius(points):
    """Compute the max common radius for points in [-0.5, 0.5]^2."""
    if len(points) != N:
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
    Returns a smooth approximation of min(min_pair/2, min_boundary).
    Lower beta = smoother (more averaged), higher beta = sharper (closer to true min).
    """
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    # Soft min of boundary distances
    # log-sum-exp trick for stability
    max_b = np.max(dist_boundary)
    soft_min_boundary = max_b - np.log(np.sum(np.exp(dist_boundary - max_b))) / len(dist_boundary)
    
    if N < 2:
        return soft_min_boundary
    
    d = pdist(points)
    max_d = np.max(d)
    soft_min_pair = max_d - np.log(np.sum(np.exp(d - max_d))) / len(d)
    
    # Combine: soft min of (soft_min_pair/2, soft_min_boundary)
    vals = np.array([soft_min_pair / 2.0, soft_min_boundary])
    max_v = np.max(vals)
    soft_min_combined = max_v - np.log(np.sum(np.exp(vals - max_v))) / len(vals)
    
    return soft_min_combined

def _neg_soft_min_radius(points, beta=50.0):
    """Negative soft min radius for minimization."""
    return -_soft_min_radius(points.reshape(-1, 2), beta)


def _local_refine(points, time_budget=5.0):
    """
    Local refinement using coordinate ascent with adaptive step size.
    For each point, try to move it in a direction that increases the min distance.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    step = 0.002
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        # Randomize the order of points to avoid bias
        order = np.random.permutation(N)
        for i in order:
            if time.time() - start_time > time_budget:
                break
            
            # Try moving point i in 8 directions
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[i] += np.array([dx, dy])
                    if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                        new_r = _compute_radius(new_points)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            points = new_points
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 2:
                # Reduce step size
                step *= 0.5
                if step < 1e-6:
                    break
                no_improve_count = 0
    
    return points, current_r


def _optimize_from_start(points, deadline):
    """
    Run L-BFGS-B optimization on the soft-min objective starting from 'points'.
    """
    start_time = time.time()
    
    # Flatten points
    x0 = points.flatten()
    
    # Define objective and constraints
    # We want to maximize soft_min_radius, so minimize negative
    # Constraints: points must be within [-0.5, 0.5]
    # L-BFGS-B handles bounds natively
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        pts = x.reshape(-1, 2)
        return _neg_soft_min_radius(pts, beta=50.0)
        
    # Run optimization
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20, 'maxfun': 500}
        )
    except Exception:
        return points, _compute_radius(points)
    
    opt_points = result.x.reshape(-1, 2)
    
    # Polish with local refinement if time permits
    if time.time() < deadline - 2:
        opt_points, r = _local_refine(opt_points, time_budget=min(2.0, deadline - time.time()))
    else:
        r = _compute_radius(opt_points)
        
    return opt_points, r


def construct():
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_points = None
    best_r = 0.0
    
    # Strategy 1: Square Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    sq_points = grid[:N]
    
    # Strategy 2: Hexagonal Grid
    # Approximate 200 points in hex grid
    # Rows ~ 14, Cols ~ 14
    hex_points = None
    for rows in range(13, 16):
        cols = int(np.ceil(N / rows))
        # Generate hex grid
        pts = []
        s_x = 1.0 / (cols + 0.5) # Add margin
        s_y = 1.0 / (rows * np.sqrt(3)/2 + 0.5)
        s = min(s_x, s_y)
        
        # Center the grid
        width = (cols - 1) * s
        height = (rows - 1) * s * np.sqrt(3) / 2
        
        for j in range(rows):
            offset = (j % 2) * s / 2
            n_in_row = cols
            if j == rows - 1 and N % rows != 0:
                 n_in_row = N % rows
            for i in range(n_in_row):
                x = -width/2 + i * s + offset
                y = -height/2 + j * s * np.sqrt(3) / 2
                pts.append([x, y])
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        
        if len(pts) == N:
            hex_points = np.array(pts)
            break
            
    if hex_points is None:
        # Fallback to square if hex generation failed
        hex_points = sq_points.copy()

    # Optimization Loop with Restarts
    # We have ~145s. Each L-BFGS run takes ~5-10s. We can do ~10-20 restarts.
    # Let's try a few different initializations.
    
    initializations = [sq_points, hex_points]
    
    # Add some random perturbations to break symmetry
    np.random.seed(42)
    for _ in range(3):
        perturbed = sq_points + np.random.normal(0, 0.01, size=(N, 2))
        # Clip to bounds
        perturbed = np.clip(perturbed, -0.49, 0.49)
        initializations.append(perturbed)
        
    for init in initializations:
        if time.time() > deadline - 15:
            break
            
        # Check if we already have a good result
        if best_r > 0.036:
             # If we are close to optimal, maybe just do a quick polish
             pass

        # Run optimization
        # Allocate time based on remaining budget
        time_per_run = min(10.0, (deadline - time.time()) / 2.0)
        
        # We need to wrap the optimizer to respect the time limit strictly
        # L-BFGS-B doesn't have a strict wall-clock limit, so we rely on maxiter/maxfun
        # and the outer loop check.
        
        # To be safe, we run the optimization in a way that can be interrupted or just trusts maxiter.
        # Given 100 iterations is fast, we can just run it.
        
        opt_points, r = _optimize_from_start(init, deadline)
        
        if r > best_r:
            best_r = r
            best_points = opt_points.copy()
            
    if best_points is None:
        best_points = sq_points
        best_r = _compute_radius(best_points)
        
    # Final Polish
    if time.time() < deadline - 2:
        best_points, best_r = _local_refine(best_points, time_budget=min(5.0, deadline - time.time()))
        
    return best_points
