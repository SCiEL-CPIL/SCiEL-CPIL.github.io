import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute pairwise distances efficiently using upper triangle
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n_points):
    """Initialize points on a hexagonal lattice clipped to the unit circle."""
    # Estimate spacing for n points in unit circle
    # Area per point ~ pi / n
    # Hex packing: area per point = (sqrt(3)/2) * d^2 where d is distance between centers
    # d ~ sqrt(2 * pi / (n * sqrt(3)))
    d_est = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
    
    # Generate a large hex lattice
    points_list = []
    # Range of indices
    max_idx = int(np.ceil(1.0 / d_est)) + 2
    
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = (i + 0.5 * (j % 2)) * d_est
            y = j * d_est * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    # If we have too many, subsample or adjust. If too few, expand.
    # For now, just take the first N if available, else return what we have (should be enough)
    if len(points) >= n_points:
        # Sort by distance from center to prioritize interior? Or random?
        # Let's just take the first N. The optimization will fix it.
        return points[:n_points]
    else:
        # Fallback: simple grid
        side = int(np.sqrt(n_points))
        x = np.linspace(-0.8, 0.8, side)
        y = np.linspace(-0.8, 0.8, side)
        xx, yy = np.meshgrid(x, y)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        # Filter outside
        norms = np.linalg.norm(pts, axis=1)
        pts = pts[norms <= 0.9]
        if len(pts) < n_points:
             # Add more points randomly inside
             while len(pts) < n_points:
                 angle = np.random.uniform(0, 2*np.pi)
                 r = np.random.uniform(0, 0.9)
                 pts = np.vstack([pts, [r*np.cos(angle), r*np.sin(angle)]])
        return pts[:n_points]

def _soft_min_and_grad(vals, beta):
    """Compute soft-min of vals and its gradient."""
    # soft_min = -1/beta * log(sum(exp(-beta * vals)))
    # grad_i = exp(-beta * vals_i) / sum(exp(-beta * vals_j))
    # To avoid overflow, subtract max
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    soft_min_val = m - np.log(sum_exp) / beta
    grad = exp_vals / sum_exp
    return soft_min_val, grad

def _objective_and_grad(x, beta):
    """
    Objective: -soft_min(constraints)
    We want to maximize the min constraint, so we minimize the negative soft-min.
    Constraints:
    1. Pairwise distances / 2
    2. Boundary distances (1 - norm)
    """
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Compute pairwise distances
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff**2, axis=1)
    pair_dists = np.sqrt(dists_sq)
    
    # Compute boundary distances
    norms = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine constraints
    # We want the min of (pair_dists/2, bound_dists)
    # Let's create a single vector of all constraints
    # c = [pair_dists/2, bound_dists]
    c = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # Soft min
    smin_val, weights = _soft_min_and_grad(c, beta)
    
    # Objective is -smin_val
    obj = -smin_val
    
    # Gradient
    # d(smin)/dc_i = weights_i
    # d(obj)/dx = - sum_i weights_i * dc_i/dx
    
    grad_c = np.zeros_like(c)
    grad_c[:len(iu[0])] = weights[:len(iu[0])]
    grad_c[len(iu[0]):] = weights[len(iu[0]):]
    
    grad_x = np.zeros_like(points)
    
    # Pairwise part
    # c_k = d_k / 2 = 0.5 * ||p_i - p_j||
    # dc_k/dp_i = 0.5 * (p_i - p_j) / d_k
    # dc_k/dp_j = 0.5 * (p_j - p_i) / d_k
    
    # We need to accumulate gradients for each point
    # Using np.add.at for vectorized accumulation
    # Note: diff is p_i - p_j
    # grad_pair_i = 0.5 * diff / d_k * w_k
    # grad_pair_j = -0.5 * diff / d_k * w_k
    
    # d_k is pair_dists
    # Avoid division by zero
    pair_dists_safe = np.where(pair_dists < 1e-10, 1e-10, pair_dists)
    
    w_pairs = grad_c[:len(iu[0])]
    
    # Contribution to p_i (index iu[0])
    # Vector: w_pairs[:, None] * (0.5 * diff / pair_dists_safe[:, None])
    contrib_i = w_pairs[:, np.newaxis] * (0.5 * diff / pair_dists_safe[:, np.newaxis])
    np.add.at(grad_x, iu[0], contrib_i)
    
    # Contribution to p_j (index iu[1])
    contrib_j = -w_pairs[:, np.newaxis] * (0.5 * diff / pair_dists_safe[:, np.newaxis])
    np.add.at(grad_x, iu[1], contrib_j)
    
    # Boundary part
    # c_k = 1 - ||p_k||
    # dc_k/dp_k = -p_k / ||p_k||
    w_bounds = grad_c[len(iu[0]):]
    norms_safe = np.where(norms < 1e-10, 1e-10, norms)
    contrib_bound = -w_bounds[:, np.newaxis] * (points / norms_safe[:, np.newaxis])
    grad_x += contrib_bound
    
    # Total gradient of objective = - grad_smin
    grad_x = -grad_x
    
    return obj, grad_x.ravel()

def _local_optimize(points, time_limit):
    """Run L-BFGS-B optimization on the given points."""
    x0 = points.ravel()
    
    # Multi-stage beta optimization
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [50, 100, 200, 400, 800]
    
    current_x = x0
    best_score = _compute_score(points)
    
    start_time = time.time()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Optimize with current beta
        # Use previous solution as start
        res = minimize(_objective_and_grad, current_x, args=(beta,), 
                       method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        
        current_x = res.x
        new_points = current_x.reshape(-1, 2)
        
        # Check score
        score = _compute_score(new_points)
        if score > best_score:
            best_score = score
            # Keep current_x for next stage
            
    # Final polish with very high beta if time remains
    if time.time() - start_time < time_limit - 2:
        res = minimize(_objective_and_grad, current_x, args=(1000,), 
                       method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        current_x = res.x
        new_points = current_x.reshape(-1, 2)
        score = _compute_score(new_points)
        if score > best_score:
            best_score = score
    
    return current_x.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initialization
    points = _hex_lattice_init(N)
    
    # 2. Optimization
    # We will run a single robust optimization pass.
    # If we have time, we might try a second restart with different init, 
    # but for now, let's focus on making the main optimization strong.
    
    # Set a time limit for the optimizer
    time_limit = deadline - start_time - 5.0
    
    try:
        optimized_points = _local_optimize(points, time_limit)
    except Exception as e:
        # Fallback if optimization fails
        optimized_points = points
    
    # Ensure validity
    # Clip points that might have drifted slightly outside due to numerical errors
    norms = np.linalg.norm(optimized_points, axis=1)
    # The optimizer should keep them inside, but just in case
    # We don't strictly project here because it might reduce the score if done naively,
    # but the score function penalizes it.
    
    return optimized_points

# Main entry point wrapper to handle any unexpected errors
def construct():
    try:
        return _construct_impl()
    except Exception:
        # Absolute fallback
        return _hex_lattice_init(N)

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple initializations to increase chance of finding a good basin
    inits = []
    
    # Init 1: Hex Lattice
    inits.append(_hex_lattice_init(N))
    
    # Init 2: Random with repulsion (simple)
    # Actually, let's stick to one strong init and optimize well.
    # Or maybe 2 inits if time permits.
    
    points = inits[0]
    
    # Optimize
    time_limit = deadline - start_time - 5.0
    optimized_points = _local_optimize(points, time_limit)
    
    return optimized_points

# Final wrapper to match the required interface
def construct():
    try:
        return _construct_impl()
    except Exception:
        return _hex_lattice_init(N)
