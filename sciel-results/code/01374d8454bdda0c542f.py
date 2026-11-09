# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """
    Calculate the metric R = (d_min / d_max)^2
    """
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def soft_objective(flat_points, beta=100.0):
    """
    Smooth surrogate for min/max using log-sum-exp.
    Minimizes: log(sum(exp(-beta*d_i))) - log(sum(exp(beta*d_i)))
    which approximates -d_min + d_max for large beta.
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e10
    
    # Soft min: -log(sum(exp(-beta*d_i)))/beta ≈ -d_min
    # Soft max: log(sum(exp(beta*d_i)))/beta ≈ d_max
    # We want to minimize d_max - d_min, so minimize soft_max - soft_min
    
    # Numerically stable log-sum-exp
    beta_d = beta * dists
    max_beta_d = np.max(beta_d)
    soft_max = (max_beta_d + np.log(np.sum(np.exp(beta_d - max_beta_d)))) / beta
    
    neg_beta_d = -beta * dists
    max_neg_beta_d = np.max(neg_beta_d)
    soft_min = (max_neg_beta_d + np.log(np.sum(np.exp(neg_beta_d - max_neg_beta_d)))) / beta
    
    # Objective: soft_max - soft_min (we want to minimize this)
    return soft_max - soft_min

def objective_function(flat_points):
    """
    Objective function for minimization: minimize d_max / d_min
    which is equivalent to maximizing d_min / d_max
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e10
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min < 1e-10:
        return 1e10
    return d_max / d_min

def generate_hex_lattice(n_points=16):
    """
    Generate a hexagonal lattice subset of n_points, centered.
    """
    a = 1.0
    b = a * np.sqrt(3) / 2
    
    points_list = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * a + (j % 2) * (a / 2)
            y = j * b
            points_list.append([x, y])
    
    points = np.array(points_list)
    points -= np.mean(points, axis=0)
    
    dists_from_center = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists_from_center)
    
    hex_points = points[sorted_indices[:n_points]]
    
    return hex_points

def local_search(points, time_limit):
    """
    Perform local search with simulated annealing to maximize the score.
    Moves multiple points at once for faster exploration.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    temp = 0.005
    min_temp = 1e-7
    cooling_rate = 0.9995
    
    step_size = 0.05
    
    # Number of points to move per iteration (start with 1, increase)
    n_move = 1
    
    while time.time() - start_time < time_limit:
        # Occasionally move multiple points
        if np.random.rand() < 0.3:
            n_move = np.random.randint(1, min(4, n + 1))
        else:
            n_move = 1
        
        # Pick random points to move
        indices = np.random.choice(n, size=n_move, replace=False)
        
        spread = np.std(current_points)
        if spread == 0:
            spread = 1.0
        
        new_points = current_points.copy()
        for idx in indices:
            delta = np.random.randn(2) * step_size * spread
            new_points[idx] += delta
        
        new_score = calculate_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            if temp > min_temp:
                delta_score = new_score - current_score
                prob = np.exp(delta_score / temp)
                if np.random.rand() < prob:
                    current_points = new_points
                    current_score = new_score
        
        temp *= cooling_rate
        step_size *= 0.99995
        
    return best_points

def refine_points(points, time_limit):
    """
    Use L-BFGS-B with soft objective for fast refinement, then Nelder-Mead for final polish.
    """
    n = points.shape[0]
    start_time = time.time()
    
    x0 = points.flatten()
    
    # Phase 1: L-BFGS-B with soft objective (fast, smooth)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        res1 = minimize(
            soft_objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        x0 = res1.x
        refined_points = res1.x.reshape(-1, 2)
    
    # Phase 2: Nelder-Mead with exact objective for final polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        res2 = minimize(
            objective_function, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        refined_points = res2.x.reshape(-1, 2)
    
    return refined_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    deadline = time.time() + 590
    
    # Initial guess: 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    best_points = points
    best_score = calculate_score(points)
    
    inits = []
    inits.append(points)
    
    pts_hex = generate_hex_lattice(n)
    inits.append(pts_hex)
    
    np.random.seed(42)
    pts_rand = np.random.randn(n, d)
    pts_rand -= np.mean(pts_rand, axis=0)
    inits.append(pts_rand)
    
    # Also try a ring-based initialization
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    r_inner = 0.5
    r_outer = 1.5
    pts_ring = np.column_stack([
        (r_inner + r_outer/2 * np.cos(angles + np.pi/16)),
        (r_inner + r_outer/2 * np.sin(angles + np.pi/16))
    ])
    inits.append(pts_ring)
    
    num_inits = len(inits)
    time_per_init = (deadline - time.time()) / (num_inits + 2)  # +2 for polish phases
    
    for init_pts in inits:
        if time.time() > deadline - 60:
            break
            
        end_time_for_this = time.time() + time_per_init
        res_points = local_search(init_pts, end_time_for_this - time.time())
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        best_points = refine_points(best_points, remaining_time)
    
    return best_points

# EVOLVE-BLOCK-END
