# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize, differential_evolution

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
    if d_min == 0:
        return 1e10
    return d_max / d_min

def smooth_objective(flat_points, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for better gradient-based optimization.
    Approximates d_max - d_min + small penalty for very small d_min.
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 1e10
    
    # Numerically stable log-sum-exp for soft max
    beta_d = beta * dists
    max_beta_d = np.max(beta_d)
    soft_max = (max_beta_d + np.log(np.sum(np.exp(beta_d - max_beta_d)))) / beta
    
    # Numerically stable log-sum-exp for soft min (negative)
    neg_beta_d = -beta * dists
    max_neg_beta_d = np.max(neg_beta_d)
    soft_min_neg = (max_neg_beta_d + np.log(np.sum(np.exp(neg_beta_d - max_neg_beta_d)))) / beta
    soft_min = -soft_min_neg
    
    # We want to minimize d_max - d_min, so minimize soft_max - soft_min
    # Add a small penalty if soft_min is too small (points too close)
    penalty = 0.0
    if soft_min < 1e-6:
        penalty = (1e-6 - soft_min) ** 2 * 1000
    
    return soft_max - soft_min + penalty

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

def generate_circular_ring(n_points=16, n_rings=2):
    """
    Generate points on concentric circles with optimal angular spacing.
    """
    points_list = []
    
    if n_rings == 1:
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        for a in angles:
            points_list.append([np.cos(a), np.sin(a)])
    else:
        # Split points between inner and outer rings
        n_inner = n_points // 2
        n_outer = n_points - n_inner
        
        inner_radius = 0.5
        outer_radius = 1.0
        
        # Inner ring
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        for a in angles_inner:
            points_list.append([inner_radius * np.cos(a), inner_radius * np.sin(a)])
        
        # Outer ring - offset angles for better spacing
        angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + np.pi / n_outer
        for a in angles_outer:
            points_list.append([outer_radius * np.cos(a), outer_radius * np.sin(a)])
    
    points = np.array(points_list)
    points -= np.mean(points, axis=0)
    
    return points

def local_search(points, time_limit):
    """
    Perform local search with simulated annealing to maximize the score.
    """
    n = points.shape[0]
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    temp = 0.01
    min_temp = 1e-6
    cooling_rate = 0.999
    step_size = 0.1
    
    while time.time() - start_time < time_limit:
        idx = np.random.randint(n)
        
        spread = np.std(current_points)
        if spread == 0:
            spread = 1.0
            
        delta = np.random.randn(2) * step_size * spread
        
        new_points = current_points.copy()
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
        step_size *= 0.9999
        
    return best_points

def refine_points(points, time_limit):
    """
    Use multiple optimization passes for high-precision refinement.
    """
    n = points.shape[0]
    start_time = time.time()
    
    # Pass 1: Nelder-Mead on the original objective
    x0 = points.flatten()
    
    res = minimize(
        objective_function, 
        x0, 
        method='Nelder-Mead', 
        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
    )
    
    refined_points = res.x.reshape(-1, 2)
    
    # Pass 2: If time allows, use L-BFGS-B with smooth objective
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.5:
        remaining = time_limit - elapsed - 2
        
        # Try L-BFGS-B with the smooth objective for better gradients
        try:
            res2 = minimize(
                smooth_objective,
                refined_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            refined_points = res2.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Pass 3: Final Nelder-Mead polish with very tight tolerance
        elapsed2 = time.time() - start_time
        if time_limit - elapsed2 > 5:
            try:
                res3 = minimize(
                    objective_function,
                    refined_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
                )
                refined_points = res3.x.reshape(-1, 2)
            except Exception:
                pass
        
    return refined_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    
    # Deadline guard: 630s limit, use 600s for safety
    deadline = time.time() + 590
    
    # Generate diverse initializations
    inits = []
    
    # 1. Standard 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid_points)
    
    # 2. Hexagonal lattice subset
    pts_hex = generate_hex_lattice(n)
    inits.append(pts_hex)
    
    # 3. Circular ring configuration
    pts_circ = generate_circular_ring(n, n_rings=2)
    inits.append(pts_circ)
    
    # 4. Random initialization with fixed seed
    np.random.seed(42)
    pts_rand = np.random.randn(n, d) * 2
    pts_rand -= np.mean(pts_rand, axis=0)
    inits.append(pts_rand)
    
    # 5. Another random with different seed
    np.random.seed(123)
    pts_rand2 = np.random.randn(n, d) * 2
    pts_rand2 -= np.mean(pts_rand2, axis=0)
    inits.append(pts_rand2)
    
    best_points = inits[0]
    best_score = calculate_score(best_points)
    
    # Allocate time for local search on each initialization
    num_inits = len(inits)
    # Reserve 30% of time for final refinement
    search_time_total = (deadline - time.time()) * 0.7
    time_per_init = search_time_total / num_inits
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 180:  # Leave time for refinement
            break
            
        end_time_for_this = time.time() + time_per_init
        res_points = local_search(init_pts, end_time_for_this - time.time())
        res_score = calculate_score(res_points)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
    
    # Final high-precision refinement on the best result
    if time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        best_points = refine_points(best_points, remaining_time)
        final_score = calculate_score(best_points)
        
        # If refinement made it worse, keep the pre-refinement points
        if final_score < best_score:
            best_points = best_points  # Keep original
    
    return best_points

# EVOLVE-BLOCK-END
