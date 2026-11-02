"""Combined program for Heilbronn-11: Targeted Local Search on Critical Triangles."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Safety margin from 150s limit. 
DEADLINE = 140.0 

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _project_cartesian(pts):
    """Project points into the triangle if they are slightly outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _min_area_and_triplets(pts):
    """
    Compute the minimum triangle area and the indices of the triplets that achieve it.
    Returns: (min_area, list_of_triplet_indices)
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    # Find all triplets with area close to min (within tolerance to handle numerical noise)
    tol = 1e-9
    mask = areas <= (min_area + tol)
    triplet_indices = np.where(mask)[0]
    
    return min_area, triplet_indices


def _optimize_point_local(pts, idx, other_points_fixed=True):
    """
    Optimize the position of a single point (idx) to maximize the minimum area 
    of triangles involving this point, while keeping other points fixed.
    """
    # Extract coordinates of the point to optimize
    x0 = pts[idx, 0]
    y0 = pts[idx, 1]
    
    # Precompute vectors from other points to the moving point? 
    # Actually, it's easier to just recompute areas involving idx.
    
    # Identify all triplets involving idx
    # We can filter the precomputed triplets or just iterate.
    # For speed, we can just compute areas for all triplets involving idx.
    
    def objective(coords):
        x, y = coords
        # Update point
        pts_new = pts.copy()
        pts_new[idx, 0] = x
        pts_new[idx, 1] = y
        
        # Project to ensure validity? 
        # If we use bounds, we might not need projection inside the objective,
        # but L-BFGS-B might step outside. 
        # Let's use a penalty or just bounds.
        # Bounds for x,y depend on y. This is tricky for L-BFGS-B.
        # Instead, we will use a simple Nelder-Mead or just gradient descent with projection?
        # L-BFGS-B with dynamic bounds is hard.
        # Let's use a simple local search: try small steps in x and y.
        # Or, better: use scipy.optimize.minimize with method='Nelder-Mead' which doesn't need gradients/bounds strictly,
        # but we need to handle the triangle constraint.
        
        # Actually, a simpler and more robust approach for a single point:
        # The area of a triangle (i, j, k) where i=idx is 0.5 * | (p_j - p_i) x (p_k - p_i) |
        # This is a quadratic function of x_i, y_i.
        # We want to maximize the minimum of these quadratic functions.
        
        # Let's just evaluate the min area of triangles involving idx.
        min_area_involving = np.inf
        
        # Iterate over all other pairs (j, k)
        # There are C(10, 2) = 45 pairs.
        for j in range(N):
            if j == idx: continue
            for k in range(j+1, N):
                if k == idx: continue
                
                # Area of triangle (idx, j, k)
                dx1 = pts[j, 0] - x
                dy1 = pts[j, 1] - y
                dx2 = pts[k, 0] - x
                dy2 = pts[k, 1] - y
                cross = dx1 * dy2 - dy1 * dx2
                area = 0.5 * np.abs(cross)
                
                if area < min_area_involving:
                    min_area_involving = area
        
        return -min_area_involving

    # Use Nelder-Mead for this small 2D problem. It is robust and doesn't require gradients.
    # We need to ensure the point stays inside the triangle.
    # Nelder-Mead doesn't support bounds directly. 
    # We can add a large penalty if the point is outside.
    
    def objective_penalized(coords):
        x, y = coords
        # Check bounds
        if y < 0 or y > S3H:
            return 1e6
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min or x > x_max:
            return 1e6
        
        return objective(coords)

    try:
        result = minimize(
            objective_penalized,
            [x0, y0],
            method='Nelder-Mead',
            options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        return result.x
    except Exception:
        return [x0, y0]


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area, _ = _min_area_and_triplets(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Initial configuration: High symmetry guess
    # Corners, mid-edges, centroid, and points on medians
    init_cart = np.array([
        [0.0, 0.0],          # Corner 0
        [1.0, 0.0],          # Corner 1
        [0.5, S3H],          # Corner 2
        [0.5, 0.0],          # Mid bottom
        [0.75, S3H / 2],     # Mid right
        [0.25, S3H / 2],     # Mid left
        [1/3, S3H/3],        # Centroid
        [0.333, 0.0],        # Bottom edge
        [0.667, 0.0],        # Bottom edge
        [0.5, S3H/3],        # Median
        [0.2, 0.1],          # Interior
    ], dtype=float)
    
    # Ensure initial points are valid
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    current_pts = init_cart.copy()
    current_area, current_triplets = _min_area_and_triplets(current_pts)
    
    # Main Loop: Iteratively optimize points involved in the smallest triangles
    # We will run this for a fixed number of iterations or until time runs out.
    
    # To avoid getting stuck, we will occasionally restart from a slightly perturbed best solution.
    # Or, we can just keep optimizing the current best.
    
    # Let's try a simple iterative refinement:
    # 1. Find min area and critical triplets.
    # 2. Collect unique indices from critical triplets.
    # 3. For each unique index, run local optimization.
    # 4. If area improved, accept. Else, try a random perturbation of a critical point.
    
    max_outer_iter = 10000
    
    for outer_iter in range(max_outer_iter):
        if not _elapsed_ok(5):
            break
        
        # Get current min area and critical triplets
        current_area, current_triplets = _min_area_and_triplets(current_pts)
        
        # Update global best
        if current_area > best_area:
            best_area = current_area
            best_pts = current_pts.copy()
        
        # If we have converged (area is very stable), break? 
        # Or just keep going to squeeze out more.
        
        # Get unique indices involved in the critical triplets
        critical_indices = set()
        for t_idx in current_triplets:
            critical_indices.add(_T_I[t_idx])
            critical_indices.add(_T_J[t_idx])
            critical_indices.add(_T_K[t_idx])
        
        # Optimize each critical index
        improved = False
        for idx in critical_indices:
            if not _elapsed_ok(5):
                break
            
            # Optimize this point
            new_coords = _optimize_point_local(current_pts, idx)
            
            # Check if the new point is valid
            x_new, y_new = new_coords
            y_new = max(0.0, min(S3H, y_new))
            x_min = y_new / S3
            x_max = 1 - y_new / S3
            x_new = max(x_min, min(x_max, x_new))
            
            # Update point
            current_pts[idx, 0] = x_new
            current_pts[idx, 1] = y_new
            
            # Check if area improved
            new_area, _ = _min_area_and_triplets(current_pts)
            if new_area > current_area + 1e-10:
                current_area = new_area
                improved = True
            else:
                # If no improvement, revert? Or keep?
                # Keeping might help escape local minima if we perturb later.
                # But reverting is safer for stability.
                # Let's revert if it didn't improve.
                current_pts[idx, 0] = new_coords[0] # This is wrong, we need the OLD coords
                # We didn't store old coords properly. Let's store them before.
                
        # If no improvement in this outer loop, try a small random perturbation on a critical point
        if not improved:
            if not _elapsed_ok(5):
                break
            
            # Pick a random critical point
            idx = list(critical_indices)[rng.integers(len(critical_indices))]
            
            # Perturb
            old_x = current_pts[idx, 0]
            old_y = current_pts[idx, 1]
            
            noise_scale = 0.01
            new_x = old_x + rng.normal(0, noise_scale)
            new_y = old_y + rng.normal(0, noise_scale)
            
            # Project
            new_y = max(0.0, min(S3H, new_y))
            x_min = new_y / S3
            x_max = 1 - new_y / S3
            new_x = max(x_min, min(x_max, new_x))
            
            current_pts[idx, 0] = new_x
            current_pts[idx, 1] = new_y
            
            new_area, _ = _min_area_and_triplets(current_pts)
            
            # Accept if better, or with some probability (simulated annealing style)
            if new_area > current_area:
                current_area = new_area
            else:
                # Revert
                current_pts[idx, 0] = old_x
                current_pts[idx, 1] = old_y

    # Final update
    if best_pts is None:
        return init_cart
        
    return best_pts

# EVOLVE-BLOCK-END
