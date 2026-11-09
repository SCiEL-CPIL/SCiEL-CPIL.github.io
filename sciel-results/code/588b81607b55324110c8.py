import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _fcc_lattice_init(n_points=100):
    """
    Generate points on an FCC lattice.
    """
    # Generate a grid of FCC points
    # FCC points in a unit cell [0,1]^3:
    # (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # We need to scale this to fit the cube [-0.5, 0.5]^3
    
    # Try different lattice constants to get close to 100 points
    # Density of FCC is ~0.74
    # Volume per point ~ 1/0.74 ~ 1.35
    # Side length ~ 1.1
    # But we are constrained to side 1.
    # So we might have to compress or just take a subset.
    
    # Let's generate a 4x4x4 grid of FCC unit cells.
    # This gives 4*4*4*4 = 256 points in a 4x4x4 volume?
    # If unit cell size is 1, the box is 4x4x4.
    # We need to map this to [-0.5, 0.5]^3.
    # Scale factor: 1.0 / 4.0 = 0.25.
    
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
    
    # Generate grid indices
    grid = np.mgrid[0:4, 0:4, 0:4]
    grid_points = np.stack(grid, axis=-1).reshape(-1, 3)
    
    # Add basis vectors
    all_points = []
    for p in grid_points:
        for b in basis:
            pt = p + b
            all_points.append(pt)
            
    all_points = np.array(all_points)
    
    # Scale to fit [-0.5, 0.5]^3
    # The points are in [0, 4]^3 roughly.
    # Center is at 2, 2, 2.
    # Scale: we want the range to be approx 1.0.
    # Current range is approx 4.0.
    # So scale by 0.25.
    # But we want to center it.
    
    # Shift to center
    all_points = all_points - 2.0
    
    # Scale
    all_points = all_points * 0.25
    
    # Now points are in [-0.5, 0.5]^3
    # We have 256 points. We need 100.
    # We should select the 100 points that are best distributed or just take the first 100?
    # Taking the first 100 might be biased.
    # Better: use a greedy farthest point sampling or just random subsampling with jitter?
    # Or, adjust the lattice constant to get closer to 100 points.
    
    # Let's try a 3x3x3 grid of FCC cells?
    # 3*3*3*4 = 108 points. This is very close to 100.
    # Let's use 3x3x3.
    
    grid = np.mgrid[0:3, 0:3, 0:3]
    grid_points = np.stack(grid, axis=-1).reshape(-1, 3)
    
    all_points = []
    for p in grid_points:
        for b in basis:
            pt = p + b
            all_points.append(pt)
            
    all_points = np.array(all_points)
    
    # Center at 1.5, 1.5, 1.5
    all_points = all_points - 1.5
    
    # Scale to fit [-0.5, 0.5]^3
    # Range is approx 3.0.
    # Scale by 1.0 / 3.0 = 0.333
    # But we need to ensure they fit inside.
    # The max coord is 1.5 * 0.333 = 0.5.
    # So they fit exactly.
    
    all_points = all_points * (1.0 / 3.0)
    
    # We have 108 points. We need 100.
    # We can remove 8 points.
    # Which ones to remove?
    # The corners are most constrained.
    # Or just randomly remove 8 with a fixed seed.
    # Or, better, use a simple optimization to pick the best 100?
    # For initialization, random removal with jitter is fine.
    
    np.random.seed(42)
    if len(all_points) > N:
        indices = np.random.choice(len(all_points), N, replace=False)
        points = all_points[indices]
    else:
        # If fewer, we need to add points?
        # 108 > 100, so we are good.
        points = all_points[:N]
        
    # Add small jitter to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    # Clip to cube
    points = np.clip(points, -0.5, 0.5)
    
    return points

def _soft_min_radius(centres, beta=50.0):
    """
    Compute the soft-min radius and its gradient.
    """
    n = len(centres)
    # Pairwise distances
    # Use broadcasting to compute all pairwise distances
    # diff[i, j] = centres[i] - centres[j]
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle to avoid double counting, 
    # but for soft min, double counting just changes the weight distribution slightly.
    # It's safer to use unique pairs.
    
    # Get indices of upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Wall distances
    # dist to -0.5: centres + 0.5
    # dist to 0.5: 0.5 - centres
    wall_dists = np.min(np.minimum(centres + 0.5, 0.5 - centres), axis=1)
    
    # Combine all constraints
    # s_k = min(dist/2, wall_dist)
    # We want to maximize min(s_k).
    # Soft min: -1/beta * log(sum(exp(-beta * s_k)))
    
    # Values to minimize (we want to maximize min, so we minimize -min)
    # Let f(x) = -soft_min_radius(x). We minimize f.
    # soft_min_radius = -1/beta * log(sum(exp(-beta * s_k)))
    # -soft_min_radius = 1/beta * log(sum(exp(-beta * s_k)))
    
    # s_k are:
    # 1. pair_dists / 2
    # 2. wall_dists
    
    s_pairs = pair_dists / 2.0
    s_walls = wall_dists
    
    s_all = np.concatenate([s_pairs, s_walls])
    
    # Calculate weights for soft min
    # w_k = exp(-beta * s_k) / sum(exp(-beta * s_j))
    # To avoid overflow, subtract min s (which makes exp(-beta * min_s) = 1? No.)
    # We want to compute log(sum(exp(-beta * s_k))).
    # Let m = min(s_k).
    # log(sum(exp(-beta * s_k))) = -beta * m + log(sum(exp(-beta * (s_k - m))))
    
    m = np.min(s_all)
    exp_terms = np.exp(-beta * (s_all - m))
    sum_exp = np.sum(exp_terms)
    
    log_sum_exp = -beta * m + np.log(sum_exp)
    
    soft_min_val = -1.0 / beta * log_sum_exp
    
    # Gradient
    # d/dx soft_min_val = 1/beta * (1/sum_exp) * sum(d/dx exp(-beta * s_k))
    # d/dx exp(-beta * s_k) = exp(-beta * s_k) * (-beta * ds_k/dx)
    # So d/dx soft_min_val = 1/beta * (1/sum_exp) * sum(exp(-beta * s_k) * (-beta * ds_k/dx))
    # = -1/sum_exp * sum(exp(-beta * s_k) * ds_k/dx)
    # = - sum(w_k * ds_k/dx)
    # where w_k = exp(-beta * s_k) / sum_exp
    
    weights = exp_terms / sum_exp
    
    grad = np.zeros((n, 3))
    
    # Pair gradients
    # ds_k/dx_i = 0.5 * (x_i - x_j) / dist_ij
    # ds_k/dx_j = 0.5 * (x_j - x_i) / dist_ij
    
    # w_pairs are the first len(pair_dists) weights
    w_pairs = weights[:len(pair_dists)]
    
    # unit_vecs = (x_i - x_j) / dist_ij
    # We have dists matrix, but we need the vectors.
    # diff[i_idx, j_idx] gives x_i - x_j
    diff_vecs = diff[i_idx, j_idx]
    # dists are in pair_dists
    # Avoid division by zero
    safe_dists = np.where(pair_dists < 1e-10, 1e-10, pair_dists)
    unit_vecs = diff_vecs / safe_dists[:, np.newaxis]
    
    # Contribution to i: 0.5 * unit_vec
    # Contribution to j: -0.5 * unit_vec
    
    contrib_i = 0.5 * unit_vecs * w_pairs[:, np.newaxis]
    contrib_j = -0.5 * unit_vecs * w_pairs[:, np.newaxis]
    
    np.add.at(grad, i_idx, contrib_i)
    np.add.at(grad, j_idx, contrib_j)
    
    # Wall gradients
    # ds_k/dx_i for wall distance
    # w_k = min(0.5 - x, x + 0.5)
    # If x < 0, w = x + 0.5, grad = 1
    # If x > 0, w = 0.5 - x, grad = -1
    # At x = 0, it's non-differentiable, but soft min handles it.
    # Actually, we used min of the two, so the gradient is the gradient of the active constraint.
    # In the soft min formulation, we sum over ALL wall distances (both + and - for each dim).
    # So we have 6N wall constraints.
    
    # Wait, in my s_walls calculation, I took min over the 6 walls for each point.
    # This is correct for the "radius" definition: r = min(min_pair, min_wall).
    # But the soft min is over the set of all constraints.
    # The set of constraints is:
    # 1. d_ij / 2 for all pairs
    # 2. 0.5 - x_i, x_i + 0.5, 0.5 - y_i, ... for all points.
    
    # So I should NOT take min over walls first.
    # I should include all 6N wall distances in s_all.
    
    # Let's recalculate s_walls to be the full 6N vector.
    
    wall_dists_full = np.zeros((n, 6))
    wall_dists_full[:, 0] = 0.5 - centres[:, 0]
    wall_dists_full[:, 1] = centres[:, 0] + 0.5
    wall_dists_full[:, 2] = 0.5 - centres[:, 1]
    wall_dists_full[:, 3] = centres[:, 1] + 0.5
    wall_dists_full[:, 4] = 0.5 - centres[:, 2]
    wall_dists_full[:, 5] = centres[:, 2] + 0.5
    
    s_walls_full = wall_dists_full.flatten()
    
    # Redo s_all
    s_all = np.concatenate([s_pairs, s_walls_full])
    
    m = np.min(s_all)
    exp_terms = np.exp(-beta * (s_all - m))
    sum_exp = np.sum(exp_terms)
    
    log_sum_exp = -beta * m + np.log(sum_exp)
    soft_min_val = -1.0 / beta * log_sum_exp
    
    weights = exp_terms / sum_exp
    
    # Redo gradients with correct wall weights
    
    grad = np.zeros((n, 3))
    
    # Pair gradients (same as before)
    w_pairs = weights[:len(pair_dists)]
    diff_vecs = diff[i_idx, j_idx]
    safe_dists = np.where(pair_dists < 1e-10, 1e-10, pair_dists)
    unit_vecs = diff_vecs / safe_dists[:, np.newaxis]
    
    contrib_i = 0.5 * unit_vecs * w_pairs[:, np.newaxis]
    contrib_j = -0.5 * unit_vecs * w_pairs[:, np.newaxis]
    
    np.add.at(grad, i_idx, contrib_i)
    np.add.at(grad, j_idx, contrib_j)
    
    # Wall gradients
    w_walls = weights[len(pair_dists):]
    w_walls_2d = w_walls.reshape(n, 6)
    
    # Gradients for each wall constraint
    # 0: 0.5 - x -> grad (-1, 0, 0)
    # 1: x + 0.5 -> grad (1, 0, 0)
    # 2: 0.5 - y -> grad (0, -1, 0)
    # 3: y + 0.5 -> grad (0, 1, 0)
    # 4: 0.5 - z -> grad (0, 0, -1)
    # 5: z + 0.5 -> grad (0, 0, 1)
    
    grad_vecs = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1]
    ])
    
    grad_wall_points = w_walls_2d @ grad_vecs
    
    grad += grad_wall_points
    
    # The gradient of soft_min_val is -sum(w_k * ds_k/dx)
    # So grad_soft_min = -grad
    # We want to maximize soft_min_val, so we move in direction grad_soft_min.
    # Or minimize -soft_min_val, so we move in direction -grad_soft_min = grad.
    
    # Let's return the gradient of the objective we want to MINIMIZE.
    # Objective: f(x) = -soft_min_val(x)
    # grad_f = -grad_soft_min = grad
    
    return -soft_min_val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization
    points = _fcc_lattice_init(N)
    
    # 2. Optimization Loop
    beta = 10.0
    step_size = 0.01
    best_radius = -np.inf
    best_points = points.copy()
    
    # Precompute indices for speed
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def compute_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists[i_idx, j_idx])
        
        wall_dists = np.min(np.minimum(pts + 0.5, 0.5 - pts), axis=1)
        min_wall = np.min(wall_dists)
        
        return min(min_pair / 2.0, min_wall)
    
    current_radius = compute_radius(points)
    if current_radius > best_radius:
        best_radius = current_radius
        best_points = points.copy()
        
    while time.time() < deadline:
        # Compute soft min and gradient
        obj, grad = _soft_min_radius(points, beta)
        
        # Update
        points += step_size * grad
        
        # Clip to cube
        points = np.clip(points, -0.5, 0.5)
        
        # Evaluate
        current_radius = compute_radius(points)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Adaptive parameters
        beta *= 1.05
        step_size *= 0.99
        
        # If step_size is very small, break or restart?
        if step_size < 1e-5:
            break
            
    # 3. Polish with SciPy
    # Use the best points found so far
    points = best_points.copy()
    
    # Set a high beta for polishing
    beta_final = 100.0
    
    def objective(x):
        x = x.reshape(N, 3)
        val, _ = _soft_min_radius(x, beta_final)
        return val
    
    def gradient(x):
        x = x.reshape(N, 3)
        _, grad = _soft_min_radius(x, beta_final)
        return grad.flatten()
        
    # Flatten points
    x0 = points.flatten()
    
    # Use L-BFGS-B
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10})
        if res.success:
            points = res.x.reshape(N, 3)
            current_radius = compute_radius(points)
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()
    except Exception as e:
        pass
        
    return best_points
