"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(pos):
    """
    Compute the largest common radius for the given positions.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    
    x, y = pos[:, 0], pos[:, 1]
    d_left = x + 0.5
    d_right = 0.5 - x
    d_bottom = y + 0.5
    d_top = 0.5 - y
    
    d_wall = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0)
    min_wall_dist = np.min(d_wall)
    
    dists = cdist(pos, pos, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    if min_wall_dist < 0:
        return -1.0
    if min_pair_dist < 0:
        return -1.0
        
    return r

def _objective_and_grad(flat_pos, beta=100.0):
    """
    Objective function for minimization.
    We want to maximize the minimum distance.
    Equivalent to minimizing -min_distance.
    To handle the non-smoothness of min and the hard constraints,
    we use a soft penalty (log-barrier style) for violations.
    
    Let d_ij be pairwise distances.
    Let d_wall_i be distance to walls.
    
    We define a "soft minimum" function S(x) = -1/beta * log(sum(exp(-beta*x))).
    This approximates min(x) as beta -> infinity.
    It is smooth and differentiable.
    
    We want to maximize S(pairwise_dists) and S(wall_dists).
    So we minimize -S(pairwise_dists) - S(wall_dists) + penalty for being out of bounds?
    Actually, if we enforce the points to stay in [-0.5, 0.5] via bounds in L-BFGS-B,
    we don't need a penalty for boundary, but we still need to maximize the distance to the boundary
    to allow for a larger radius.
    
    Wait, the radius is limited by:
    1. min_pair_dist / 2
    2. min_wall_dist
    
    So we want to maximize R = min( min_pair_dist/2, min_wall_dist ).
    
    Let's define the objective to minimize:
    f = -S( min_pair_dist/2 ) - S( min_wall_dist )
    
    Where S(x) is the soft-min.
    
    However, computing the full distance matrix for every gradient evaluation is O(N^2).
    N=200, N^2=40000. This is fast.
    L-BFGS-B might take many iterations.
    
    Alternative simpler objective:
    Minimize sum of "violations" or maximize sum of distances? No, max-min is key.
    
    Let's use the soft-min approach.
    beta controls the sharpness.
    
    Variables: pos (N, 2) flattened to (2N,).
    
    Constraints:
    -0.5 <= x_i <= 0.5
    -0.5 <= y_i <= 0.5
    
    We will use bounds in L-BFGS-B.
    """
    pos = flat_pos.reshape(N, 2)
    
    # Pairwise distances
    # To save memory and time, we can compute the upper triangle only?
    # cdist is fine for N=200.
    dists = cdist(pos, pos, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    
    # We only care about the lower part of the distribution of distances?
    # Soft min of all pairwise distances.
    # S_pair = -1/beta * log(sum(exp(-beta * dists/2)))  # Note: radius is dist/2
    # To avoid underflow/overflow with exp(-beta * x), we use log-sum-exp trick.
    # log(sum(exp(A))) = max(A) + log(sum(exp(A - max(A))))
    
    # Let's scale the distances to be around 1 to help numerical stability?
    # Typical distance ~ 0.07 (since r~0.036).
    # beta = 100. beta * 0.07 = 7. exp(-7) is small. exp(7) is large.
    # If we minimize -S, we want S to be large.
    # S(x) approx min(x).
    # We want to maximize min(dist/2) and min(wall_dist).
    
    # Let's define the objective as:
    # obj = - S_pair - S_wall
    # where S_pair = soft_min(dists/2)
    # S_wall = soft_min(wall_dists)
    
    # Gradient calculation:
    # d/dx S(x) = sum(exp(-beta*x_i) * (-beta)) / sum(exp(-beta*x_j)) ... wait.
    # S(x) = -1/beta * log( sum( exp(-beta * x_i) ) )
    # dS/dx_k = -1/beta * ( 1 / sum(exp) ) * ( exp(-beta*x_k) * (-beta) )
    # dS/dx_k = exp(-beta*x_k) / sum(exp(-beta*x_j))
    # This is a softmax-like weight, but with negative exponent.
    # It gives higher weight to smaller x_k (since exp(-beta*small) is larger).
    # So the gradient points in the direction that increases the smallest values.
    # We want to MAXIMIZE S. So we MINIMIZE -S.
    # grad(-S) = - grad(S).
    
    # Let's compute the weights for pairwise distances.
    # We need to be careful with the sum over all pairs.
    
    # Vectorized computation for stability
    # dists is (N, N). Diagonal is inf.
    # We want soft_min of all off-diagonal elements.
    
    # To avoid issues with inf in exp, we ignore diagonal.
    # Let's flatten the upper triangle to avoid double counting?
    # Or just use the whole matrix with diagonal masked.
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_masked = np.where(mask, dists, np.inf)
    
    # Scale for soft min
    # We want to maximize min(dists/2) and min(wall_dists).
    # Let's combine them?
    # No, better to treat them separately and sum the objectives.
    # But we can also just concatenate the vectors of distances and wall distances
    # and take the soft min of the combined set?
    # R = min( min_pair/2, min_wall ).
    # This is equivalent to min( all_pair_dists/2, all_wall_dists ).
    # So we can create a single vector V containing all pair distances (divided by 2) and all wall distances.
    # Then S = soft_min(V).
    # Maximize S.
    
    # Construct V
    # Pair distances:
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0  # (N*(N-1)/2,)
    
    # Wall distances
    x, y = pos[:, 0], pos[:, 1]
    wall_dists = np.stack([
        x + 0.5,
        0.5 - x,
        y + 0.5,
        0.5 - y
    ], axis=1).min(axis=1)  # (N,)
    
    # Combine
    V = np.concatenate([pair_dists, wall_dists])
    
    # Soft min
    # S = -1/beta * log(sum(exp(-beta * V)))
    # To compute this and its gradient stably:
    # Let M = min(V).
    # log(sum(exp(-beta*V))) = -beta*M + log(sum(exp(-beta*(V-M))))
    # S = M - 1/beta * log(sum(exp(-beta*(V-M))))
    
    # This is numerically stable if M is the min.
    # But we need the gradient w.r.t. positions.
    # dS/dV_k = exp(-beta*(V_k - M)) / sum(exp(-beta*(V_j - M)))
    # Let w_k = dS/dV_k. Sum(w_k) = 1.
    
    # Now we need dV_k / dpos.
    # This is sparse.
    # For a pair (i,j), V_k = 0.5 * ||p_i - p_j||.
    # dV_k / dp_i = 0.5 * (p_i - p_j) / ||p_i - p_j||
    # dV_k / dp_j = 0.5 * (p_j - p_i) / ||p_i - p_j||
    
    # For a wall dist for point i, V_k = min(x_i+0.5, ...).
    # This is not differentiable at the corners of the square or when multiple walls are equally close.
    # However, inside the square, it's linear.
    # The min is differentiable almost everywhere.
    # We can approximate the gradient by picking the active wall.
    # Or, since we are using L-BFGS-B which handles bounds, maybe we don't need to maximize wall distance
    # if we just push points away from walls?
    # No, the radius is limited by the wall distance. We must maximize it.
    
    # Let's compute the gradient of the objective f = -S(V) w.r.t. pos.
    # grad_f = - sum_k w_k * dV_k/dpos
    
    # Initialize gradient array
    grad_pos = np.zeros_like(pos)
    
    # 1. Pairwise contributions
    # w_pair is the weight for each pair in V.
    # V = [pair_dists, wall_dists]
    # n_pairs = len(pair_dists)
    n_pairs = len(pair_dists)
    
    # Compute weights w
    # V_shifted = V - min(V)
    V_min = np.min(V)
    V_shifted = V - V_min
    # exp_terms = exp(-beta * V_shifted)
    # To avoid underflow, we can just use exp(-beta * V_shifted).
    # Since V_shifted >= 0, exp(-beta * V_shifted) <= 1.
    # The max term is 1 (for the min element).
    # Sum will be at least 1.
    exp_terms = np.exp(-beta * V_shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    # S = V_min - (1/beta) * log(sum_exp)
    # We don't need S value for gradient, but maybe for objective value.
    # Objective to minimize: -S
    # obj = - (V_min - 1/beta * log(sum_exp))
    
    # Gradient of S w.r.t V is w.
    # So Gradient of -S w.r.t V is -w.
    
    # Now accumulate gradient w.r.t pos.
    # For each pair k (index in 0..n_pairs-1):
    # weight = w[k]
    # dV_k/dp_i = 0.5 * (p_i - p_j) / d_ij
    # dV_k/dp_j = 0.5 * (p_j - p_i) / d_ij
    # Contribution to grad_pos[i] += -w[k] * 0.5 * (p_i - p_j) / d_ij
    # Contribution to grad_pos[j] += -w[k] * 0.5 * (p_j - p_i) / d_ij
    
    # Vectorized pair gradient
    # iu[0] contains i, iu[1] contains j
    i_idx, j_idx = iu
    # d_ij is pair_dists * 2
    d_ij = pair_dists * 2.0
    # Avoid division by zero
    d_ij_safe = np.where(d_ij < 1e-10, 1e-10, d_ij)
    
    # Vector from j to i
    vec = pos[i_idx] - pos[j_idx]
    # Unit vector
    unit_vec = vec / d_ij_safe[:, np.newaxis]
    
    # dV/dp_i = 0.5 * unit_vec
    # dV/dp_j = -0.5 * unit_vec
    
    # Weighted sum
    # weights for pairs are w[:n_pairs]
    w_pairs = w[:n_pairs]
    
    # grad_pair_i = sum_k w_k * 0.5 * unit_vec_k  (added to pos[i])
    # grad_pair_j = sum_k w_k * (-0.5) * unit_vec_k (added to pos[j])
    
    # We need to scatter these.
    # Using np.add.at for efficiency and correctness with repeated indices.
    
    # Contribution to i
    contrib_i = w_pairs[:, np.newaxis] * 0.5 * unit_vec
    np.add.at(grad_pos, i_idx, contrib_i)
    
    # Contribution to j
    contrib_j = -w_pairs[:, np.newaxis] * 0.5 * unit_vec
    np.add.at(grad_pos, j_idx, contrib_j)
    
    # 2. Wall contributions
    # w_wall = w[n_pairs:]
    w_walls = w[n_pairs:]
    
    # For each point i, the wall distance is min of 4 linear functions.
    # The gradient of min is the gradient of the active function.
    # If there's a tie, it's the average of the gradients? Or just pick one?
    # For L-BFGS, picking one is usually fine, or we can use a subgradient.
    # Let's pick the active wall (the one that gives the min distance).
    # If multiple, we can average them to be smoother.
    
    # Calculate distances to each wall
    dx_l = x + 0.5
    dx_r = 0.5 - x
    dy_b = y + 0.5
    dy_t = 0.5 - y
    
    d_walls = np.stack([dx_l, dx_r, dy_b, dy_t], axis=1) # (N, 4)
    
    # Find active walls (indices where d_walls == min)
    # min_val = d_walls.min(axis=1)
    # active_mask = (d_walls == min_val[:, np.newaxis])
    
    # To make it smooth, we can use the same softmax trick on the wall distances?
    # But wall distances are already part of V.
    # The weight w_wall[i] is the weight assigned to the min wall distance of point i in the global soft-min.
    # Wait, V contains the *minimum* wall distance for each point.
    # So V_wall_i = min(dx_l, dx_r, dy_b, dy_t).
    # The derivative of V_wall_i w.r.t pos_i is the gradient of the active wall.
    
    # Let's determine the active wall for each point.
    # To handle ties smoothly, we can compute a weighted average of the wall gradients
    # based on a local softmax?
    # Or just pick the argmin.
    
    # Let's pick the argmin.
    active_idx = np.argmin(d_walls, axis=1) # (N,)
    
    # Gradients of walls:
    # Wall 0 (Left, x+0.5): grad = (1, 0)
    # Wall 1 (Right, 0.5-x): grad = (-1, 0)
    # Wall 2 (Bottom, y+0.5): grad = (0, 1)
    # Wall 3 (Top, 0.5-y): grad = (0, -1)
    
    grad_walls = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0]
    ])
    
    # dV_wall_i / dp_i = grad_walls[active_idx[i]]
    # Contribution to grad_pos[i] += -w_wall[i] * grad_walls[active_idx[i]]
    
    contrib_walls = -w_walls[:, np.newaxis] * grad_walls[active_idx]
    grad_pos += contrib_walls
    
    # Total gradient of -S w.r.t pos is grad_pos.
    # Wait, I calculated dS/dV = w.
    # So d(-S)/dV = -w.
    # d(-S)/dpos = sum_k (-w_k) * dV_k/dpos.
    # My accumulation above used:
    # contrib_i = w_pairs * ... (this is dS/dpos)
    # So I need to negate the final grad_pos?
    # Let's re-verify.
    # Objective f = -S.
    # df/dpos = - dS/dpos.
    # dS/dpos = sum w_k dV_k/dpos.
    # So df/dpos = - sum w_k dV_k/dpos.
    
    # In my code:
    # contrib_i (for pairs) was w_pairs * 0.5 * unit_vec. This is dS/dp_i part.
    # So I should negate it.
    # Let's just return -grad_pos as the gradient of the objective.
    
    # Objective value:
    # S = V_min - 1/beta * log(sum_exp)
    # f = -S
    obj = -(V_min - (1.0/beta) * np.log(sum_exp))
    
    # Flatten gradient
    grad_flat = (-grad_pos).flatten()
    
    return obj, grad_flat

def _hexagonal_init(n):
    """
    Initialize points in a hexagonal lattice pattern scaled to fit the square.
    """
    s = np.sqrt(2.0 / (np.sqrt(3.0) * n))
    y_step = s * np.sqrt(3.0) / 2.0
    
    n_rows = int(np.ceil(1.0 / y_step)) + 1
    n_cols = int(np.ceil(1.0 / s)) + 1
    
    # Generate grid
    pts = []
    # Center the grid
    # We'll generate indices relative to center
    # x = i * s + (j % 2) * s/2
    # y = j * y_step
    
    # Let's generate a large enough grid and center it
    # To ensure we get exactly n points, we can generate more and select the closest n to the center?
    # Or just generate a grid that fits and if we have more, remove random ones?
    # If we have fewer, add random ones?
    
    # Better: Generate a grid covering the square, then if > n, keep the n closest to the center?
    # No, we want them spread out.
    
    # Let's try a simple approach:
    # Generate a grid of size (n_rows, n_cols) centered at 0.
    # If count > n, randomly remove from boundary?
    # If count < n, add random points?
    
    # Actually, for N=200, a square grid of 14x14 = 196 is close.
    # 15x15 = 225.
    # Let's just use a square grid for simplicity and robustness in init,
    # then let the optimizer do the work.
    # Hexagonal is better, but let's make sure it's correct.
    
    # Let's generate points in a hex lattice
    # and then select the n points that are within the square [-0.5, 0.5]?
    # If we have more than n, we have a problem.
    # If we have less, we have a problem.
    
    # Let's estimate the number of points in a hex lattice in a unit square.
    # Area = 1.
    # Points per area = 1 / (s^2 * sqrt(3)/2) = n.
    # So it should be roughly n.
    
    # Let's generate a grid of size (n_rows, n_cols)
    # and then if we have too many, we can randomly drop some.
    # If too few, add some.
    
    # To be safe and simple, let's use a random initialization for now,
    # or a very simple grid.
    # The optimizer is strong, so init doesn't need to be perfect.
    
    # Let's use a square grid.
    m = int(np.ceil(np.sqrt(n)))
    # Adjust m to get close to n
    # If m*m > n, we can remove some.
    # If m*m < n, we can add some.
    
    # Let's just generate m x m grid.
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) > n:
        # Randomly select n
        idx = np.random.choice(len(grid), n, replace=False)
        grid = grid[idx]
    elif len(grid) < n:
        # Add random points
        extra = np.random.uniform(-0.49, 0.49, (n - len(grid), 2))
        grid = np.vstack([grid, extra])
        
    # Add small random jitter to break symmetry
    grid += np.random.uniform(-0.01, 0.01, grid.shape)
    
    # Clip to bounds
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    best_pos = None
    best_r = -1.0
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > deadline - 30.0:
            break
            
        pos = _hexagonal_init(N)
        
        # Flatten
        flat_pos = pos.flatten()
        
        # Bounds
        bounds = [(-0.499, 0.499)] * (2 * N)
        
        # Optimize
        # L-BFGS-B is good for box constraints
        # maxiter should be enough
        # time limit per optimization?
        # We can't easily interrupt minimize, but we can set a low maxiter
        # or use a callback to check time.
        
        # Let's use a callback to stop if time is running out.
        stop_flag = [False]
        
        def callback(xk):
            if time.time() > deadline - 10.0:
                stop_flag[0] = True
            return stop_flag[0]
            
        try:
            res = minimize(
                _objective_and_grad,
                flat_pos,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
                callback=callback
            )
            
            pos_opt = res.x.reshape(N, 2)
            r = _compute_radius(pos_opt)
            
            if r > best_r:
                best_r = r
                best_pos = pos_opt.copy()
                
        except Exception:
            # If optimization fails, use the initial position
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
        
    # If we still have time, do a final polish
    if best_pos is not None and time.time() < deadline - 10.0:
        # Run one more optimization from the best position found
        flat_pos = best_pos.flatten()
        bounds = [(-0.499, 0.499)] * (2 * N)
        
        stop_flag = [False]
        def callback(xk):
            if time.time() > deadline - 5.0:
                stop_flag[0] = True
            return stop_flag[0]
            
        try:
            res = minimize(
                _objective_and_grad,
                flat_pos,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10},
                callback=callback
            )
            pos_opt = res.x.reshape(N, 2)
            r = _compute_radius(pos_opt)
            if r > best_r:
                best_r = r
                best_pos = pos_opt.copy()
        except Exception:
            pass

    if best_pos is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_pos

# EVOLVE-BLOCK-END
