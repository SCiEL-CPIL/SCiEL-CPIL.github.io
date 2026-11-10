"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    d = pdist(pts, 'euclidean')
    min_pair = d.min() if len(d) > 0 else 0.0
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r

def _smooth_min_distance(pts, beta):
    """
    Compute a smooth approximation of the minimum distance metric.
    We want to maximize r = min(d_ij/2, wall_dist_i).
    Let g_i be the values: d_ij/2 for all pairs, and wall_dist_i for all points.
    We approximate min(g) using the soft-min: -1/beta * log(sum(exp(-beta * g_i))).
    As beta -> inf, this approaches the true min.
    """
    p = pts
    
    # 1. Pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, 0.0)
    
    # We only care about i < j to avoid double counting and self-distance
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[i_idx, j_idx]
    pair_dists = np.sqrt(pair_dists_sq)
    pair_metrics = pair_dists / 2.0
    
    # 2. Wall distances
    # Distance to nearest wall for each point
    # max(|x|, |y|) gives distance to nearest boundary in Chebyshev sense?
    # No, distance to boundary is 0.5 - max(|x|, |y|) is NOT correct for Euclidean circle.
    # The constraint is x^2 + y^2 is not the constraint. The constraint is |x| <= 0.5 - r AND |y| <= 0.5 - r.
    # So distance to wall for point i is min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i).
    # Since pts are in [-0.5, 0.5], this is 0.5 - max(|x_i|, |y_i|).
    abs_pts = np.abs(p)
    max_abs = np.max(abs_pts, axis=1)
    wall_metrics = 0.5 - max_abs
    
    # Combine all metrics
    # Total metrics vector
    all_metrics = np.concatenate([pair_metrics, wall_metrics])
    
    # Soft min approximation
    # To prevent overflow in exp, shift by min
    min_val = np.min(all_metrics)
    shifted = -beta * (all_metrics - min_val)
    # Clip to avoid underflow issues if beta is large and differences are large
    shifted = np.clip(shifted, -500, 0)
    
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_min = min_val - (1.0 / beta) * log_sum_exp
    
    return smooth_min

def _objective_and_grad(x_flat, beta, deadline):
    """
    Objective: Minimize -smooth_min_distance.
    Gradient: Computed analytically or via finite diff?
    Analytical gradient of soft-min is complex but doable.
    However, L-BFGS-B with numerical gradient might be too slow for 400 dims.
    Let's try to compute the gradient analytically.
    
    Let M be the vector of metrics.
    f(M) = min(M) approximated by S(M) = m - (1/beta) * log(sum(exp(-beta*(M_k - m))))
    where m = min(M).
    
    Actually, the standard soft-min is:
    S(M) = -1/beta * log(sum(exp(-beta * M_k)))
    Gradient dS/dM_k = (exp(-beta * M_k)) / sum(exp(-beta * M_j))
    
    Then we need dM_k/dx.
    
    This is getting complicated to implement robustly with the min() inside the wall metric.
    Instead, let's use a simpler approach:
    Use L-BFGS-B with a **finite difference** gradient but limit the number of function calls
    by using a good initial guess and stopping early.
    
    Wait, 400 dimensions * finite diff = 800 function calls per gradient step.
    Each function call is O(N^2) ~ 40k ops.
    800 * 40k = 32M ops per gradient.
    L-BFGS-B might take 100 steps. 3.2B ops. Too slow in Python loops.
    
    Alternative Strategy:
    Use **COBYLA** (Complexity-Optimized BY Linear Approximation)? No, also slow.
    
    Let's go back to the **Analytical Gradient** but simplify the metric.
    Instead of soft-min over ALL pairs, we can just maximize the minimum of:
    1. The minimum pairwise distance.
    2. The minimum wall distance.
    
    We can use a **weighted sum** of the soft-min of pairs and soft-min of walls?
    Or just maximize the soft-min of the combined vector.
    
    Let's implement the gradient for the combined soft-min.
    
    M_k are the metrics.
    S = -1/beta * log(sum(exp(-beta * M_k)))
    dS/dx = sum_k (dS/dM_k * dM_k/dx)
    
    dS/dM_k = exp(-beta * M_k) / sum(exp(-beta * M_j)) = w_k
    
    So we need dM_k/dx.
    
    Case 1: M_k is a pair metric d_ij / 2.
    d_ij = ||p_i - p_j||
    d(d_ij)/dx = (p_i - p_j) / d_ij for p_i, and (p_j - p_i) / d_ij for p_j.
    d(M_k)/dx = 0.5 * d(d_ij)/dx.
    
    Case 2: M_k is a wall metric 0.5 - max(|x_i|, |y_i|).
    This is non-smooth.
    Let's replace the wall metric with a smooth approximation too?
    Or just handle the gradient carefully.
    If |x_i| > |y_i|, M_k = 0.5 - |x_i|.
    dM_k/dx_i = -sign(x_i).
    
    This is doable. Let's implement it.
    """
    if time.time() > deadline:
        return 0.0, np.zeros_like(x_flat)
        
    p = x_flat.reshape(-1, 2)
    
    # Compute all metrics
    # Pairs
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, 1e-10)
    dists = np.sqrt(dists_sq)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    pair_metrics = pair_dists / 2.0
    
    # Walls
    abs_pts = np.abs(p)
    max_abs = np.max(abs_pts, axis=1)
    wall_metrics = 0.5 - max_abs
    
    all_metrics = np.concatenate([pair_metrics, wall_metrics])
    
    # Compute weights w_k for the soft-min
    # To avoid overflow, shift by min
    min_val = np.min(all_metrics)
    shifted = -beta * (all_metrics - min_val)
    shifted = np.clip(shifted, -500, 0)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Objective value
    obj_val = -(min_val - (1.0 / beta) * np.log(sum_exp))
    
    # Gradient
    grad = np.zeros_like(x_flat)
    
    # 1. Pair contributions
    # weights[0 : num_pairs] correspond to pair_metrics
    pair_weights = weights[:len(pair_metrics)]
    
    # We need to accumulate gradient contributions for each pair (i, j)
    # d(M_pair)/dx_i = 0.5 * (p_i - p_j) / d_ij
    # d(M_pair)/dx_j = 0.5 * (p_j - p_i) / d_ij
    
    # Vectorized accumulation is tricky. Let's loop over pairs if N is small enough?
    # N=200, pairs=19900. Looping 19900 times in Python is ~10ms? Maybe.
    # Let's try vectorization.
    
    # We have pair_weights (size P)
    # We need to add to grad[2*i:2*i+2] += w_k * 0.5 * (p_i - p_j) / d_ij
    # and grad[2*j:2*j+2] += w_k * 0.5 * (p_j - p_i) / d_ij
    
    # Let's create arrays for the contributions
    # d_ij is pair_dists
    # unit_vecs = (p[i] - p[j]) / d_ij
    # contrib_i = 0.5 * w_k * unit_vec
    # contrib_j = -0.5 * w_k * unit_vec
    
    # We can use np.add.at for vectorized accumulation
    
    i_flat = i_idx
    j_flat = j_idx
    
    # Compute unit vectors for all pairs
    # p_i - p_j for all pairs
    p_i = p[i_flat]
    p_j = p[j_flat]
    vecs = p_i - p_j
    # dists are pair_dists
    # Avoid division by zero
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    unit_vecs = vecs / safe_dists[:, np.newaxis]
    
    # Weighted contributions
    # w_k is pair_weights
    # Factor is 0.5 * w_k
    factors = 0.5 * pair_weights[:, np.newaxis]
    
    contrib_i = factors * unit_vecs
    contrib_j = -contrib_i
    
    # Accumulate into grad
    # grad is flat [x0, y0, x1, y1, ...]
    # p_i corresponds to indices 2*i_flat, 2*i_flat+1
    # We need to scatter add
    
    # Create indices for x and y
    idx_i_x = 2 * i_flat
    idx_i_y = 2 * i_flat + 1
    idx_j_x = 2 * j_flat
    idx_j_y = 2 * j_flat + 1
    
    np.add.at(grad, idx_i_x, contrib_i[:, 0])
    np.add.at(grad, idx_i_y, contrib_i[:, 1])
    np.add.at(grad, idx_j_x, contrib_j[:, 0])
    np.add.at(grad, idx_j_y, contrib_j[:, 1])
    
    # 2. Wall contributions
    # weights[len(pair_metrics) : ] correspond to wall_metrics
    wall_weights = weights[len(pair_metrics):]
    
    # M_wall_i = 0.5 - max(|x_i|, |y_i|)
    # If |x_i| >= |y_i|: dM/dx_i = -sign(x_i), dM/dy_i = 0
    # If |x_i| < |y_i|: dM/dx_i = 0, dM/dy_i = -sign(y_i)
    
    for i in range(N):
        w = wall_weights[i]
        if w < 1e-10: continue # Optimization: skip if weight is negligible
        
        xi = p[i, 0]
        yi = p[i, 1]
        
        if abs(xi) >= abs(yi):
            if xi > 0:
                grad[2*i] -= w
            elif xi < 0:
                grad[2*i] += w
        else:
            if yi > 0:
                grad[2*i+1] -= w
            elif yi < 0:
                grad[2*i+1] += w
                
    # We are minimizing -S, so we return -grad(S)
    # Wait. Objective is -S.
    # d(-S)/dx = -dS/dx.
    # My grad calculation above is dS/dx.
    # So return -grad.
    
    return obj_val, -grad

def construct():
    t_start = time.time()
    deadline = t_start + 130.0
    
    # 1. Initialization
    # Hexagonal grid is good for initial packing
    # Try to fit 200 points in hex grid
    # Approximate density: 200 points in area 1.
    # Hex grid spacing s ~ sqrt(1/200 * 2/sqrt(3)) ?
    # Let's just use a square grid with jitter, it's safer for initialization.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    # Add jitter
    rng = np.random.RandomState(42)
    pts += rng.randn(N, 2) * 0.005
    pts = np.clip(pts, -0.499, 0.499)
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # 2. Optimization with increasing beta (annealing)
    # Start with low beta (smooth), increase to high beta (sharp min)
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        x0 = best_pts.flatten()
        
        def obj_grad(x):
            return _objective_and_grad(x, beta, deadline)
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            res = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_pts = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                
        except Exception as e:
            pass # Keep previous best if optimization fails
            
    # 3. Final local refinement (optional, if time permits)
    if time.time() < deadline - 5:
        # Simple push-away for closest pairs
        current_pts = best_pts.copy()
        current_r = best_r
        
        for _ in range(20):
            if time.time() > deadline - 2:
                break
                
            d = pdist(current_pts, 'euclidean')
            min_idx = np.argmin(d)
            dists = squareform(d)
            np.fill_diagonal(dists, np.inf)
            i, j = np.unravel_index(np.argmin(dists), (N, N))
            
            vec = current_pts[i] - current_pts[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
                step = 0.0005
                new_i = current_pts[i] + vec * step
                new_j = current_pts[j] - vec * step
                new_i = np.clip(new_i, -0.499, 0.499)
                new_j = np.clip(new_j, -0.499, 0.499)
                
                temp_pts = current_pts.copy()
                temp_pts[i] = new_i
                temp_pts[j] = new_j
                temp_r = _compute_radius(temp_pts)
                
                if temp_r > current_r:
                    current_pts = temp_pts
                    current_r = temp_r
                    if current_r > best_r:
                        best_r = current_r
                        best_pts = current_pts.copy()
                        
    best_pts = np.clip(best_pts, -0.499, 0.499)
    return best_pts
# EVOLVE-BLOCK-END
