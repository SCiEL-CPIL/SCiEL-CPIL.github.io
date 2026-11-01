"""
Strategy: Direct smooth optimization of a soft-minimum surrogate for packing radius.
Instead of binary searching for a feasible radius r and minimizing violations (which has flat zero-loss regions and requires multiple L-BFGS runs),
we directly maximize a smooth approximation of the minimum distance.
Objective: -log(sum(exp(-d_ij / T))) for pairs and walls.
This is differentiable everywhere and guides the optimizer to increase the minimum distance globally.
We use multiple diverse initializations to escape local optima.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(cents):
    """Compute the true packing radius."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _soft_min_obj_and_grad(x, T=0.01):
    """
    Objective: -log(sum(exp(-d/T))) for all constraints (pairs and walls).
    This approximates the minimum distance. We want to maximize this, so we minimize the negative.
    x: (2N,) array
    """
    c = x.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    wall_dists_flat = wall_dists.flatten()
    
    # Combine all distances into one vector
    # Only consider pairs i < j to avoid double counting, but for soft min, double counting just changes T effectively.
    # Let's stick to unique pairs.
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    all_dists = np.concatenate([pair_dists, wall_dists_flat])
    
    # Soft min: -T * log(sum(exp(-d/T)))
    # To avoid overflow/underflow, subtract max d
    d_max = np.max(all_dists)
    exp_terms = np.exp(-(all_dists - d_max) / T)
    s = np.sum(exp_terms)
    obj = -T * np.log(s) + d_max # Wait, -T log(sum exp(-d/T)) = -T log(sum exp(-(d-dmax)/T)) + dmax?
    # Let's re-derive:
    # m = min(d_i)
    # soft_min = -T log( sum exp(-d_i/T) )
    # = -T log( sum exp(-(d_i - d_max)/T) * exp(-d_max/T) )
    # = -T log( sum exp(-(d_i - d_max)/T) ) - T log(exp(-d_max/T))
    # = -T log( sum exp(-(d_i - d_max)/T) ) + d_max
    
    # Gradient with respect to x
    # d obj / d d_i = -T * (1/s) * (-1/T) * exp(-d_i/T) = exp(-d_i/T) / s
    # Let p_i = exp(-(d_i - d_max)/T) / s. Sum p_i = 1.
    # d obj / d d_i = p_i
    
    p = exp_terms / s
    
    # Gradient of distance d_ij w.r.t x_i and x_j
    # d d_ij / d x_i = (x_i - x_j) / d_ij
    # d d_ij / d x_j = (x_j - x_i) / d_ij
    
    grad = np.zeros_like(x)
    
    # Pairwise contributions
    # For each pair (i, j), contribution to grad[x_i] is p_k * (x_i - x_j) / d_k
    # contribution to grad[x_j] is p_k * (x_j - x_i) / d_k
    # We need to accumulate these.
    
    # Efficiently:
    # diff is (N, N, 2)
    # We can use broadcasting, but it's memory heavy for N=50? 50*50*2 = 5000 floats. Fine.
    
    # p is shape (num_pairs,)
    # We need to map p back to the (N, N) matrix.
    p_matrix = np.zeros((N, N))
    p_matrix[iu] = p[:len(iu[0])]
    p_matrix = p_matrix + p_matrix.T
    
    # dists is (N, N)
    # Avoid division by zero (diagonal is inf, but let's be safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    
    # grad_x_i = sum_j p_ij * unit_vec_ij
    # grad_x_j = sum_i p_ji * unit_vec_ji = - sum_i p_ij * unit_vec_ij (since p is symmetric, unit is antisymmetric)
    # So grad_x_i = sum_j p_ij * unit_vec_ij
    
    grad_pairs = np.sum(p_matrix[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall contributions
    # d wall_i / d x_i = -sign(x_i) if |x_i| < 0.5 - r? No, wall dist is min(0.5-|x|, 0.5-|y|).
    # This is non-smooth at x=0 or y=0.
    # Approximation: use subgradient or just gradient of 0.5 - |x| and 0.5 - |y| and take min?
    # Actually, wall dist w_i = min(0.5 - |x_i|, 0.5 - |y_i|).
    # Let dx_i = 0.5 - |x_i|, dy_i = 0.5 - |y_i|.
    # w_i = min(dx_i, dy_i).
    # Soft min of dx_i and dy_i?
    # For simplicity, let's approximate the wall gradient by the gradient of the smaller coordinate.
    # Or better, include both in the soft min?
    # The current formulation treats wall dist as a single distance value.
    # Let's refine:
    # Wall distances for x: wx = 0.5 - |x|
    # Wall distances for y: wy = 0.5 - |y|
    # We want min over all pairs and all walls.
    # Let's just add wx and wy to the list of distances.
    
    # Redo with separate wx and wy
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    
    # Combine pair dists, wx, wy
    # We already computed pair dists.
    # Let's rebuild all_dists and gradients properly.
    
    # This is getting complex. Let's simplify the wall part.
    # If a point is close to a wall, it's usually close to one specific wall (left/right or top/bottom).
    # We can approximate the gradient by assuming the min is achieved by the closest wall.
    
    # Let's stick to the previous simple approach for walls but be careful with gradients.
    # For wall dist w_i = min(0.5-|x_i|, 0.5-|y_i|).
    # If 0.5-|x_i| < 0.5-|y_i|, then w_i = 0.5-|x_i|. Grad w.r.t x_i is -sign(x_i). Grad w.r.t y_i is 0.
    # Else w_i = 0.5-|y_i|. Grad w.r.t x_i is 0. Grad w.r.t y_i is -sign(y_i).
    # At ties, use average or just one.
    
    # Let's recompute obj and grad from scratch with this logic.
    
    # 1. Pairwise
    # dists (N, N), p_matrix (N, N)
    # grad_pairs computed above.
    
    # 2. Walls
    # We need to include wx and wy in the soft min.
    # Total distances: P pairs, 2N walls.
    
    # Let's restart the gradient calculation properly.
    
    # Distances:
    d_pairs = dists[iu]
    d_walls_x = wx
    d_walls_y = wy
    
    all_dists = np.concatenate([d_pairs, d_walls_x, d_walls_y])
    
    d_max = np.max(all_dists)
    exp_terms = np.exp(-(all_dists - d_max) / T)
    s = np.sum(exp_terms)
    obj = -T * np.log(s) + d_max
    
    p = exp_terms / s
    
    # Gradients
    grad = np.zeros_like(x)
    
    # Pairs
    p_pairs = p[:len(iu[0])]
    # Accumulate pair gradients
    # For each pair (i, j), weight p_k
    # grad[i] += p_k * (x_i - x_j) / d_k
    # grad[j] += p_k * (x_j - x_i) / d_k
    
    # We can do this with a loop for clarity, N=50 is small.
    # Or vectorize.
    # Let's vectorize.
    # unit_vecs is (N, N, 2).
    # p_matrix is (N, N).
    # grad_pairs = sum_j p_ij * unit_vec_ij
    # This was computed as grad_pairs above.
    # But p_matrix was based on the OLD p which only had pairs.
    # Now p has pairs, wx, wy.
    # So p_matrix should only contain the pair part of p.
    
    p_pairs_matrix = np.zeros((N, N))
    p_pairs_matrix[iu] = p_pairs
    p_pairs_matrix = p_pairs_matrix + p_pairs_matrix.T
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
        
    grad_pairs = np.sum(p_pairs_matrix[:, :, np.newaxis] * unit_vecs, axis=1)
    grad += grad_pairs
    
    # Walls
    p_wx = p[len(iu[0]):len(iu[0])+N]
    p_wy = p[len(iu[0])+N:]
    
    # For each i:
    # If it's the "active" wall (min), the gradient is -sign(x) or -sign(y).
    # But in soft min, all walls contribute with weight p.
    # So grad_x_i += p_wx_i * (-sign(x_i))
    # grad_y_i += p_wy_i * (-sign(y_i))
    # Wait, d(0.5 - |x|)/dx = -sign(x).
    # So we add p_wx_i * (-sign(x_i)) to grad[2*i].
    # And p_wy_i * (-sign(y_i)) to grad[2*i+1].
    
    signs_x = np.sign(c[:, 0])
    signs_y = np.sign(c[:, 1])
    # Handle 0
    signs_x[signs_x == 0] = 1.0
    signs_y[signs_y == 0] = 1.0
    
    grad[0::2] += -p_wx * signs_x
    grad[1::2] += -p_wy * signs_y
    
    return obj, grad

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _compute_radius(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _compute_radius(best_cents)

    return best_cents, best_r

def _construct_grid():
    """Create a square grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    cents = grid[:N]
    r = _compute_radius(cents)
    return cents, r

def _construct_jittered(seed=0):
    """Create a jittered grid arrangement."""
    rng = np.random.default_rng(seed)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    cents = grid[:N].copy()
    # Add small random jitter
    cents += rng.normal(0, 0.01, cents.shape)
    # Clip to bounds
    cents = np.clip(cents, -0.45, 0.45)
    r = _compute_radius(cents)
    return cents, r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initialize with diverse starts
    inits = []
    hex_cents, hex_r = _construct_hexagonal()
    inits.append((hex_cents, hex_r))
    
    grid_cents, grid_r = _construct_grid()
    inits.append((grid_cents, grid_r))
    
    for seed in [1, 2, 3, 4]:
        jit_cents, jit_r = _construct_jittered(seed)
        inits.append((jit_cents, jit_r))
    
    best_cents = inits[0][0]
    best_r = inits[0][1]
    
    # Optimize each initialization
    for i, (cents, r_init) in enumerate(inits):
        if time.time() - start_time > deadline - 10:
            break
            
        x0 = cents.reshape(-1)
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        # Anneal T? Start with larger T for global search, smaller for local.
        # Let's try a few T values.
        T_values = [0.05, 0.02, 0.01, 0.005]
        
        current_cents = cents
        for T in T_values:
            if time.time() - start_time > deadline - 5:
                break
                
            result = minimize(
                _soft_min_obj_and_grad, x0, args=(T,),
                jac=True,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_cents = result.x.reshape(N, 2)
            new_r = _compute_radius(new_cents)
            
            if new_r > best_r:
                best_r = new_r
                best_cents = new_cents
                # Use this as the start for the next T (or next init if we were chaining)
                x0 = result.x
            else:
                # If it didn't improve, maybe stick with the previous best x0 for this T?
                # Or just move to next T with original x0?
                # Let's use the result as x0 anyway, it might be closer to the optimum for smaller T.
                x0 = result.x

    # Final polish with very small T on the best found
    if time.time() - start_time < deadline - 5:
        x0 = best_cents.reshape(-1)
        bounds = [(-0.5, 0.5)] * (2 * N)
        result = minimize(
            _soft_min_obj_and_grad, x0, args=(0.001,),
            jac=True,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        new_cents = result.x.reshape(N, 2)
        new_r = _compute_radius(new_cents)
        if new_r > best_r:
            best_r = new_r
            best_cents = new_cents

    best_cents = np.clip(best_cents, -0.5, 0.5)
    return best_cents

# EVOLVE-BLOCK-END
