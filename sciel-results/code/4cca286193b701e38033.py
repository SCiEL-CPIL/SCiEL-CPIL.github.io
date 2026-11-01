"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _soft_min_radius(centers_flat, beta):
    """
    Compute the negative of the soft-minimum radius.
    centers_flat: 1D array of shape (2*N,)
    beta: sharpness parameter for the soft-min approximation.
    """
    centers = centers_flat.reshape(-1, 2)
    
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    # We only need the upper triangle to avoid double counting, but exp(-beta*inf) is 0 anyway.
    # Actually, cdist returns a symmetric matrix. We can just take all off-diagonals.
    # To save memory/time, we could take upper triangle, but N=200 is small enough.
    # d has shape (N, N).
    
    # Wall distances
    # x coords: centers[:, 0], y coords: centers[:, 1]
    # dist to left wall: 0.5 - x
    # dist to right wall: 0.5 + x
    # dist to bottom wall: 0.5 - y
    # dist to top wall: 0.5 + y
    wall_dists = np.array([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ]).flatten()
    
    # Combine all distances
    # We want to minimize -softmin, which is equivalent to maximizing softmin.
    # softmin(d) = -1/beta * log(sum(exp(-beta * d)))
    # We want to MINIMIZE -softmin(d) = 1/beta * log(sum(exp(-beta * d)))
    
    # To avoid overflow/underflow in exp, we can use the log-sum-exp trick.
    # But since we are minimizing, and d > 0, exp(-beta*d) < 1.
    # If beta is large, exp(-beta*d) becomes very small for large d.
    # The sum is dominated by the smallest d.
    
    # Let's collect all finite distances
    pair_dists = d[d < np.inf]
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Filter out non-finite just in case
    all_dists = all_dists[np.isfinite(all_dists)]
    
    if len(all_dists) == 0:
        return 0.0
        
    # Compute log-sum-exp of -beta * all_dists
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    x = -beta * all_dists
    max_x = np.max(x)
    # If max_x is very negative, exp(x - max_x) is safe.
    log_sum = max_x + np.log(np.sum(np.exp(x - max_x)))
    
    # softmin = -1/beta * log_sum
    # objective = -softmin = 1/beta * log_sum
    return log_sum / beta

def _obj_wrapper(centers_flat, beta):
    return _soft_min_radius(centers_flat, beta)

def _hex_init(n):
    """Create a hexagonal close-packed initial configuration."""
    centers = []
    # Try to fit hexagonal pattern
    # For n circles, use approximately sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    spacing = 0.95 / (rows + 1)  # slightly less than unit spacing for safety
    row = 0
    col = 0
    while len(centers) < n:
        x = (col + 0.5) * spacing - 0.5
        y = (row + 0.5) * spacing - 0.5
        if row % 2 == 1:
            x += spacing / 2
        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
            centers.append([x, y])
        col += 1
        if col > rows:
            col = 0
            row += 1
    return np.array(centers[:n])

def _square_init(n):
    """Create a square grid initial configuration."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n):
    """Create a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(n, 2))

def _compute_true_radius(centers):
    """Compute the actual radius for scoring/validation."""
    if len(centers) < 2:
        return 0.5
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    wall_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_wall_dist = wall_dists.min()
    return 0.5 * min(min_pair_dist, min_wall_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0
    
    # Initializations to try
    inits = [
        _hex_init(N),
        _square_init(N),
        _random_init(N),
    ]
    
    # Define bounds for L-BFGS-B
    # Each coordinate must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for init_centers in inits:
        if time.time() > deadline - 30:
            break
        
        current = init_centers.copy()
        # Ensure initial points are strictly inside to avoid boundary issues during optimization start
        current = np.clip(current, -0.4999, 0.4999)
        
        # Annealing schedule for beta
        # Start with small beta (smooth), end with large beta (sharp)
        betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            x0 = current.flatten()
            
            # Use L-BFGS-B with bounds
            # Note: jac is not provided, so finite difference will be used.
            # This is slow but robust. Given N=200, 400 variables.
            # Finite difference gradient requires 400 function evaluations per step.
            # This might be too slow.
            
            # Alternative: Use COBYLA? No, it's also slow.
            # Alternative: Provide gradient?
            # Gradient of soft-min is tractable.
            
            # Let's check time. 150s total.
            # If we use finite differences, it will likely time out.
            # We need an analytical gradient or a faster optimizer.
            
            # Let's implement the gradient for _soft_min_radius.
            # f(x) = (1/beta) * log(sum(exp(-beta * d_k(x))))
            # df/dx = (1/beta) * (1/sum(exp(...))) * sum(exp(-beta*d_k) * (-beta) * dd_k/dx)
            # df/dx = - sum(exp(-beta*d_k) * dd_k/dx) / sum(exp(-beta*d_k))
            # This is a weighted average of the gradients of the distances.
            
            def _obj_and_grad(centers_flat):
                centers = centers_flat.reshape(-1, 2)
                d = cdist(centers, centers)
                np.fill_diagonal(d, np.inf)
                
                wall_dists = np.array([
                    0.5 - centers[:, 0],
                    0.5 + centers[:, 0],
                    0.5 - centers[:, 1],
                    0.5 + centers[:, 1]
                ]).flatten()
                
                pair_dists = d[d < np.inf]
                all_dists = np.concatenate([pair_dists, wall_dists])
                all_dists = all_dists[np.isfinite(all_dists)]
                
                if len(all_dists) == 0:
                    return 0.0, np.zeros_like(centers_flat)
                
                x = -beta * all_dists
                max_x = np.max(x)
                exp_x = np.exp(x - max_x)
                sum_exp = np.sum(exp_x)
                log_sum = max_x + np.log(sum_exp)
                val = log_sum / beta
                
                # Gradient calculation
                # We need the gradient of each distance term d_k with respect to x.
                # This is complex to map back to the flat vector efficiently without storing indices.
                # Given the time constraint, let's try a simpler approach:
                # Use Nelder-Mead? No, scales poorly with 400 dims.
                # Use L-BFGS-B with finite differences but limit iterations?
                # Or use a smaller N for testing? No, N is fixed.
                
                # Let's try to compute the gradient properly.
                # It's easier to compute gradients for pair distances and wall distances separately.
                
                # Wall gradients are trivial.
                # Pair gradients: d_ij = ||c_i - c_j||.
                # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
                # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
                
                # We need weights w_k = exp(-beta * d_k) / sum(exp(-beta * d_k))
                # Note: The sum in the denominator is over ALL distances (pairs + walls).
                # So we need to normalize weights across all terms.
                
                # Let's compute weights for all terms.
                # This requires iterating or vectorizing the gradient accumulation.
                
                # For efficiency, let's just use the finite difference gradient from L-BFGS-B
                # but reduce the number of iterations and rely on the smoothness.
                # If it's too slow, we fall back to a simpler heuristic.
                
                # Actually, let's just return the value and let L-BFGS-B handle the gradient.
                # To speed up, we can use `maxiter` limits.
                
                return val, None # Let L-BFGS-B compute gradient numerically
            
            # To make it faster, we can provide the gradient.
            # Let's implement a fast gradient computation.
            
            def _compute_grad(centers_flat):
                centers = centers_flat.reshape(-1, 2)
                n = centers.shape[0]
                grad = np.zeros_like(centers_flat)
                
                # 1. Wall distances
                # d_w1 = 0.5 - x, grad = [-1, 0]
                # d_w2 = 0.5 + x, grad = [1, 0]
                # d_w3 = 0.5 - y, grad = [0, -1]
                # d_w4 = 0.5 + y, grad = [0, 1]
                
                # Distances
                dw1 = 0.5 - centers[:, 0]
                dw2 = 0.5 + centers[:, 0]
                dw3 = 0.5 - centers[:, 1]
                dw4 = 0.5 + centers[:, 1]
                
                all_dists = np.concatenate([dw1, dw2, dw3, dw4])
                # Pair distances
                d = cdist(centers, centers)
                np.fill_diagonal(d, np.inf)
                pair_dists_flat = d[d < np.inf]
                all_dists = np.concatenate([all_dists, pair_dists_flat])
                
                # Filter finite
                mask = np.isfinite(all_dists)
                all_dists = all_dists[mask]
                
                if len(all_dists) == 0:
                    return np.zeros_like(centers_flat)
                
                # Weights
                x = -beta * all_dists
                max_x = np.max(x)
                exp_x = np.exp(x - max_x)
                sum_exp = np.sum(exp_x)
                weights = exp_x / sum_exp # These are w_k / sum_exp, but we need w_k / sum_exp for the gradient formula?
                # Formula: grad = - sum( w_k * grad_d_k ) / sum_exp ?
                # f = (1/beta) log(sum exp(-beta d))
                # df/dx = (1/beta) * (1/S) * sum(exp(-beta d_k) * (-beta) * dd_k/dx)
                # df/dx = - (1/S) * sum(exp(-beta d_k) * dd_k/dx)
                # Let W_k = exp(-beta d_k) / S. Then df/dx = - sum(W_k * dd_k/dx)
                
                # We need to map weights back to specific distances.
                # This is tricky because we flattened them.
                
                # Let's do it in two parts: Walls and Pairs.
                
                # Walls
                w_all_dists = np.concatenate([dw1, dw2, dw3, dw4, pair_dists_flat])
                w_mask = np.isfinite(w_all_dists)
                w_dists = w_all_dists[w_mask]
                
                w_x = -beta * w_dists
                w_max = np.max(w_x)
                w_exp = np.exp(w_x - w_max)
                w_sum = np.sum(w_exp)
                w_weights = w_exp / w_sum
                
                # Now distribute weights back.
                # First 4*N are walls.
                # Rest are pairs.
                
                n_walls = 4 * n
                n_pairs = len(w_dists) - n_walls
                
                # Wall weights
                w_walls = w_weights[:n_walls]
                w_pairs = w_weights[n_walls:]
                
                # Wall gradients
                # dw1/dx = -1, dw2/dx = 1, dw3/dy = -1, dw4/dy = 1
                # grad_x += -w_walls[0:n] * (-1) + w_walls[n:2n] * (1)
                # grad_y += -w_walls[2n:3n] * (-1) + w_walls[3n:4n] * (1)
                
                g_x = np.zeros(n)
                g_y = np.zeros(n)
                
                g_x += w_walls[n:2*n] # from dw2 (grad +1)
                g_x -= w_walls[0:n]   # from dw1 (grad -1) -> -w * (-1) = +w? No.
                # Formula: - sum(W_k * dd_k/dx)
                # dw1 = 0.5-x, dd1/dx = -1. Term: -W1 * (-1) = W1.
                # dw2 = 0.5+x, dd2/dx = 1. Term: -W2 * (1) = -W2.
                # So g_x += W1 - W2.
                
                g_x = w_walls[0:n] - w_walls[n:2*n]
                
                # dw3 = 0.5-y, dd3/dy = -1. Term: W3.
                # dw4 = 0.5+y, dd4/dy = 1. Term: -W4.
                # g_y += W3 - W4.
                g_y = w_walls[2*n:3*n] - w_walls[3*n:4*n]
                
                # Pair gradients
                # We need to sum over all pairs (i,j) with i<j.
                # w_pairs corresponds to the flattened upper triangle of d.
                # We need to map these back to (i,j).
                
                # This is expensive to do in pure Python loops.
                # Given N=200, N^2/2 = 20000 pairs.
                # A loop of 20000 might be acceptable if it's simple numpy operations?
                # Or we can use sparse matrices?
                
                # Let's try a vectorized approach or a simple loop if it's fast enough.
                # For now, let's assume the wall gradient dominates or is sufficient for a rough guide.
                # Or, we can just use the numerical gradient from L-BFGS-B.
                
                # Let's stick to numerical gradient but optimize the function evaluation.
                # The function evaluation is O(N^2).
                # Numerical gradient is O(N^2 * 2N) = O(N^3).
                # 200^3 = 8,000,000 ops per gradient step.
                # If we do 100 steps, that's 800M ops. In Python/NumPy, this might take seconds per step.
                # 100 steps * 1s = 100s. Too slow.
                
                # We MUST use analytical gradient.
                
                # Let's implement the pair gradient accumulation.
                # Create an index map for the upper triangle.
                # This is complex. 
                
                # Alternative: Use `scipy.optimize.minimize` with `method='Nelder-Mead'`?
                # No, too slow in high dimensions.
                
                # Let's use `method='L-BFGS-B'` and provide the gradient.
                # I will implement the pair gradient loop. 20k iterations is fast in Python if vectorized per pair? No, scalar loop.
                # 20k iterations in Python takes ~0.02s.
                # So one gradient evaluation takes ~0.02s.
                # 100 iterations * 0.02s = 2s.
                # This is feasible!
                
                # Let's refine the gradient function to include pairs.
                
                # Get indices for upper triangle
                i_idx, j_idx = np.triu_indices(n, k=1)
                # The order in cdist upper triangle matches np.triu_indices order?
                # cdist returns a full matrix. d[d<inf] flattens row by row.
                # np.triu_indices gives row-major order of upper triangle.
                # Yes, they match.
                
                # d_ij values
                # We already have w_pairs.
                
                # Gradient of d_ij = ||c_i - c_j||
                # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
                # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
                
                # Term in sum: -W_k * dd_k/dx
                # For pair (i,j):
                # Contribution to c_i: -W_k * (c_i - c_j) / d_ij
                # Contribution to c_j: -W_k * (c_j - c_i) / d_ij = W_k * (c_i - c_j) / d_ij
                
                # So:
                # grad[c_i] -= W_k * (c_i - c_j) / d_ij
                # grad[c_j] += W_k * (c_i - c_j) / d_ij
                
                # We can vectorize this using np.add.at
                # But we need to be careful with the division by d_ij.
                
                # d_ij values are in w_dists[n_walls:]
                pair_dists = w_dists[n_walls:]
                
                # Avoid division by zero (should not happen if distinct)
                safe_pair_dists = np.where(pair_dists > 1e-9, pair_dists, 1.0)
                
                # Weights for pairs
                # w_pairs is already normalized by total sum_exp.
                
                # Differences
                # c_i - c_j
                diff = centers[i_idx] - centers[j_idx]
                
                # Scaling factor: W_k / d_ij
                # Note: w_pairs are the final weights W_k.
                scale = w_pairs / safe_pair_dists
                
                # Vectorized update
                # grad[c_i] -= scale * diff
                # grad[c_j] += scale * diff
                
                # We need to accumulate these into g_x and g_y.
                # Since i_idx and j_idx are 1D arrays, we can use np.add.at
                
                # g_x[i] -= scale * diff_x
                # g_x[j] += scale * diff_x
                
                diff_x = diff[:, 0]
                diff_y = diff[:, 1]
                
                np.add.at(g_x, i_idx, -scale * diff_x)
                np.add.at(g_y, i_idx, -scale * diff_y)
                
                np.add.at(g_x, j_idx, scale * diff_x)
                np.add.at(g_y, j_idx, scale * diff_y)
                
                grad_flat = np.zeros(2*n)
                grad_flat[0::2] = g_x
                grad_flat[1::2] = g_y
                
                return grad_flat
            
            def _obj_grad(centers_flat):
                val = _soft_min_radius(centers_flat, beta)
                grad = _compute_grad(centers_flat)
                return val, grad
            
            # Optimization
            result = minimize(
                _obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True, # Use provided gradient
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            current = result.x.reshape(-1, 2)
            current = np.clip(current, -0.5, 0.5)
            
            # Evaluate true radius
            r_true = _compute_true_radius(current)
            if r_true > best_radius:
                best_radius = r_true
                best_centers = current
    
    if best_centers is None:
        # Fallback
        best_centers = _square_init(N)
        best_radius = _compute_true_radius(best_centers)
        
    return best_centers
# EVOLVE-BLOCK-END
