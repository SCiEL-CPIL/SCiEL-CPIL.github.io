import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _score_from_centers(centers):
    """Compute the common radius for a set of centers inside [-0.5, 0.5]^2."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to boundaries
    # Centers must be in [-0.5, 0.5]
    dist_bound = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1), axis=1)
    min_bound = np.min(dist_bound)
    
    if min_bound < 0:
        return 0.0

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    if min_pair < 0:
        return 0.0

    return min(min_bound, min_pair)

def _smooth_neg_score(centers_flat, beta=50.0):
    """
    Smooth surrogate for the negative radius.
    Uses log-sum-exp to approximate the minimum.
    r = min( min_pair_dist/2, min_bound_dist )
    We want to maximize r, so minimize -r.
    -r = -min(A, B) = min(-A, -B)
    soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    """
    centers = centers_flat.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return 10.0
    
    # Distance to boundaries
    dist_bound = np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ], axis=1)
    
    # We need the min over all boundary distances for all points
    # min_bound = min( min_i (0.5 - |x_i|), min_i (0.5 - |y_i|) )
    # Let's compute the soft min of all boundary distances
    # There are 2*N boundary distances
    # But it's easier to compute min_bound_exact first to check validity roughly?
    # No, let's stick to the formula.
    
    # Soft min of boundary distances
    # vals_bound = [0.5 - |x_1|, ..., 0.5 - |x_N|, 0.5 - |y_1|, ..., 0.5 - |y_N|]
    # This is 100 values.
    # To avoid large arrays, we can compute soft min of x-bounds and y-bounds separately?
    # min(min_x, min_y) is not soft_min(min_x, min_y) easily unless we combine.
    # Let's just compute the exact min_bound for a quick check, and if it's very negative, return high penalty.
    min_bound_exact = np.min(dist_bound)
    if min_bound_exact < -0.01:
        return 10.0

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft min of squared distances
    # min_d = min(sqrt(d_sq)) = sqrt(min(d_sq))
    # soft_min_d_sq = -1/beta * log(sum(exp(-beta * d_sq)))
    # But sum has 2450 terms. exp(-beta * d_sq) might underflow if d_sq is large.
    # Use log-sum-exp trick.
    
    # Since d_sq are positive, -beta * d_sq are negative.
    # max_term = max(-beta * d_sq) = -beta * min(d_sq)
    # log_sum_exp = max_term + log(sum(exp(-beta * d_sq - max_term)))
    
    # To save time, we can just use the exact min for the pair distance in the objective
    # and only smooth the boundary? Or smooth both?
    # The bottleneck is usually the pair distances.
    # Let's try a simpler approach: use exact min for pairs, and smooth boundary?
    # Actually, Nelder-Mead handles non-smooth well. The main issue is getting stuck in local minima.
    # So a smooth objective is mainly for the final polish.
    
    # For the polish, let's use a smooth approximation of the radius.
    # r = min(r_pair, r_bound)
    # -r = min(-r_pair, -r_bound)
    
    # Let's compute soft min of r_pair and r_bound.
    # r_pair = 0.5 * min(d_ij)
    # r_bound = min(d_ik)
    
    # Soft min of d_ij:
    # Let's approximate min(d_ij) by soft_min(d_ij).
    # d_ij = sqrt(d_ij_sq)
    # soft_min_d = sqrt(soft_min_d_sq) is an approximation.
    
    # Let's just use the exact radius for Nelder-Mead, and for the final polish,
    # we can use a high-beta smooth version.
    
    # For this function, let's return the exact negative radius if beta is large,
    # or a smooth version.
    
    # Let's implement the smooth version properly.
    # Soft min of pairwise distances
    # We have 2450 distances.
    # min_d = min(dists)
    # soft_min_d = -1/beta * log(sum(exp(-beta * dists)))
    
    # To avoid overflow/underflow:
    # log_sum_exp = max_d * (-beta) + log(sum(exp(-beta * (dists - max_d))))
    # Wait, max of -beta * dists is -beta * min_dists.
    # Let m = min(dists).
    # -beta * m is the max of the exponents.
    # log_sum_exp = -beta * m + log(sum(exp(-beta * (dists - m))))
    # soft_min_d = -log_sum_exp / beta = m - (1/beta) * log(sum(exp(-beta * (dists - m))))
    # This is stable.
    
    dists = np.sqrt(dists_sq)
    min_d = np.min(dists)
    # If min_d is 0, we have overlap.
    if min_d < 1e-12:
        return 10.0
        
    # Calculate soft min of dists
    # dists - m >= 0
    # exp(-beta * (dists - m)) <= 1
    # Sum is <= 2450
    # log(sum) is safe.
    
    # Optimization: we don't need to compute exp for all 2450 if beta is large,
    # but 2450 is small.
    
    # However, computing dists for every evaluation is expensive.
    # The exact score function already does this.
    # So _smooth_neg_score is only for the final polish where we might use gradient methods?
    # No, we are using Nelder-Mead which doesn't use gradients.
    # So we can just use the exact score function for Nelder-Mead.
    # The smooth function is useful if we switch to L-BFGS-B or similar.
    
    # Let's just use the exact score for Nelder-Mead.
    # For the final polish, we can use a "tighter" Nelder-Mead on the exact score.
    
    # So, I will remove the complex smooth implementation and just use the exact score
    # with careful parameter tuning for Nelder-Mead.
    
    # Let's rewrite this function to just be the exact negative score,
    # but I'll keep the name for compatibility if needed, or just use _neg_score.
    
    return _neg_score(centers_flat)

def _neg_score(centers_flat):
    """Objective function: negative of the radius."""
    centers = centers_flat.reshape(-1, 2)
    score = _score_from_centers(centers)
    if score <= 0:
        return 10.0
    return -score

def _generate_hex_initial(n_rows, row_lengths, sx, sy, x_shift, x_global, y_global):
    """
    Generate centers from hex parameters.
    row_lengths: list of ints, sum to N.
    """
    centers = []
    for i, length in enumerate(row_lengths):
        if length == 0:
            continue
        # y position
        y = y_global + (i - (len(row_lengths) - 1) / 2.0) * sy
        
        # x start position
        # Center the row of 'length' circles around x_global
        # Shift odd rows by x_shift
        shift = x_shift if i % 2 != 0 else 0.0
        x_start = x_global + shift - sx * (length - 1) / 2.0
        
        for j in range(length):
            x = x_start + j * sx
            centers.append([x, y])
            
    centers = np.array(centers)
    if len(centers) != N:
        return None
    return centers

def _optimize_from_start(start_centers, deadline):
    """
    Optimize a set of centers.
    Returns (best_score, best_centers).
    """
    best_score = _score_from_centers(start_centers)
    best_centers = start_centers.copy()
    x0 = start_centers.flatten()
    
    # Stage 1: Nelder-Mead with moderate tolerance
    try:
        result = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < 10.0:
            opt_centers = result.x.reshape(-1, 2)
            opt_score = _score_from_centers(opt_centers)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = opt_centers.copy()
                x0 = opt_centers.flatten()
    except Exception:
        pass
    
    # Stage 2: Powell for potentially better convergence
    if time.time() < deadline - 5:
        try:
            result = minimize(_neg_score, x0, method='Powell',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < 10.0:
                opt_centers = result.x.reshape(-1, 2)
                opt_score = _score_from_centers(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers.copy()
                    x0 = opt_centers.flatten()
        except Exception:
            pass
            
    # Stage 3: Final Nelder-Mead polish with tight tolerance
    if time.time() < deadline - 2:
        try:
            result = minimize(_neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < 10.0:
                opt_centers = result.x.reshape(-1, 2)
                opt_score = _score_from_centers(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers.copy()
        except Exception:
            pass
            
    return best_score, best_centers

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = -np.inf
    best_centers = None
    
    # Define candidate row patterns for hexagonal packing
    # Sum must be 50.
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4], # 8 rows
        [6, 7, 6, 7, 6, 7, 6, 5], # 8 rows
        [8, 7, 7, 7, 7, 7, 7],    # 7 rows
        [7, 8, 7, 8, 7, 8, 5],    # 7 rows
        [6, 7, 7, 7, 7, 7, 8],    # 7 rows
        [5, 7, 7, 7, 7, 7, 9],    # 7 rows
        [6, 8, 6, 8, 6, 8, 8],    # 7 rows
        [8, 6, 8, 6, 8, 6, 8],    # 7 rows
        [6, 6, 7, 7, 7, 7, 7, 3], # 8 rows
        [5, 6, 7, 7, 7, 7, 7, 4], # 8 rows
        [4, 7, 7, 7, 7, 7, 7, 4], # 8 rows
        [8, 8, 7, 7, 7, 7, 6],    # 7 rows
        [6, 7, 7, 7, 7, 7, 7, 2], # 8 rows
        [7, 7, 7, 7, 7, 7, 6, 2], # 8 rows
        [7, 6, 7, 6, 7, 6, 6, 5], # 8 rows
        [6, 7, 6, 7, 6, 7, 7, 4], # 8 rows
        [5, 5, 5, 5, 5, 5, 5, 10], # 8 rows (grid-like)
        [10, 5, 5, 5, 5, 5, 5, 10], # 8 rows
        [4, 6, 6, 6, 6, 6, 6, 6], # 8 rows
        [6, 6, 6, 6, 6, 6, 6, 4], # 8 rows
    ]
    
    valid_patterns = [p for p in patterns if sum(p) == N]
    
    # Generate initial guesses
    # 1. Hexagonal lattices
    for pattern in valid_patterns:
        if time.time() > deadline - 10:
            break
            
        # Estimate spacing
        # If 7 rows, sy ~ 1/6 = 0.166, sx ~ 0.166
        # If 8 rows, sy ~ 1/7 = 0.142, sx ~ 0.142
        n_rows = len(pattern)
        sy_init = 1.0 / (n_rows + 0.5)  # Approx
        sx_init = sy_init * 2.0 / np.sqrt(3.0) # Hex spacing
        
        # Try a few offsets
        for x_shift in [0.0, sx_init/2]:
            for x_global in [-0.05, 0.0, 0.05]:
                for y_global in [-0.05, 0.0, 0.05]:
                    if time.time() > deadline - 5:
                        continue
                        
                    centers = _generate_hex_initial(n_rows, pattern, sx_init, sy_init, x_shift, x_global, y_global)
                    if centers is None:
                        continue
                    
                    score, opt_centers = _optimize_from_start(centers, deadline)
                    if score > best_score:
                        best_score = score
                        best_centers = opt_centers
    
    # 2. Grid-like initializations
    # 5x10 grid
    if time.time() < deadline - 10:
        for nx, ny in [(5, 10), (10, 5), (7, 8), (8, 7)]:
            if nx * ny != N:
                continue
                
            # Create grid
            xs = np.linspace(-0.5 + 1.0/(2*nx), 0.5 - 1.0/(2*nx), nx)
            ys = np.linspace(-0.5 + 1.0/(2*ny), 0.5 - 1.0/(2*ny), ny)
            
            grid_centers = []
            for y in ys:
                for x in xs:
                    grid_centers.append([x, y])
            
            grid_centers = np.array(grid_centers)
            
            # Add some jitter
            np.random.seed(42)
            jittered = grid_centers + np.random.normal(0, 0.01, grid_centers.shape)
            
            score, opt_centers = _optimize_from_start(jittered, deadline)
            if score > best_score:
                best_score = score
                best_centers = opt_centers
                
            # Also try without jitter
            score, opt_centers = _optimize_from_start(grid_centers, deadline)
            if score > best_score:
                best_score = score
                best_centers = opt_centers

    # Fallback
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_score = _score_from_centers(best_centers)
    
    return best_centers
