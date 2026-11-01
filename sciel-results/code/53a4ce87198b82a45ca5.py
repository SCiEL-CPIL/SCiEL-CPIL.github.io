"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    
    # Wall distances
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall = np.min(wall_dists, axis=1)
    min_wall_global = np.min(min_wall)
    
    # Pairwise distances
    # Use vectorized calculation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    return min(min_wall_global, min_dist / 2.0)

def _soft_min_objective(x, beta):
    """
    Objective function to minimize.
    We want to maximize the minimum of:
    1. Wall distances w_i
    2. Half pairwise distances d_ij / 2
    
    Equivalent to minimizing:
    - min( {w_i} U {d_ij/2} )
    
    Using LogSumExp approximation for the minimum:
    min(S) ≈ -1/beta * log(sum(exp(-beta * s)))
    
    So we minimize:
    1/beta * log(sum(exp(-beta * s)))
    
    This is smooth and differentiable.
    """
    c = x.reshape(N, 3)
    
    # Wall terms: w_i = min(0.5 - |c_ik|)
    # Actually, we need min over all walls and points.
    # Let's construct the vector of all candidate distances.
    
    # Wall distances for each point
    abs_c = np.abs(c)
    # Dist to 6 walls for each point
    # x: 0.5 - c[:, 0], 0.5 + c[:, 0]
    # y: 0.5 - c[:, 1], 0.5 + c[:, 1]
    # z: 0.5 - c[:, 2], 0.5 + c[:, 2]
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # We want the minimum of all wall distances and all half-pairwise distances.
    # Let S be the set of values.
    # Wall values: all entries in walls (N*6)
    # Pair values: dist / 2.0 (N*N, but we only need upper triangle or all non-diag)
    
    # To avoid memory issues with large beta and many terms, 
    # we can compute the log-sum-exp in a stable way.
    
    # Values:
    v_walls = walls.flatten()
    v_pairs = dist.flatten() / 2.0
    v_pairs = v_pairs[v_pairs < np.inf] # Remove diagonals
    
    # Combine
    # Note: exp(-beta * v) can underflow/overflow.
    # log(sum(exp(-beta * v))) = -beta * max(v) + log(sum(exp(-beta * (v - max(v)))))
    # Wait, min(v) corresponds to max(-v).
    # Let m = min(v).
    # log(sum(exp(-beta * v))) = log(exp(-beta*m) * sum(exp(-beta*(v-m))))
    # = -beta*m + log(sum(exp(-beta*(v-m))))
    
    # We want to minimize (1/beta) * log(sum(exp(-beta * v)))
    # = -m + (1/beta) * log(sum(exp(-beta*(v-m))))
    
    # Let's compute m = min(v_walls, v_pairs)
    # For efficiency, we can just use the raw formula if beta isn't too huge, 
    # but for high precision beta can be large.
    
    # Let's compute the objective and gradient simultaneously to save time?
    # No, separate functions for clarity, but we can optimize later.
    
    # Objective only
    all_vals = np.concatenate([v_walls, v_pairs])
    
    # Stable log-sum-exp for min
    m = np.min(all_vals)
    # If m is very small/negative (shouldn't happen if inside cube), handle carefully
    # v >= 0 always for valid points.
    
    # exp(-beta * (v - m))
    # v - m >= 0
    # So -beta * (v - m) <= 0
    # exp is safe.
    
    terms = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(terms))
    
    obj = -m + log_sum / beta
    
    return obj

def _soft_min_grad(x, beta):
    """
    Gradient of the soft-min objective.
    f(x) = -m + (1/beta) * log(sum(exp(-beta * v_i)))
    where m = min(v_i)
    
    Note: m depends on x.
    Let's derive carefully.
    f(x) = (1/beta) * log(sum_j exp(-beta * v_j(x)))
    
    df/dx = (1/beta) * (1/sum_j exp(-beta * v_j)) * sum_k d/dx (exp(-beta * v_k))
    = (1/beta) * (1/sum_j exp(-beta * v_j)) * sum_k (-beta * exp(-beta * v_k)) * dv_k/dx
    = - (1/sum_j exp(-beta * v_j)) * sum_k exp(-beta * v_k) * dv_k/dx
    = - sum_k p_k * dv_k/dx
    where p_k = exp(-beta * v_k) / sum_j exp(-beta * v_j)
    
    This is a convex combination of the gradients of the individual constraints.
    The weights p_k are proportional to exp(-beta * v_k).
    As beta -> infinity, p_k concentrates on the minimum v_k.
    """
    c = x.reshape(N, 3)
    
    # Compute values again
    abs_c = np.abs(c)
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    v_walls = walls.flatten()
    v_pairs = dist.flatten() / 2.0
    v_pairs = v_pairs[v_pairs < np.inf]
    
    all_vals = np.concatenate([v_walls, v_pairs])
    
    # Compute weights p_k
    m = np.min(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    sum_exp = np.sum(exp_terms)
    p = exp_terms / sum_exp
    
    # Now compute gradient: - sum_k p_k * dv_k/dx
    grad = np.zeros_like(c)
    
    # 1. Wall gradients
    # p_walls are the first N*6 elements of p
    p_walls = p[:N*6].reshape(N, 6)
    
    # dv/dx for walls:
    # 0: 0.5 - x -> d/dx = -1
    # 1: 0.5 + x -> d/dx = +1
    # 2: 0.5 - y -> d/dy = -1
    # 3: 0.5 + y -> d/dy = +1
    # 4: 0.5 - z -> d/dz = -1
    # 5: 0.5 + z -> d/dz = +1
    
    # grad_x -= sum(p_walls[:, 0] * (-1) + p_walls[:, 1] * (1))
    # grad_x += p_walls[:, 0] - p_walls[:, 1]
    
    grad[:, 0] += p_walls[:, 0] - p_walls[:, 1]
    grad[:, 1] += p_walls[:, 2] - p_walls[:, 3]
    grad[:, 2] += p_walls[:, 4] - p_walls[:, 5]
    
    # 2. Pair gradients
    # p_pairs are the remaining elements
    p_pairs = p[N*6:]
    
    # We need to map p_pairs back to the (i, j) indices
    # dist matrix was (N, N). We flattened it.
    # The order is row-major: (0,0), (0,1), ..., (0,N-1), (1,0), ...
    # We removed the diagonal.
    # Let's reconstruct the matrix of weights.
    
    # Create a mask for valid pairs
    # Actually, easier to just iterate or vectorize the accumulation.
    
    # Let's create a matrix P_pairs of shape (N, N)
    # Initialize with 0
    P_mat = np.zeros((N, N))
    
    # We need to fill P_mat with p_pairs in the correct order.
    # The flattening order of dist was:
    # d[0,0], d[0,1], ..., d[0,N-1]
    # d[1,0], ...
    # We set diagonal to inf, then filtered out inf.
    
    # Let's just loop over i, j to fill P_mat, it's N^2 = 10000 operations, very fast.
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            P_mat[i, j] = p_pairs[idx]
            idx += 1
            
    # Gradient contribution from pairs:
    # v_ij = dist_ij / 2
    # dv_ij / d c_i = (c_i - c_j) / (2 * dist_ij)
    # dv_ij / d c_j = (c_j - c_i) / (2 * dist_ij)
    
    # grad_i -= sum_j P_mat[i, j] * dv_ij / d c_i
    # grad_i -= sum_j P_mat[i, j] * (c_i - c_j) / (2 * dist_ij)
    
    # dist_safe to avoid division by zero
    dist_safe = np.where(dist > 1e-8, dist, 1e-8)
    
    # coeff[i, j] = P_mat[i, j] / (2 * dist_safe[i, j])
    coeff = P_mat / (2.0 * dist_safe)
    
    # grad_i = - sum_j coeff[i, j] * (c_i - c_j)
    # (c_i - c_j) is diff[i, j]
    
    # weighted_diff = coeff[:, :, np.newaxis] * diff
    # grad -= sum_j weighted_diff[i, j, k]
    
    weighted_diff = coeff[:, :, np.newaxis] * diff
    grad -= np.sum(weighted_diff, axis=1)
    
    return grad.flatten()

def _soft_min_jac(x, beta):
    """Combined objective and gradient for L-BFGS-B."""
    obj = _soft_min_objective(x, beta)
    grad = _soft_min_grad(x, beta)
    return obj, grad

def _build_initial_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _build_bcc_init():
    """Body-Centered Cubic initialization."""
    # BCC packing density is higher than simple cubic.
    # Unit cell has 2 atoms.
    # For N=100, we need 50 unit cells.
    # 50 is not a perfect cube. 3x3x6 = 54.
    # Let's use 3x3x6 grid of BCC cells.
    nx, ny, nz = 3, 3, 6
    total_cells = nx * ny * nz
    # Each cell contributes 2 points.
    # Total points = 108. We take first 100.
    
    points = []
    # Scale factor to fit in [-0.5, 0.5]
    # Cell size a.
    # Total size nx*a <= 1.
    # Let's start with a = 1.0 / max(nx, ny, nz)
    a = 1.0 / max(nx, ny, nz)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner atom
                # Center is at (i+0.5)*a, ...
                # Let's center the whole structure.
                x0 = (i + 0.5) * a - 0.5
                y0 = (j + 0.5) * a - 0.5
                z0 = (k + 0.5) * a - 0.5
                
                # Corner at (0,0,0) relative to cell
                points.append([x0, y0, z0])
                # Center at (0.5, 0.5, 0.5) relative to cell
                points.append([x0 + 0.5*a, y0 + 0.5*a, z0 + 0.5*a])
                
    points = np.array(points)
    return points[:N].copy()

def _optimize_from_initial(c0, time_limit):
    """
    Optimize starting from c0.
    Uses Simulated Annealing followed by L-BFGS-B polish.
    """
    c = c0.copy()
    start_time = time.time()
    
    # 1. Simulated Annealing
    # Objective to minimize: _soft_min_objective
    # We want to minimize this.
    
    # Parameters
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.95
    beta_sa = 50.0 # Initial beta for SA
    
    max_sa_iters = 5000
    
    c_best = c.copy()
    f_best = _soft_min_objective(c.flatten(), beta_sa)
    
    T = T_start
    
    for i in range(max_sa_iters):
        if time.time() > time_limit - 20.0: # Reserve 20s for polish
            break
            
        # Generate candidate
        # Step size proportional to T?
        # Random move
        step_size = 0.01 * (T / T_start) + 0.001
        c_new = c + np.random.normal(0, step_size, size=(N, 3))
        
        # Clip to cube (soft boundary, or hard?)
        # Hard clip is safer to avoid huge penalties, but might stick to walls.
        # Let's clip to [-0.49, 0.49] to keep away from boundaries for SA
        c_new = np.clip(c_new, -0.49, 0.49)
        
        # Evaluate
        f_new = _soft_min_objective(c_new.flatten(), beta_sa)
        f_old = _soft_min_objective(c.flatten(), beta_sa)
        
        delta_f = f_new - f_old
        
        if delta_f < 0 or np.random.random() < np.exp(-delta_f / T):
            c = c_new
            if f_new < f_best:
                f_best = f_new
                c_best = c.copy()
                
        T *= alpha
        
        # Adapt beta?
        # Keep beta constant during SA to maintain smoothness.
        # Or increase beta slowly to sharpen the minimum.
        if i % 100 == 0:
            beta_sa *= 1.05
            
    # 2. L-BFGS-B Polish
    # Use a higher beta for final precision
    beta_final = 500.0
    
    # Run L-BFGS-B
    try:
        res = minimize(_soft_min_jac, c_best.flatten(), args=(beta_final,), 
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        
        c_polished = res.x.reshape(N, 3)
        c_polished = np.clip(c_polished, -0.5, 0.5)
        
        # Check if polished is better
        f_polished = _soft_min_objective(c_polished.flatten(), beta_final)
        f_best_current = _compute_radius(c_best)
        f_polished_radius = _compute_radius(c_polished)
        
        if f_polished_radius > f_best_current:
            c_best = c_polished
            
    except Exception as e:
        pass
        
    return c_best

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Try multiple initializations
    inits = [
        _build_initial_grid(),
        _build_bcc_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3)) # Random start
    ]
    
    best_c = None
    best_score = -1.0
    
    # Allocate time for each init
    # We have ~140s.
    # Let's run SA+Polish for each.
    # If time is tight, prioritize grid and BCC.
    
    for i, c0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Estimate time remaining
        time_left = deadline - time.time()
        
        # Run optimization
        c_opt = _optimize_from_initial(c0, time.time() + time_left - 2.0)
        
        score = _compute_radius(c_opt)
        
        if score > best_score:
            best_score = score
            best_c = c_opt
            
        # If we found a very good score (close to theoretical max ~0.1113), we can stop early?
        # Theoretical max for 100 spheres is likely around 0.111.
        # If we exceed 0.111, we are doing great.
        
    # Final safety check
    if best_c is None:
        return _build_initial_grid()
        
    return best_c
# EVOLVE-BLOCK-END
