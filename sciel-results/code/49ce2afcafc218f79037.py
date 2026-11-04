import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Minimum distance to boundary
    min_boundary = HALF - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_obj_grad(x, beta):
    """
    Soft-min objective and gradient.
    Minimizes this objective maximizes the minimum distance (radius).
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0) # Avoid div by zero, value will be ignored
    
    # D_ij = dists / 2
    D = dists / 2.0
    
    # Wall distances
    wall_dists = HALF - np.abs(c)
    W = wall_dists
    
    # We want to minimize J = (1/beta) * log( sum exp(-beta * D_ij) + sum exp(-beta * W_ik) )
    # To avoid overflow/underflow, we can subtract a baseline, but let's try direct first.
    # If beta is large, exp(-beta * D) is small.
    
    # Calculate terms
    # Mask for i < j to avoid double counting and self terms
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # exp terms for pairs
    exp_pairs = np.exp(-beta * D)
    exp_pairs[~mask] = 0.0
    
    # exp terms for walls
    exp_walls = np.exp(-beta * W)
    
    sum_pairs = np.sum(exp_pairs)
    sum_walls = np.sum(exp_walls)
    S = sum_pairs + sum_walls
    
    if S == 0:
        return 0.0, np.zeros_like(x)
        
    obj = (1.0 / beta) * np.log(S)
    
    # Gradient
    # grad J = (1 / (beta * S)) * grad S
    # grad S = sum grad exp_pairs + sum grad exp_walls
    
    # grad exp_pairs w.r.t c_i
    # d/dc_i exp(-beta * D_ij) = exp(-beta * D_ij) * (-beta) * dD_ij/dc_i
    # dD_ij/dc_i = 0.5 * (c_i - c_j) / dist_ij
    # So term = -0.5 * beta * exp(-beta * D_ij) / dist_ij * (c_i - c_j)
    
    # Vectorized:
    # We need to sum over j for each i.
    # weight_ij = -0.5 * beta * exp_pairs[i,j] / dists[i,j]
    # grad_pair_i = sum_j weight_ij * (c_i - c_j)
    
    # Note: exp_pairs is 0 for i>=j.
    # We need dists for normalization.
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    
    weights = -0.5 * beta * exp_pairs / safe_dists
    # weights[i,j] is the coefficient for (c_i - c_j)
    
    # grad_pair_i = sum_j weights[i,j] * (c_i - c_j)
    # = sum_j weights[i,j] * c_i - sum_j weights[i,j] * c_j
    # = c_i * sum_j weights[i,j] - sum_j weights[i,j] * c_j
    
    sum_weights = np.sum(weights, axis=1) # shape (N,)
    # sum_j weights[i,j] * c_j
    weighted_c = np.sum(weights[:, :, np.newaxis] * c[np.newaxis, :, :], axis=1) # shape (N, 3)
    
    grad_pairs = sum_weights[:, np.newaxis] * c - weighted_c
    
    # grad exp_walls w.r.t c_ik
    # d/dc_ik exp(-beta * W_ik) = exp(-beta * W_ik) * (-beta) * dW_ik/dc_ik
    # W_ik = 0.5 - |c_ik|
    # dW_ik/dc_ik = -sign(c_ik)
    # So term = -beta * exp(-beta * W_ik) * (-sign(c_ik)) = beta * exp(-beta * W_ik) * sign(c_ik)
    
    signs = np.sign(c)
    grad_walls = beta * exp_walls * signs
    
    total_grad_c = grad_pairs + grad_walls
    total_grad = (1.0 / (beta * S)) * total_grad_c.flatten()
    
    return obj, total_grad

def _optimize_soft_min(init_pts, beta, max_iter=1000, time_limit=50.0):
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _soft_min_obj_grad(x, beta)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-10, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def _init_grid():
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = _init_grid()
    best_points = points.copy()
    best_r = _compute_radius(best_points)
    
    rng = np.random.RandomState(42)
    current_pts = best_points.copy()
    
    # Anneal beta from small to large
    betas = [10, 50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining_time = deadline - (time.time() - start_time) - 5.0
        if remaining_time < 2.0:
            break
            
        opt_pts, success = _optimize_soft_min(
            init_pts, 
            beta, 
            max_iter=2000, 
            time_limit=remaining_time
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_points = opt_pts.copy()
                current_pts = opt_pts.copy()
            else:
                # Even if radius didn't improve, the configuration might be "tighter" 
                # and better for the next higher beta.
                # However, if it got worse, we should probably revert to best_points.
                # Let's keep current_pts as the result of the optimization step, 
                # as it minimizes the soft-min, which is a good proxy for packing quality.
                current_pts = opt_pts.copy()

    # Final polish with hard constraints to ensure validity and squeeze a bit more
    # Use the original hard energy method for a final pass?
    # Or just return best_points.
    
    return best_points

# EVOLVE-BLOCK-END
