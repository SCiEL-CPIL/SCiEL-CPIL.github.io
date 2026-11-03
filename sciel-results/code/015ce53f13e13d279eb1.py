"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    if len(pts) < 2:
        return 0.5
    n = len(pts)
    # Vectorized pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to walls
    min_wall = np.inf
    for k in range(3):
        d = 0.5 - np.abs(pts[:, k])
        min_wall = min(min_wall, np.min(d))
    
    return min(min_dist / 2.0, min_wall)

def _logsumexp(x, beta):
    """Stable log-sum-exp computation."""
    max_x = np.max(x)
    if np.isinf(max_x):
        return max_x
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def _surrogate_loss(pts, beta):
    """
    Smooth surrogate for the negative minimum distance.
    We want to maximize min(d_ij, d_wall).
    Equivalent to minimizing -min(d_ij, d_wall).
    Using log-sum-exp on negative distances:
    L = (1/beta) * log(sum(exp(-beta * d_ij))) approximates -min(d_ij) as beta -> inf.
    We combine sphere-sphere and sphere-wall distances.
    """
    pts = pts.reshape(-1, 3)
    n = len(pts)
    
    # Sphere-sphere distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Only upper triangle to avoid double counting, but for LSE it doesn't matter much 
    # except for the constant factor. Let's use all pairs i<j.
    # Actually, standard LSE over all pairs is fine.
    # We need to exclude diagonal.
    dists_upper = dists[np.triu_indices(n, k=1)]
    
    # Wall distances
    # For each point and each dimension, distance to nearest wall.
    # d_wall_i = min_k (0.5 - |x_ik|)
    # Actually, the constraint is x_ik + r <= 0.5 and -x_ik + r <= 0.5.
    # Distance to boundary in dim k is min(0.5 - x_ik, 0.5 + x_ik).
    # Since we are in [-0.5, 0.5], dist to wall is min(0.5 - x, 0.5 + x).
    wall_dists = np.empty((n, 3))
    for k in range(3):
        wall_dists[:, k] = 0.5 - np.abs(pts[:, k])
    
    # We want to maximize the minimum of all these distances.
    # So we minimize the "soft min" which is -(1/beta) log sum exp(-beta * d).
    # Let's combine all distances into one array.
    all_dists = np.concatenate([dists_upper, wall_dists.flatten()])
    
    # Filter out very small or invalid distances just in case
    all_dists = all_dists[all_dists > 1e-6]
    
    if len(all_dists) == 0:
        return 0.0
        
    loss = -_logsumexp(-beta * all_dists, 1.0) / beta
    return loss

def _surrogate_grad(pts, beta):
    """
    Gradient of the log-sum-exp surrogate.
    L = -1/beta * log(sum_j exp(-beta * d_j))
    dL/dx = -1/beta * (1/sum(exp)) * sum_j (-exp(-beta*d_j) * beta * grad_x d_j)
          = (1/sum(exp)) * sum_j exp(-beta*d_j) * grad_x d_j
    Note: d_j is distance. grad_x d_j is the unit vector from j to i (for i-j) or normal to wall.
    """
    pts = pts.reshape(-1, 3)
    n = len(pts)
    
    # Sphere-sphere
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2) # (N, N)
    np.fill_diagonal(dists, 1.0) # Avoid div by zero, will mask later
    
    # Weights w_ij = exp(-beta * d_ij)
    weights = np.exp(-beta * dists)
    np.fill_diagonal(weights, 0.0)
    
    # Sum of weights for normalization
    total_weight = np.sum(weights)
    
    # Gradient contribution from sphere-sphere
    # dL/dx_i = sum_j w_ij / total_weight * (x_i - x_j) / d_ij
    # Note: grad_x_i d_ij = (x_i - x_j) / d_ij
    # So term is w_ij * (x_i - x_j) / d_ij
    # Vectorized:
    # grad_ss[i] = sum_j (w_ij / d_ij) * (x_i - x_j)
    inv_dists = np.zeros_like(dists)
    np.fill_diagonal(inv_dists, 0.0)
    # Avoid division by zero where dist is very small (though weights would be huge)
    # But if dist is 0, it's a problem. We assume distinct points.
    # Use safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists = np.where(dists > 1e-8, 1.0/dists, 0.0)
        
    coeff = weights * inv_dists
    np.fill_diagonal(coeff, 0.0)
    
    # grad_ss = sum_j coeff[i,j] * diff[i,j]
    grad_ss = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    # Wall distances
    # d_wall_ik = 0.5 - |x_ik|
    # grad_x_ik d_wall_ik = -sign(x_ik) if x_ik != 0
    # Weight for wall constraint k at point i: w_wall_ik = exp(-beta * d_wall_ik)
    
    wall_dists = np.empty((n, 3))
    wall_grads = np.empty((n, 3))
    wall_weights = np.empty((n, 3))
    
    for k in range(3):
        wall_dists[:, k] = 0.5 - np.abs(pts[:, k])
        # Gradient of (0.5 - |x|) wrt x is -sign(x)
        wall_grads[:, k] = -np.sign(pts[:, k])
        # If x is exactly 0, gradient is undefined, but subgradient is [-1, 1]. 
        # Using 0 is a reasonable choice or we can just leave it.
        wall_grads[:, k] = np.where(pts[:, k] == 0, 0.0, wall_grads[:, k])
        wall_weights[:, k] = np.exp(-beta * wall_dists[:, k])
        
    # Total weight includes wall weights
    total_weight += np.sum(wall_weights)
    
    # Normalize
    if total_weight < 1e-10:
        return np.zeros_like(pts)
        
    grad_ss /= total_weight
    grad_wall = wall_weights * wall_grads / total_weight
    
    grad_total = grad_ss + grad_wall
    
    return grad_total

def _optimize_l_bfgs(pts_init, beta, time_limit, max_iter=1000):
    """
    Run L-BFGS-B on the surrogate loss.
    """
    start_time = time.time()
    x0 = pts_init.flatten()
    
    # We need to handle the time limit inside the callback or just rely on max_iter
    # and check time periodically. Since L-BFGS-B doesn't support easy time-based stopping 
    # in the middle of an iteration, we will estimate.
    
    # Bounds: points must be in [-0.5, 0.5]. 
    # Actually, to allow the optimizer to "see" the wall penalty, we should let it go slightly out 
    # and penalize, OR use bounds.
    # Using bounds is safer for validity. But if a point is at the bound, the gradient 
    # from the wall term pushes it inward, which is correct.
    # However, L-BFGS-B with bounds projects the solution.
    # Let's use bounds [-0.5, 0.5].
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0.copy()
    best_loss = _surrogate_loss(x0, beta)
    
    # Callback to check time and update best
    def callback(xk):
        nonlocal best_x, best_loss
        if time.time() - start_time > time_limit - 2.0:
            # Raise an exception to stop? No, scipy doesn't handle that cleanly.
            # We'll just let it finish the iteration and check after.
            # To be safe, we can raise a custom exception or just return True to stop?
            # Returning True from callback stops the optimization.
            # But we want to stop *before* the time limit.
            # If we return True, it stops.
            # Let's check time before each iteration? No, callback is per iteration.
            # If time is up, return True.
            return True
            
        current_loss = _surrogate_loss(xk, beta)
        if current_loss < best_loss:
            best_loss = current_loss
            best_x = xk.copy()
        return False

    try:
        res = minimize(
            _surrogate_loss,
            x0,
            jac=_surrogate_grad,
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20},
            callback=callback
        )
    except Exception as e:
        # If something goes wrong, return best found
        pass
    
    # Check final result
    final_loss = _surrogate_loss(res.x, beta)
    if final_loss < best_loss:
        best_loss = final_loss
        best_x = res.x.copy()
        
    return best_x.reshape(-1, 3), best_loss

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    best_pts = None
    best_r = -1.0
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts_init = grid[:N].copy()
    
    # Add small noise to break symmetry
    rng = np.random.RandomState(42)
    pts_init += rng.uniform(-0.01, 0.01, pts_init.shape)
    pts_init = np.clip(pts_init, -0.49, 0.49)
    
    # Strategy: 
    # 1. Run L-BFGS-B with increasing beta (annealing) to smooth out the landscape initially 
    #    and sharpen it later.
    # 2. Alternatively, run with a fixed high beta but multiple restarts if time permits.
    # Given the smoothness of LSE, a single long run with moderate beta might be best, 
    # or a sequence of runs with increasing beta.
    
    # Let's try a sequence of betas.
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_pts = pts_init
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        remaining = time_limit - (time.time() - start_time) - 2.0
        if remaining <= 0:
            continue
            
        # Allocate time proportional to beta? Or equal chunks?
        # Higher beta is harder, so maybe give it more time?
        # Or just split remaining time evenly among remaining betas.
        num_remaining = len(betas) - betas.index(beta)
        chunk_time = remaining / num_remaining
        
        # Min 5s per run, max 30s per run
        chunk_time = max(5.0, min(30.0, chunk_time))
        
        current_pts, _ = _optimize_l_bfgs(current_pts, beta, chunk_time)
        
        # Evaluate actual radius
        r = _compute_radius(current_pts)
        if r > best_r:
            best_r = r
            best_pts = current_pts.copy()
            
    # Final polish: if time remains, do a very high beta run
    if best_pts is not None and time.time() - start_time < time_limit - 5.0:
        remaining = time_limit - (time.time() - start_time) - 2.0
        if remaining > 5.0:
            current_pts, _ = _optimize_l_bfgs(best_pts, 200.0, remaining)
            r = _compute_radius(current_pts)
            if r > best_r:
                best_r = r
                best_pts = current_pts.copy()
                
    if best_pts is None:
        best_pts = pts_init
        best_r = _compute_radius(best_pts)
        
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    final_r = _compute_radius(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
