"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _radius(pts):
    """Compute the largest common radius for the given centres."""
    # Wall distance: min over i,k of (0.5 - |c_ik|)
    # c_ik is coordinate k of point i
    # dist to wall is 0.5 - max(|x|, |y|)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(min_wall, min_pair)

def _penalty(pts_flat, r_target):
    """
    Objective function for L-BFGS-B.
    Minimizes the squared violation of the constraints for radius r_target.
    Constraints:
    1. d_ij >= 2 * r_target
    2. dist_to_wall >= r_target
    """
    pts = pts_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Violations for pairs
    # We want d >= 2*r_target
    viol_pairs = np.maximum(0, 2 * r_target - d)
    # Only sum unique pairs to avoid double counting? 
    # Symmetric matrix, so sum all and divide by 2, or use triu.
    # Using triu_indices for efficiency
    i_idx, j_idx = np.triu_indices(N, k=1)
    viol_pairs_unique = viol_pairs[i_idx, j_idx]
    E_pairs = np.sum(viol_pairs_unique**2)
    
    # Wall distances
    # dist to wall is 0.5 - max(|x|, |y|)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    viol_walls = np.maximum(0, r_target - wall_dists)
    E_walls = np.sum(viol_walls**2)
    
    return E_pairs + E_walls

def _grad_penalty(pts_flat, r_target):
    """
    Gradient of the penalty function.
    """
    pts = pts_flat.reshape(N, 2)
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-8, d, 1.0)
    
    # viol = max(0, 2r - d)
    # d(viol^2)/d(pts) = 2 * viol * d(viol)/d(pts)
    # d(viol)/d(pts_i) = - (pts_i - pts_j) / d_ij  if viol > 0
    # So d(E)/d(pts_i) = sum_j 2 * viol_ij * (-(pts_i - pts_j) / d_ij)
    
    # We only care about viol > 0
    mask = (2 * r_target - d) > 0
    # Set diagonal to False just in case
    np.fill_diagonal(mask, False)
    
    # Weight matrix W_ij = 2 * viol_ij / d_ij
    W = np.zeros((N, N))
    W[mask] = 2 * (2 * r_target - d[mask]) / d_safe[mask]
    
    # grad_i = - sum_j W_ij (pts_i - pts_j) = - (sum_j W_ij) pts_i + sum_j W_ij pts_j
    row_sums = W.sum(axis=1)
    grad_pairs = -row_sums[:, np.newaxis] * pts + W.dot(pts)
    
    # Wall gradient
    # viol_wall = max(0, r - (0.5 - max(|x|,|y|)))
    # Let w_i = 0.5 - max(|x_i|, |y_i|)
    # viol_i = max(0, r - w_i)
    # E_wall = sum viol_i^2
    # dE/d(pts_i) = 2 * viol_i * d(viol_i)/d(pts_i)
    # d(viol_i)/d(pts_i) = - d(w_i)/d(pts_i)
    
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    viol_walls = np.maximum(0, r_target - wall_dists)
    
    # Gradient of wall dist
    # w_i = 0.5 - max(|x|, |y|)
    # If |x| > |y|, w = 0.5 - |x|. dw/dx = -sign(x), dw/dy = 0
    # If |y| > |x|, w = 0.5 - |y|. dw/dx = 0, dw/dy = -sign(y)
    # If tie, average? Or just pick one. It's non-diff at tie, but measure 0.
    
    grad_walls = np.zeros_like(pts)
    for i in range(N):
        if viol_walls[i] > 0:
            abs_x = abs(pts[i, 0])
            abs_y = abs(pts[i, 1])
            if abs_x > abs_y + 1e-9:
                grad_walls[i, 0] = -np.sign(pts[i, 0])
            elif abs_y > abs_x + 1e-9:
                grad_walls[i, 1] = -np.sign(pts[i, 1])
            else:
                # Tie case: subgradient
                grad_walls[i, 0] = -0.5 * np.sign(pts[i, 0])
                grad_walls[i, 1] = -0.5 * np.sign(pts[i, 1])
            
            grad_walls[i] *= 2 * viol_walls[i]
            
    grad = grad_pairs + grad_walls
    return grad.flatten()

def construct():
    t0 = time.time()
    timeout = 140.0 # Safety margin
    
    # Initial guess: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    best_pts = grid.copy()
    best_r = _radius(best_pts)
    
    # Binary search for radius
    # Lower bound: current best
    # Upper bound: theoretical max (approx 0.08)
    r_low = best_r
    r_high = 0.08
    
    rng = np.random.default_rng(42)
    
    # We will try to improve r_low
    # Iterate binary search steps
    max_steps = 20
    for step in range(max_steps):
        if time.time() - t0 > timeout:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        feasible = False
        best_pts_for_r = None
        
        # Multiple restarts to avoid local minima
        n_restarts = 3
        for k in range(n_restarts):
            if time.time() - t0 > timeout - 10:
                break
                
            # Init: Jittered grid or random
            if k == 0:
                init_pts = grid.copy()
                init_pts += rng.normal(0, 0.005, init_pts.shape)
            else:
                init_pts = rng.uniform(-0.5, 0.5, size=(N, 2))
                # Simple push apart for random init to avoid coincident points
                for _ in range(10):
                    d = np.sqrt(np.sum((init_pts[:, None, :] - init_pts[None, :, :]) ** 2, axis=2))
                    np.fill_diagonal(d, np.inf)
                    for i in range(N):
                        for j in range(i + 1, N):
                            if d[i, j] < 0.05:
                                diff = init_pts[i] - init_pts[j]
                                if np.linalg.norm(diff) < 1e-8:
                                    diff = rng.normal(size=2)
                                diff /= np.linalg.norm(diff)
                                init_pts[i] += 0.5 * diff * 0.05
                                init_pts[j] -= 0.5 * diff * 0.05
            init_pts = np.clip(init_pts, -0.5, 0.5)
            
            try:
                res = minimize(
                    _penalty,
                    init_pts.flatten(),
                    args=(r_mid,),
                    jac=_grad_penalty,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 200, 'gtol': 1e-5}
                )
                
                # Check if penalty is close to 0
                if res.fun < 1e-6:
                    feasible = True
                    cand_pts = res.x.reshape(N, 2)
                    # Verify radius
                    r_cand = _radius(cand_pts)
                    if r_cand >= r_mid - 1e-5:
                        if best_pts_for_r is None or r_cand > _radius(best_pts_for_r):
                            best_pts_for_r = cand_pts
            except Exception:
                pass
        
        if feasible and best_pts_for_r is not None:
            r_low = r_mid
            best_pts = best_pts_for_r
            best_r = _radius(best_pts)
        else:
            r_high = r_mid
            # If not feasible, we can't increase r_low.
            # We should probably stop or reduce step size?
            # Binary search usually continues.
            
        # If we've converged enough
        if r_high - r_low < 1e-5:
            break

    # Final polish with Nelder-Mead on the best found points?
    # Or just return best_pts.
    # Nelder-Mead might refine the non-smooth min distance slightly better.
    if time.time() - t0 < timeout - 5:
        try:
            # Minimize negative radius? No, non-smooth.
            # Let's try to polish with L-BFGS-B on a tighter target
            # Or just return.
            pass
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
