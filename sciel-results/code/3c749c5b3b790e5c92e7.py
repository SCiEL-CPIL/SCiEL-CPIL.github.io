import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time as _time

_DEADLINE = _time.time() + 140.0  # 150s limit, keep 10s margin
_N = 100

def _compute_radius(centers):
    """Compute the exact largest common radius for given centers inside [-0.5, 0.5]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (_N, 3):
        return 0.0
    # Check for non-finite values
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    half = 0.5
    # Distance from each center to cube walls
    # c_ik is coordinate k of center i. Wall distance is min(0.5 - c, 0.5 + c) = 0.5 - |c|
    wall_dists = np.min(half - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use broadcasting for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_objective(x, beta):
    """
    Compute the soft-min of the distances (radius candidates).
    x: flattened array of shape (N*3,)
    beta: sharpness parameter
    Returns: -soft_min(distances) so we can minimize it? 
             Actually, we want to MAXIMIZE the min distance.
             So we MINIMIZE -min distance.
             Soft-min approximates min. So we minimize soft_min.
             Wait, if soft_min approximates min, and we want max min, 
             we minimize -soft_min? No.
             Let r = min(distances). We want to maximize r.
             Equivalent to minimizing -r.
             Let S_beta(x) = soft_min(distances). S_beta approx min.
             So we minimize -S_beta(x)? No, if S_beta is positive, minimizing -S_beta maximizes S_beta.
             But usually soft-min is defined to be close to min.
             Let's define f(x) = - soft_min(distances, beta).
             Then minimizing f(x) maximizes soft_min, which approximates max(min).
             
             Let's stick to: minimize g(x) = - soft_min(distances, beta).
    """
    centers = x.reshape(_N, 3)
    
    # Calculate pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    half = 0.5
    wall_dists = np.min(half - np.abs(centers), axis=1)
    
    # Combine all distance candidates (radius limits)
    # Radius candidates are d_ij / 2 and w_i
    # We want the minimum of these.
    # Let's flatten the upper triangle of dists/2 and wall_dists
    iu = np.triu_indices(_N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Stable soft-min calculation
    # soft_min = min_val - (1/beta) * log(sum(exp(-beta * (dists - min_val))))
    min_val = np.min(all_dists)
    # Avoid overflow/underflow by shifting
    shifted = all_dists - min_val
    # exp(-beta * shifted)
    # If beta is large, shifted > 0 leads to exp(-large) -> 0. shifted == 0 leads to 1.
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_val - (1.0 / beta) * log_sum
    
    # We want to maximize soft_min_val, so we minimize -soft_min_val
    return -soft_min_val

def _build_initial_grid(jitter=0.0):
    """Build a dense grid and pick N points with good separation."""
    # Use a 5x5x4 = 100 grid exactly
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    if jitter > 0:
        # Add small random jitter to break symmetry
        noise = np.random.randn(*grid.shape) * jitter
        grid = grid + noise
        # Clip to ensure inside
        np.clip(grid, -0.49, 0.49, out=grid)
        
    return grid[:_N]

def construct():
    start_time = _time.time()
    
    # Try a few different initializations
    best_centers = None
    best_r = -np.inf
    
    # 1. Pure grid
    c_grid = _build_initial_grid(jitter=0.0)
    r_grid = _compute_radius(c_grid)
    if r_grid > best_r:
        best_r = r_grid
        best_centers = c_grid.copy()
        
    # 2. Jittered grid
    np.random.seed(42) # Reproducibility for the jitter
    c_jitter = _build_initial_grid(jitter=0.01)
    r_jitter = _compute_radius(c_jitter)
    if r_jitter > best_r:
        best_r = r_jitter
        best_centers = c_jitter.copy()
        
    # Start optimization from the best initial guess
    x0 = best_centers.flatten()
    bounds = [(-0.5, 0.5)] * (_N * 3)
    
    # Adaptive beta schedule
    betas = [10, 50, 100, 500, 1000, 2000]
    current_x = x0.copy()
    
    for beta in betas:
        if _time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            # Check if result is valid
            if res.success or res.status == 0:
                new_centers = res.x.reshape(_N, 3)
                new_r = _compute_radius(new_centers)
                
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
                    current_x = res.x.copy()
                else:
                    # If it didn't improve, try to keep the previous best state for next beta
                    # But maybe res.x is closer to the true min for this beta?
                    # Let's just use res.x as the starting point for the next sharper beta
                    current_x = res.x.copy()
            else:
                # If optimization failed, keep current_x
                pass
                
        except Exception as e:
            # If anything goes wrong, skip this beta step
            pass
            
    # Final polish with Nelder-Mead on the best found center
    if _time.time() < _DEADLINE - 5 and best_centers is not None:
        try:
            # Use a tighter tolerance and more iterations
            def neg_radius(c):
                c = np.reshape(c, (-1, 3))
                return -_compute_radius(c)
            
            res_nm = minimize(
                neg_radius,
                best_centers.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_nm.success:
                new_centers = res_nm.x.reshape(_N, 3)
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
        except Exception:
            pass

    # Final safety check
    if best_centers is None:
        # Fallback to grid
        best_centers = _build_initial_grid(jitter=0.0)
        
    # Ensure return type is correct
    return best_centers

# EVOLVE-BLOCK-END
