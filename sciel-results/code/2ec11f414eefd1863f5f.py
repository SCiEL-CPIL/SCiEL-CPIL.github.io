"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import time
import itertools

def compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Distance from walls
    wall_dists = np.min(np.abs(centers), axis=1) + 0.5  # 0.5 - |c_ik|
    wall_min = np.min(wall_dists)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(centers, metric='euclidean')
    pair_min = np.min(d) / 2.0
    
    return min(wall_min, pair_min)

def smooth_objective_and_grad(centers, beta=100.0):
    """
    Smooth surrogate: minimize -soft_min of all constraints.
    Constraints: pairwise distances >= 2r, wall distances >= r
    We maximize r, so minimize -r.
    Using log-sum-exp for soft-min.
    """
    centers = centers.reshape(-1, 3)
    n = len(centers)
    
    # Wall distances: 0.5 - |c_ik| for each center and each axis
    # We want these >= r, so constraint violation = r - (0.5 - |c_ik|)
    # For soft-min, we look at min of all (distance - r) values
    
    # Pairwise distances / 2
    from scipy.spatial.distance import pdist
    pd = pdist(centers, metric='euclidean') / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_flat = wall_dists.flatten()
    
    # All "margins": pairwise/2 and wall distances
    # We want max of min(all margins)
    # Soft-min via log-sum-exp: soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    
    all_margins = np.concatenate([pd, wall_flat])
    
    # Soft-min
    min_val = np.min(all_margins)
    # Use log-sum-exp for numerical stability
    beta_safe = min(beta, 1000.0)
    exp_vals = np.exp(-beta_safe * (all_margins - min_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta_safe
    
    # Objective: maximize soft_min, so minimize -soft_min
    obj = -soft_min
    
    # Gradient
    # d(soft_min)/d(x_i) = -1/beta * sum_j exp(-beta*(x_j - min_val)) * (-beta) * dx_j/dx_i / sum_j exp(-beta*(x_j - min_val))
    # = sum_j exp(-beta*(x_j - min_val)) * dx_j/dx_i / sum_j exp(-beta*(x_j - min_val))
    
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of pairwise distances
    grad = np.zeros_like(centers)
    
    # Pairwise: for each pair (i,j), margin = ||c_i - c_j||/2
    # d(margin_ij)/d(c_i) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d(margin_ij)/d(c_j) = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # Get pairwise indices
    ii, jj = np.triu_indices(n, k=1)
    diffs = centers[ii] - centers[jj]
    dists = np.linalg.norm(diffs, axis=1)
    dists_safe = np.maximum(dists, 1e-12)
    
    # d(margin)/d(c_i) for pair (i,j) = (c_i - c_j) / (2 * dist)
    pair_grads = diffs / (2.0 * dists_safe[:, None])
    
    # Weighted sum
    for k in range(len(ii)):
        grad[ii[k]] += weights[k] * pair_grads[k]
        grad[jj[k]] += weights[k] * (-pair_grads[k])
    
    # Wall distances: margin = 0.5 - |c_ik|
    # d(margin)/d(c_ik) = -sign(c_ik)
    wall_grads = -np.sign(centers)
    wall_weights = weights[len(pd):].reshape(-1, 3)
    grad += wall_weights * wall_grads
    
    # Objective gradient: d(-soft_min)/d(c) = -d(soft_min)/d(c) = -grad
    # Wait: obj = -soft_min, so d(obj)/d(c) = -d(soft_min)/d(c)
    # d(soft_min)/d(c) = grad (computed above as the gradient of soft_min w.r.t. centers)
    # So d(obj)/d(c) = -grad
    
    return obj, -grad

def polish(centers, max_iter=500, tol=1e-12):
    """High-precision polish using exact min (nonsmooth) with small steps."""
    centers = centers.copy()
    n = len(centers)
    
    from scipy.spatial.distance import pdist
    
    for it in range(max_iter):
        # Compute all margins
        pd = pdist(centers, metric='euclidean') / 2.0
        wall_dists = (0.5 - np.abs(centers)).flatten()
        all_margins = np.concatenate([pd, wall_dists])
        current_r = np.min(all_margins)
        
        # Find the critical pair(s) and wall(s)
        min_idx = np.argmin(all_margins)
        
        # Try to increase r by moving centers
        # Simple approach: find the tightest constraint and push apart
        if min_idx < len(pd):
            # Pairwise constraint
            ii, jj = np.triu_indices(n, k=1)
            k = min_idx
            i, j = ii[k], jj[k]
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            # Push i and j apart
            step = 0.001 * current_r
            centers[i] += step * diff / dist
            centers[j] -= step * diff / dist
        else:
            # Wall constraint
            k = min_idx - len(pd)
            ci = k // 3
            axis = k % 3
            # Push center away from wall
            sign = np.sign(centers[ci, axis])
            if sign == 0:
                sign = 1.0
            step = 0.001 * current_r
            centers[ci, axis] += step * sign
        
        # Check if improved
        new_pd = pdist(centers, metric='euclidean') / 2.0
        new_wall = (0.5 - np.abs(centers)).flatten()
        new_all = np.concatenate([new_pd, new_wall])
        new_r = np.min(new_all)
        
        if new_r <= current_r + tol:
            # Try smaller step
            if step < 1e-15:
                break
            # Revert and try smaller step
            centers[ii[k] if min_idx < len(pd) else ci, jj[k] if min_idx < len(pd) else axis] *= 1.0  # no-op
            # Just break if no improvement
            if it > 10:
                break
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Multi-start with different initial configurations
    np.random.seed(42)
    
    # Start 1: Grid-based
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_starts = [grid[:N]]
    
    # Start 2: Random
    for _ in range(5):
        rand_centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initial_starts.append(rand_centers)
    
    # Start 3: Body-centered cubic-like
    for shift in [0.0, 0.25]:
        m2 = int(np.ceil((N / 2) ** (1.0 / 3.0) - 1e-9))
        ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
        grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
        if len(grid2) >= N:
            initial_starts.append(grid2[:N] + shift)
    
    for start_idx, init in enumerate(initial_starts):
        if time.time() > deadline - 20.0:
            break
        
        # Optimize with smooth objective
        x0 = init.flatten()
        
        try:
            result = minimize(
                smooth_objective_and_grad,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < -best_radius:
                centers = result.x.reshape(-1, 3)
                # Ensure inside cube
                centers = np.clip(centers, -0.4999, 0.4999)
                r = compute_radius(centers)
                
                if r > best_radius:
                    best_radius = r
                    best_centers = centers.copy()
                    
                    # Polish
                    if time.time() < deadline - 10.0:
                        polished = polish(centers, max_iter=200)
                        polished = np.clip(polished, -0.4999, 0.4999)
                        r_polished = compute_radius(polished)
                        if r_polished > best_radius:
                            best_radius = r_polished
                            best_centers = polished.copy()
                            
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if best_centers is not None and time.time() < deadline - 5.0:
        final = polish(best_centers, max_iter=500, tol=1e-13)
        final = np.clip(final, -0.4999, 0.4999)
        r_final = compute_radius(final)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
