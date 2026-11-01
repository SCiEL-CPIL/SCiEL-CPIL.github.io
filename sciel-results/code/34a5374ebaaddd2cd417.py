import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers > 0.5) or np.any(centers < -0.5):
        return 0.0
    wall_dists = 0.5 - np.abs(centers)
    r_wall = np.min(wall_dists)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    r_pair = np.min(dists[:, 1]) / 2.0
    return min(r_wall, r_pair)

def _build_initializations():
    """Generate a list of initial center configurations."""
    inits = []
    
    # 1. BCC Lattice
    # 100 points. 4x4x4 grid is 64. 5x5x4 is 100.
    # Let's use a 5x5x4 grid for simplicity, then jitter.
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers_grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    inits.append(centers_grid)
    
    # 2. Jittered Grid
    np.random.seed(42)
    for _ in range(3):
        jitter = np.random.normal(0, 0.02, (N, 3))
        c = centers_grid + jitter
        c = np.clip(c, -0.49, 0.49)
        inits.append(c)
        
    # 3. Random
    for _ in range(2):
        c = np.random.uniform(-0.45, 0.45, (N, 3))
        inits.append(c)
        
    return inits

def _violation_and_gradient(x, r):
    """
    Compute violation and gradient for a fixed radius r.
    x is flattened centers.
    """
    centers = x.reshape(N, 3)
    if not np.all(np.isfinite(centers)):
        return 1e10, np.zeros_like(x)
    
    # Clip to valid range to avoid huge gradients from out-of-bounds
    # Note: The optimizer bounds will handle this, but for safety in calculation
    centers_clipped = np.clip(centers, -0.5, 0.5)
    
    # Wall Violations
    wall_dists = 0.5 - np.abs(centers_clipped)
    wall_viol = np.maximum(0, r - wall_dists)
    obj = np.sum(wall_viol ** 2)
    grad = np.zeros_like(centers)
    
    # Gradient for walls
    # d/dx (r - (0.5 - |x|))^2 = 2(r - (0.5 - |x|)) * sign(x)
    # Note: if r - wall_dist > 0, then wall_dist < r
    mask_wall = wall_viol > 0
    if np.any(mask_wall):
        # grad += 2 * wall_viol * sign(centers)
        # Be careful with sign(0)
        signs = np.sign(centers_clipped)
        signs[signs == 0] = 1.0 # arbitrary, doesn't matter much
        grad[mask_wall] += 2.0 * wall_viol[mask_wall, np.newaxis] * signs[mask_wall]
        
    # Pairwise Violations
    tree = cKDTree(centers_clipped)
    # Find pairs with distance < 2r
    # query_ball_point returns list of lists
    # This can be slow. 
    # Alternative: query all pairs if N is small? N=100, 100*99/2 = 4950 pairs.
    # 4950 is small enough to compute all pairwise distances directly in numpy?
    # Yes, (100, 100) matrix.
    
    # Compute all pairwise distances
    diff = centers_clipped[:, np.newaxis, :] - centers_clipped[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j and dist < 2r
    # Create a mask for upper triangle
    mask_pairs = np.triu(np.ones((N, N), dtype=bool), k=1)
    violating_pairs = mask_pairs & (dists < 2.0 * r)
    
    if np.any(violating_pairs):
        # Get indices
        i_idx, j_idx = np.where(violating_pairs)
        
        # Distances for these pairs
        d_pairs = dists[i_idx, j_idx]
        
        # Violation amounts
        v_pairs = 2.0 * r - d_pairs
        
        # Add to objective
        obj += np.sum(v_pairs ** 2)
        
        # Gradient
        # d/dc_i (2r - ||c_i - c_j||)^2 = 2(2r - d) * (c_i - c_j) / d
        # d/dc_j (2r - ||c_i - c_j||)^2 = 2(2r - d) * (c_j - c_i) / d
        
        # Vectorized update
        # diff_ij = centers[i] - centers[j]
        diff_ij = diff[i_idx, j_idx]
        
        # grad_i += 2 * v * diff_ij / d
        # grad_j += 2 * v * (-diff_ij) / d
        
        factor = 2.0 * v_pairs / d_pairs
        # factor shape (M,)
        # diff_ij shape (M, 3)
        grad_contrib = factor[:, np.newaxis] * diff_ij
        
        # Scatter add to grad
        # np.add.at is slow but correct. 
        # For N=100, M is small, so it's fine.
        np.add.at(grad, i_idx, grad_contrib)
        np.add.at(grad, j_idx, -grad_contrib)
        
    return obj, grad.flatten()

def _try_optimize_radius(centers_init, r_target, time_limit):
    """
    Try to find a configuration for r_target starting from centers_init.
    Returns (success, centers).
    """
    x0 = centers_init.flatten().copy()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    start_time = time.time()
    
    def objective_and_grad(x):
        return _violation_and_gradient(x, r_target)
        
    try:
        result = minimize(
            lambda x: _violation_and_gradient(x, r_target)[0],
            x0,
            jac=lambda x: _violation_and_gradient(x, r_target)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'maxfun': 1000,
                'ftol': 1e-10,
                'gtol': 1e-6,
                'maxls': 20,
                'maxcor': 10
            }
        )
        
        final_obj, _ = _violation_and_gradient(result.x, r_target)
        # If violation is small enough, consider it feasible
        if final_obj < 1e-6:
            return True, result.x.reshape(N, 3)
        else:
            return False, result.x.reshape(N, 3)
            
    except Exception:
        return False, centers_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    inits = _build_initializations()
    
    # Pick the best initial configuration
    best_init = None
    best_init_r = -1
    for init in inits:
        r = _compute_radius(init)
        if r > best_init_r:
            best_init_r = r
            best_init = init
            
    current_centers = best_init
    current_r = best_init_r
    
    # Binary Search Bounds
    r_low = current_r
    r_high = 0.12 # Known upper bound is ~0.111, so 0.12 is safe
    
    # We will iterate binary search
    # To avoid getting stuck, we keep a history of "failed" centers? 
    # Or just restart from current_centers?
    # If r_mid fails, we lower r_high. We don't have new centers.
    # If r_mid succeeds, we update current_centers and r_low.
    
    # To improve robustness, we can try a few random restarts if we stall.
    # But let's start with simple binary search.
    
    # We also want to polish the final result.
    
    while time.time() < deadline - 10:
        r_mid = (r_low + r_high) / 2.0
        
        # If r_mid is too close to r_low, break
        if r_high - r_low < 1e-6:
            break
            
        # Try to find feasible solution for r_mid
        # Use current_centers as start
        success, new_centers = _try_optimize_radius(current_centers, r_mid, time_limit=deadline - 5)
        
        if success:
            r_low = r_mid
            current_centers = new_centers
        else:
            r_high = r_mid
            # Optional: If we fail, maybe try a random restart?
            # For now, just lower the bar.
            
    # Final Polish
    # Now we have current_centers that are feasible for r_low.
    # We can try to optimize them to maximize r directly?
    # Or just run one more optimization step for r_low to minimize violation to 0.
    
    # Let's do a final L-BFGS run on the violation for r_low to ensure it's as tight as possible.
    # Actually, the binary search already does this.
    
    # One more check: compute the actual radius of current_centers
    final_r = _compute_radius(current_centers)
    
    # If final_r is significantly less than r_low, something went wrong (numerical issues).
    # But usually they match.
    
    return current_centers
