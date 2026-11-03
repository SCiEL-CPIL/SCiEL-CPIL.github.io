"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _compute_radius(circles, W, H):
    """Compute max feasible radius given positions and rectangle.
    circles: (n, 2) array of centers.
    Returns the max r such that all circles fit and are disjoint.
    """
    n = len(circles)
    # Constraint 1: each circle must fit in rectangle
    r_boundary = np.min(np.minimum(np.minimum(circles[:, 0], W - circles[:, 0]),
                                   np.minimum(circles[:, 1], H - circles[:, 1])))
    # Constraint 2: pairwise distances
    r_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(circles[i] - circles[j])
            r_pair = min(r_pair, d / 2.0)
    return max(0.0, min(r_boundary, r_pair))


def _validate(circles, W, H, r):
    """Check if circles with radius r fit in W x H rectangle without overlap."""
    if r <= 0:
        return False
    if W <= 0 or H <= 0:
        return False
    # Boundary check
    if np.any(circles[:, 0] < r - 1e-9) or np.any(circles[:, 0] > W - r + 1e-9):
        return False
    if np.any(circles[:, 1] < r - 1e-9) or np.any(circles[:, 1] > H - r + 1e-9):
        return False
    # Pairwise distance check
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            d = np.linalg.norm(circles[i] - circles[j])
            if d < 2 * r - 1e-9:
                return False
    return True


def _smooth_objective(params, W, H, r_target):
    """Smooth objective: penalize violations of radius r_target.
    Lower is better. Returns 0 if all constraints satisfied with margin.
    """
    centers = params.reshape(-1, 2)
    n = len(centers)
    
    # Boundary violations
    boundary_penalty = 0.0
    for i in range(n):
        x, y = centers[i]
        # Distance to each boundary
        dists = [x, W - x, y, H - y]
        min_dist = min(dists)
        if min_dist < r_target:
            boundary_penalty += (r_target - min_dist) ** 2
    
    # Pairwise distance violations
    pair_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            min_d = 2 * r_target
            if d < min_d:
                pair_penalty += (min_d - d) ** 2
    
    total_penalty = boundary_penalty + pair_penalty
    return total_penalty


def _try_optimize(centers, W, H, r_target, deadline, max_iter=200):
    """Try to optimize centers to achieve radius r_target using L-BFGS-B."""
    def objective(params):
        return _smooth_objective(params, W, H, r_target)
    
    x0 = centers.flatten()
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            return None, None
        
        if result.fun < 1e-6:
            # Check if actually feasible
            new_centers = result.x.reshape(-1, 2)
            if _validate(new_centers, W, H, r_target):
                return new_centers, r_target
    except Exception:
        pass
    
    return None, None


def _pack_for_aspect(W, H, deadline):
    """Pack circles in W x H rectangle. Returns (circles, W, H, sum_radii)."""
    # Initial grid placement
    cols = max(1, int(np.round(np.sqrt(N * W / H))))
    rows = max(1, int(np.ceil(N / cols)))
    
    # Scale to fill rectangle
    cell_w = W / cols
    cell_h = H / rows
    
    centers = np.zeros((N, 2))
    for i in range(N):
        c = i % cols
        r = i // cols
        centers[i, 0] = (c + 0.5) * cell_w
        centers[i, 1] = (r + 0.5) * cell_h
    
    # Clamp to valid positions
    r_est = min(cell_w, cell_h) / 2.0
    centers[:, 0] = np.clip(centers[:, 0], r_est, W - r_est)
    centers[:, 1] = np.clip(centers[:, 1], r_est, H - r_est)
    
    best_centers = centers.copy()
    best_r = _compute_radius(centers, W, H)
    best_sum = best_r * N
    
    # Iterative radius growth with optimization
    r = best_r
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Try to grow radius
        r_new = r * 1.03  # grow by 3%
        
        # Attempt optimization to achieve r_new
        new_centers, achieved_r = _try_optimize(best_centers, W, H, r_new, deadline, max_iter=150)
        
        if new_centers is not None:
            best_centers = new_centers
            best_r = achieved_r
            best_sum = best_r * N
            r = best_r
        else:
            # Try smaller growth
            r_new = r * 1.01
            new_centers, achieved_r = _try_optimize(best_centers, W, H, r_new, deadline, max_iter=100)
            if new_centers is not None:
                best_centers = new_centers
                best_r = achieved_r
                best_sum = best_r * N
                r = best_r
            else:
                # Can't grow further, break
                break
    
    # Final polish: try to optimize at current best_r
    if time.time() < deadline - 5:
        new_centers, achieved_r = _try_optimize(best_centers, W, H, best_r, deadline, max_iter=300)
        if new_centers is not None and achieved_r >= best_r:
            best_centers = new_centers
            best_r = achieved_r
            best_sum = best_r * N
    
    return best_centers, W, H, best_sum


def _random_init(W, H, seed=None):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N, 2))
    
    # Try to place circles with some spacing
    r_est = min(W, H) / (2 * np.ceil(np.sqrt(N)))
    
    for i in range(N):
        for attempt in range(100):
            x = rng.uniform(r_est, W - r_est)
            y = rng.uniform(r_est, H - r_est)
            
            # Check distance to existing circles
            min_dist = np.inf
            for j in range(i):
                d = np.linalg.norm([x, y] - centers[j])
                min_dist = min(min_dist, d)
            
            if min_dist > 2 * r_est * 0.8:
                centers[i] = [x, y]
                break
        else:
            # Fallback: place at random position
            centers[i] = [rng.uniform(r_est, W - r_est), rng.uniform(r_est, H - r_est)]
    
    return centers


def construct():
    """Construct 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_sum = 0.0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(W + H) = 4, so W + H = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4, 0.75, 1.33, 0.9, 1.1, 0.65, 1.55]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W * H with W + H = 2
        # W = ar * H, so H = 2/(1+ar), W = 2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        # Try multiple random restarts for this aspect ratio
        best_for_ar = 0.0
        best_centers_for_ar = None
        
        for restart in range(3):
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Use grid init
                centers, W_out, H_out, s = _pack_for_aspect(W, H, deadline)
            else:
                # Use random init
                rng_seed = int(time.time() * 1000) + restart * 100
                centers = _random_init(W, H, seed=rng_seed)
                best_centers = centers.copy()
                best_r = _compute_radius(centers, W, H)
                best_sum_ar = best_r * N
                
                # Try to optimize from random start
                for iter_num in range(50):
                    if time.time() > deadline:
                        break
                    r_new = best_r * 1.03
                    new_centers, achieved_r = _try_optimize(best_centers, W, H, r_new, deadline, max_iter=100)
                    if new_centers is not None:
                        best_centers = new_centers
                        best_r = achieved_r
                        best_sum_ar = best_r * N
                    else:
                        r_new = best_r * 1.01
                        new_centers, achieved_r = _try_optimize(best_centers, W, H, r_new, deadline, max_iter=50)
                        if new_centers is not None:
                            best_centers = new_centers
                            best_r = achieved_r
                            best_sum_ar = best_r * N
                        else:
                            break
                
                centers = best_centers
                s = best_sum_ar
            
            if s > best_for_ar:
                best_for_ar = s
                best_centers_for_ar = centers
        
        if best_for_ar > best_sum:
            best_sum = best_for_ar
            best_result = (best_centers_for_ar, W, H)
    
    if best_result is None:
        # Fallback: square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    centers, W, H = best_result
    r = _compute_radius(centers, W, H)
    
    # Build output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = r
    
    return result
