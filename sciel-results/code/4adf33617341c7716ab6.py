"""Optimized packing of 200 equal circles in a unit square."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def _hexagonal_init(n):
    """
    Generate initial positions using a hexagonal lattice pattern.
    We want to place n points in [-0.5, 0.5]^2.
    A hexagonal lattice has rows offset by half the spacing.
    """
    # Estimate spacing
    # Area per point ~ 1/n
    # Hex lattice area per point = (sqrt(3)/2) * s^2
    # s ~ sqrt(2 / (sqrt(3) * n))
    s = np.sqrt(2.0 / (np.sqrt(3.0) * n))
    
    # Number of points per row
    # Width is 1.0. Number of points in a row ~ 1/s
    pts_per_row = int(np.ceil(1.0 / s))
    
    points = []
    y = -0.5 + s / 2  # Start near bottom
    row_idx = 0
    while len(points) < n and y <= 0.5:
        offset = 0.0
        if row_idx % 2 == 1:
            offset = s / 2.0
        
        x = -0.5 + s / 2 + offset
        while x <= 0.5 and len(points) < n:
            # Check if x is within bounds (with small margin)
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row_idx += 1
        
    # If we have too many, trim
    points = points[:n]
    
    # If we have too few (shouldn't happen with this logic, but just in case),
    # fill with random points or grid
    if len(points) < n:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = list(grid[:n])
        
    return np.array(points)


def _score(pts):
    """
    Calculate the maximum radius for given points.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    # Pairwise distances
    # For efficiency, we can just compute the minimum distance.
    # N=200, so N^2 = 40000 distances. This is fast.
    dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Boundary distance
    # min_i min_k (0.5 - |c_ik|)
    # This is equivalent to 0.5 - max_i max_k |c_ik|
    max_coord = np.max(np.abs(pts))
    boundary_dist = 0.5 - max_coord
    
    return min(min_dist / 2.0, boundary_dist)


def _optimize(pts, max_iters=5000, tol=1e-10, time_limit=100.0):
    """
    Optimize the packing using a gradient-based approach on the objective function.
    We maximize r, which is min(min_dist/2, boundary_dist).
    This is a non-smooth function. We can use a smooth approximation or
    simply use a subgradient method.
    
    A better approach: 
    Let r be the variable. We want to maximize r such that:
    1. |c_i - c_j| >= 2r for all i < j
    2. |c_ik| <= 0.5 - r for all i, k
    
    This is a non-convex problem.
    
    Alternative: Maximize the sum of pairwise distances? No, that's not the same.
    
    Let's use a penalty method or simply optimize the minimum distance directly.
    Objective: f(pts) = min( min_{i<j} |c_i - c_j|, 2*(0.5 - max(|c_ik|)) )
    
    We can use a simple hill climbing / gradient ascent with subgradients.
    Or, we can use `scipy.optimize.minimize` with a negative objective.
    
    Let's try a custom gradient ascent.
    Gradient of min_dist/2:
    The term that achieves the minimum dominates.
    If min_dist < 2*boundary_dist, then gradient comes from the closest pair.
    If boundary_dist < min_dist/2, then gradient comes from the point closest to the boundary.
    
    Actually, let's use a smoother surrogate or just use scipy.optimize.
    `scipy.optimize.minimize` with method='L-BFGS-B' might struggle with the non-smoothness.
    
    Let's try maximizing the sum of squared distances? No.
    
    Let's stick to the subgradient idea but implement it carefully.
    Or, we can use a "soft-min" approximation:
    soft_min(x, y) = -log(exp(-x/t) + exp(-y/t)) * t
    As t -> 0, it approaches min(x, y).
    
    Let's use the soft-min on the two terms:
    d_min = min_{i<j} |c_i - c_j|
    b_min = 2 * (0.5 - max(|c_ik|)) = 1 - 2*max(|c_ik|)
    
    We want to maximize min(d_min/2, b_min/2) which is min(d_min, b_min)/2.
    
    Let's define objective to maximize:
    F = -soft_min(d_min, b_min, t)
    And minimize F.
    
    However, calculating gradients for d_min and b_min is tricky.
    
    Let's try a simpler approach: 
    1. Start with hex lattice.
    2. Use local search: for each point, move it in small random directions or along gradients.
    
    Actually, `scipy.optimize.minimize` with a custom objective and gradient is probably best.
    
    Let's implement the gradient of the soft-min.
    Let u = d_min, v = b_min.
    f = -t * log( exp(-u/t) + exp(-v/t) )
    df/du = exp(-u/t) / (exp(-u/t) + exp(-v/t))
    df/dv = exp(-v/t) / (exp(-u/t) + exp(-v/t))
    
    Now, u = min_{i<j} |c_i - c_j|.
    The gradient of u with respect to c_k is 0 for most k, and for the pair (i,j) that achieves the min, it's along the vector connecting them.
    Specifically, if |c_i - c_j| is the minimum, then:
    d( |c_i - c_j| ) / dc_i = -(c_i - c_j) / |c_i - c_j|
    d( |c_i - c_j| ) / dc_j = (c_i - c_j) / |c_i - c_j|
    
    Similarly for v = 1 - 2*max(|c_ik|).
    Let m = max(|c_ik|).
    v = 1 - 2m.
    dv/dm = -2.
    dm/dc: The max is achieved by some c_pq.
    If c_pq > 0, dm/dc_pq = 1, else -1.
    
    This is doable but complex to implement robustly.
    
    Let's try a simpler heuristic first: 
    1. Hex init.
    2. Run a few iterations of "push apart" and "pull in".
    
    Actually, let's just use `scipy.optimize.minimize` on the negative of the soft-min objective, and provide the gradient.
    """
    start_time = time.time()
    
    def objective_and_grad(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        
        # Calculate d_min
        # Vectorized distance calculation is memory heavy for N=200 (40k floats, ok).
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_dist_idx
        d_min = dists[i, j]
        
        # Calculate b_min
        max_abs = np.max(np.abs(pts))
        # Find the coordinate that achieves max_abs
        max_idx = np.unravel_index(np.argmax(np.abs(pts)), pts.shape)
        p, q = max_idx
        b_min = 1.0 - 2.0 * max_abs
        
        # Soft min
        t = 0.01 # Temperature
        # Avoid overflow/underflow
        # f = -t * log( exp(-u/t) + exp(-v/t) )
        # Let a = -u/t, b = -v/t
        # f = -t * log( exp(a) + exp(b) ) = -t * ( max(a,b) + log( exp(a-max) + exp(b-max) ) )
        a = -d_min / t
        b = -b_min / t
        m = max(a, b)
        log_sum = m + np.log(np.exp(a - m) + np.exp(b - m))
        val = -t * log_sum
        
        # Gradient
        # df/du = exp(-u/t) / (exp(-u/t) + exp(-v/t)) = exp(a) / (exp(a) + exp(b))
        # = 1 / (1 + exp(b-a)) = 1 / (1 + exp(-(b_min-d_min)/t))
        # Let's compute weights
        w_u = np.exp(a - m) / (np.exp(a - m) + np.exp(b - m))
        w_v = np.exp(b - m) / (np.exp(a - m) + np.exp(b - m))
        
        # d(val)/d(pts) = w_u * d(d_min)/d(pts) + w_v * d(b_min)/d(pts)
        # Note: val = -soft_min, so we are minimizing val, which means maximizing soft_min.
        # Wait, objective in minimize should be negative of what we want to maximize.
        # We want to maximize r = min(d_min, b_min)/2.
        # So we minimize -r.
        # Let's minimize -soft_min(d_min, b_min, t)/2.
        # So val_above is -soft_min. We need to minimize val_above / 2.
        # So gradient is 0.5 * (w_u * grad_d + w_v * grad_b)
        
        grad = np.zeros_like(pts)
        
        # Gradient of d_min
        # d(d_min)/dc_i = -(c_i - c_j) / d_min
        # d(d_min)/dc_j = (c_i - c_j) / d_min
        if d_min > 1e-12:
            diff = pts[i] - pts[j]
            grad[i] -= w_u * diff / d_min
            grad[j] += w_u * diff / d_min
        
        # Gradient of b_min
        # b_min = 1 - 2*max_abs
        # db_min/dmax_abs = -2
        # dmax_abs/dc_pq = sign(c_pq)
        if max_abs > 1e-12:
            sign = np.sign(pts[p, q])
            grad[p, q] += w_v * (-2.0) * sign
        
        return val / 2.0, (grad / 2.0).flatten()
    
    pts_flat = pts.flatten()
    
    # Use L-BFGS-B
    # We need to provide bounds? 
    # The constraints are handled by the objective (soft min penalizes boundary violations heavily if t is small, 
    # but actually if a point goes out, b_min becomes negative, and soft min becomes very negative, 
    # so -soft_min becomes very positive, so we avoid it.
    # However, if a point goes far out, d_min might be large, but b_min is very negative.
    # So the optimizer should keep them inside.
    # But to be safe, we can add bounds.
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    from scipy.optimize import minimize
    
    res = minimize(
        lambda x: objective_and_grad(x)[0],
        pts_flat,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    best_pts = res.x.reshape(-1, 2)
    best_score = _score(best_pts)
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_pts, best_score
        
    # Maybe do a second pass with smaller t?
    # For now, just return
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    # 1. Initialize with hex lattice
    pts = _hexagonal_init(N)
    current_score = _score(pts)
    
    # 2. Optimize
    try:
        pts, score = _optimize(pts, max_iters=1000, time_limit=time_limit - (time.time() - start_time))
        if score > current_score:
            current_score = score
    except Exception as e:
        # Fallback if optimization fails
        pass
        
    # 3. If time remains, maybe do a simple local search?
    # Let's check time
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5.0:
        # Simple random jitter local search
        best_pts = pts.copy()
        best_score = current_score
        step_size = 0.01
        while time.time() - start_time < time_limit - 5.0:
            # Pick a random point
            idx = np.random.randint(N)
            orig = best_pts[idx].copy()
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            delta = np.array([np.cos(angle), np.sin(angle)]) * step_size
            new_pt = best_pts[idx] + delta
            
            # Check bounds
            if np.any(np.abs(new_pt) > 0.5):
                continue
                
            best_pts[idx] = new_pt
            new_score = _score(best_pts)
            
            if new_score > best_score:
                best_score = new_score
                step_size *= 1.05 # Increase step slightly if successful
            else:
                best_pts[idx] = orig
                step_size *= 0.95 # Decrease step if failed
                if step_size < 1e-6:
                    break # Stuck in local optimum?
    
    return best_pts
# EVOLVE-BLOCK-END
