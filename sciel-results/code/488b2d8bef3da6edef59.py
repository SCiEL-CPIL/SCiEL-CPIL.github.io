import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 140.0  # Safety margin before 150s limit

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _smooth_max(x, eps=1e-4):
    """Smooth approximation of max(x, 0)."""
    return 0.5 * (x + np.sqrt(x**2 + eps**2))

def _packing_violation_and_grad(x, r_target, eps=1e-4):
    """
    Objective: Sum of squared smooth-max violations.
    x: flattened array [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y]
    r_target: fixed radius to test.
    Returns (objective_value, gradient)
    """
    c = x.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    # Violation: 2*r - dist
    viol_pair = 2.0 * r_target - pair_dist
    sm_pair = _smooth_max(viol_pair, eps)
    obj_pair = np.sum(sm_pair ** 2)
    
    # Wall distances
    d_left = c[:, 0] + HALF
    d_right = HALF - c[:, 0]
    d_bottom = c[:, 1] + HALF
    d_top = HALF - c[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    viol_wall = r_target - wall_dist
    sm_wall = _smooth_max(viol_wall, eps)
    obj_wall = np.sum(sm_wall ** 2)
    
    obj = obj_pair + obj_wall
    
    # Gradients
    # d(sm_max(x)^2)/dx = 2 * sm_max(x) * d(sm_max(x))/dx
    # d(sm_max(x))/dx = 0.5 * (1 + x / sqrt(x^2 + eps^2))
    
    grad_c = np.zeros_like(c)
    
    # Pair gradients
    d_sm_dx = 0.5 * (1.0 + viol_pair / np.sqrt(viol_pair**2 + eps**2))
    d_obj_d_viol = 2.0 * sm_pair * d_sm_dx
    
    # Reconstruct full matrix indexing for vectorization? 
    # Loop is fine for N=50 (1225 pairs)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        d = pair_dist[k]
        if d < 1e-9: continue
        vec = (c[i] - c[j]) / d
        # Gradient contribution to i: -d_obj_d_viol * vec
        # Gradient contribution to j: +d_obj_d_viol * vec
        grad_c[i] -= d_obj_d_viol[k] * vec
        grad_c[j] += d_obj_d_viol[k] * vec
        
    # Wall gradients
    d_sm_wall_dx = 0.5 * (1.0 + viol_wall / np.sqrt(viol_wall**2 + eps**2))
    d_obj_d_viol_wall = 2.0 * sm_wall * d_sm_wall_dx
    
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = d_obj_d_viol_wall[i]
        
        # Use a small tolerance to handle cases where multiple walls are equidistant
        tol = 1e-8
        if abs(dl - wmin) < tol:
            # w = x + 0.5, dw/dx = 1
            # d viol/dx = -1
            # d obj/dx = s * (-1)
            grad_c[i, 0] -= s
        if abs(dr - wmin) < tol:
            # w = 0.5 - x, dw/dx = -1
            # d viol/dx = -(-1) = 1
            # d obj/dx = s * (1)
            grad_c[i, 0] += s
        if abs(db - wmin) < tol:
            # w = y + 0.5, dw/dy = 1
            # d viol/dy = -1
            # d obj/dy = s * (-1)
            grad_c[i, 1] -= s
        if abs(dt - wmin) < tol:
            # w = 0.5 - y, dw/dy = -1
            # d viol/dy = 1
            # d obj/dy = s * (1)
            grad_c[i, 1] += s

    grad = grad_c.ravel()
    return float(obj), grad

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def _optimize_for_radius(r_target, P_init, deadline):
    """
    Attempt to find a valid configuration for the given radius r_target.
    Returns (success, best_pts, min_violation)
    """
    x0 = P_init.ravel()
    
    def obj_func(x):
        return _packing_violation_and_grad(x, r_target)[0]
    
    def jac_func(x):
        return _packing_violation_and_grad(x, r_target)[1]
        
    try:
        res = minimize(
            obj_func,
            x0,
            jac=jac_func,
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50, 'disp': False}
        )
        
        x_opt = res.x
        c_opt = x_opt.reshape(N, 2)
        c_opt = np.clip(c_opt, -HALF, HALF)
        
        # Check final violation
        viol, _ = _packing_violation_and_grad(c_opt.ravel(), r_target)
        
        # Also check actual feasible radius
        r_actual = _compute_r(c_opt)
        
        # If actual radius is >= r_target - epsilon, consider it a success
        # We use a slightly looser epsilon to account for numerical noise in r_target check
        if r_actual >= r_target - 1e-7:
            return True, c_opt, viol
        else:
            return False, c_opt, viol
            
    except Exception as e:
        return False, P_init, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    # Initializations to try
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.005),
        _grid_init(rng, jitter=0.005),
        _random_init(rng),
    ]
    
    # Binary search for radius
    # Known best is ~0.071377
    # We will search from 0.0710 to 0.0720
    low = 0.0710
    high = 0.0720
    
    # First, ensure we have a valid starting point at a low radius
    # Try to find a valid config for 'low'
    valid_config = None
    for init in inits:
        if time.time() > deadline - 30: break
        success, pts, viol = _optimize_for_radius(low, init, deadline)
        if success:
            valid_config = pts
            r_check = _compute_r(pts)
            if r_check > best_r:
                best_r = r_check
                best_pts = pts
            break
            
    if valid_config is None:
        # If we couldn't even get 'low', return best effort from init
        for init in inits:
            r_check = _compute_r(init)
            if r_check > best_r:
                best_r = r_check
                best_pts = init
        return best_pts if best_pts is not None else _hex_init(rng)

    # Binary Search
    # We will run for a fixed number of iterations or until time runs out
    # 20 iterations gives precision of 0.001 / 2^20 ~ 1e-6, which is plenty
    max_iters = 25
    for _ in range(max_iters):
        if time.time() > deadline - 15:
            break
            
        mid = (low + high) / 2.0
        
        # Try to find a valid config for mid
        # Use the current valid_config as the primary init (warm start)
        # Also try a few jittered versions if time permits
        success = False
        new_config = None
        
        # Primary attempt with warm start
        success, pts, viol = _optimize_for_radius(mid, valid_config, deadline)
        if success:
            new_config = pts
            success = True
        else:
            # If warm start failed, try a few other inits quickly?
            # To save time, we only do this if we have plenty of time left
            if time.time() < deadline - 60:
                for init in inits[:2]: # Hex and Grid
                    if time.time() > deadline - 15: break
                    s, p, v = _optimize_for_radius(mid, init, deadline)
                    if s:
                        new_config = p
                        success = True
                        break
        
        if success:
            # mid is feasible
            low = mid
            r_check = _compute_r(new_config)
            if r_check > best_r:
                best_r = r_check
                best_pts = new_config
            valid_config = new_config
        else:
            # mid is infeasible
            high = mid
            
    # Final Polish
    # Try to squeeze out a bit more radius from the best found config
    if best_pts is not None and time.time() < deadline - 2:
        # Try slightly higher radius
        target_r = best_r + 1e-5
        s, pts, v = _optimize_for_radius(target_r, best_pts, deadline)
        if s:
            r_check = _compute_r(pts)
            if r_check > best_r:
                best_r = r_check
                best_pts = pts
        
        # Also try a generic polish on best_pts with its own radius
        # This might help if the binary search stopped early
        target_r = best_r
        s, pts, v = _optimize_for_radius(target_r, best_pts, deadline)
        if s:
            r_check = _compute_r(pts)
            if r_check > best_r:
                best_r = r_check
                best_pts = pts

    return best_pts if best_pts is not None else _hex_init(rng)
