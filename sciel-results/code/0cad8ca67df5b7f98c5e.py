"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21

# EVOLVE-BLOCK-START

def _check_validity(xs, ys, rs, W, H, tol=1e-7):
    """Check if circles are valid: inside [0,W]x[0,H] and non-overlapping."""
    if np.any(xs < -tol) or np.any(xs > W + tol) or np.any(ys < -tol) or np.any(ys > H + tol):
        return False
    if np.any(xs + rs > W + tol) or np.any(xs - rs < -tol) or np.any(ys + rs > H + tol) or np.any(ys - rs < -tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d_sq = dx * dx + dy * dy
            min_d = rs[i] + rs[j]
            if d_sq < min_d * min_d - tol * tol:
                return False
    return True


def _make_grid_init(W, H, n):
    """Generate a grid initialization of n circles in W x H rectangle."""
    # Use a hex-like offset grid
    cols = max(1, int(np.ceil(np.sqrt(n * W / H))))
    rows = max(1, int(np.ceil(n / cols)))
    
    xs = np.zeros(n)
    ys = np.zeros(n)
    
    for i in range(n):
        c = i % cols
        r_idx = i // cols
        # Hex offset
        offset = 0.5 * W / cols if (r_idx % 2 == 1) else 0.0
        xs[i] = (c + 0.5) * W / cols + offset
        ys[i] = (r_idx + 0.5) * H / rows
        
    # Clip to valid range with small radius
    r_est = 0.05 * min(W, H)
    xs = np.clip(xs, r_est, W - r_est)
    ys = np.clip(ys, r_est, H - r_est)
    
    return xs, ys


def _relax_positions(xs, ys, rs, W, H, max_iter=500, tol=1e-8):
    """
    Relax positions to remove overlaps and keep circles inside.
    Uses simple gradient descent on overlap penalty.
    Returns (xs, ys, success)
    """
    xs = xs.copy()
    ys = ys.copy()
    
    for it in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx * dx + dy * dy
                d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
                min_d = rs[i] + rs[j]
                if d < min_d:
                    overlap = min_d - d
                    nx = dx / d
                    ny = dy / d
                    fx[i] += overlap * nx
                    fy[i] += overlap * ny
                    fx[j] -= overlap * nx
                    fy[j] -= overlap * ny
        
        # Wall repulsion (strong)
        for i in range(N):
            if xs[i] < rs[i]:
                fx[i] += 10.0 * (rs[i] - xs[i])
            if xs[i] > W - rs[i]:
                fx[i] -= 10.0 * (xs[i] - (W - rs[i]))
            if ys[i] < rs[i]:
                fy[i] += 10.0 * (rs[i] - ys[i])
            if ys[i] > H - rs[i]:
                fy[i] -= 10.0 * (ys[i] - (H - rs[i]))
        
        # Update with adaptive learning rate
        lr = 0.5 / (1.0 + it * 0.01)
        xs += lr * fx
        ys += lr * fy
        
        # Hard clip
        xs = np.clip(xs, 0, W)
        ys = np.clip(ys, 0, H)
        
        # Check convergence
        if np.max(np.abs(fx)) < tol and np.max(np.abs(fy)) < tol:
            break
    
    # Final validity check
    valid = _check_validity(xs, ys, rs, W, H)
    return xs, ys, valid


def _try_packing(d, max_time=10.0):
    """
    Try to pack N circles in rectangle W=1+d, H=1-d.
    Returns (xs, ys, rs, W, H, success)
    """
    start_time = time.time()
    W = 1.0 + d
    H = 1.0 - d
    if W <= 0.1 or H <= 0.1:
        return None, None, None, W, H, False
    
    # Estimate max possible equal radius
    area_rect = W * H
    # Dense packing density ~0.907 for circles in plane
    r_max_est = np.sqrt(0.907 * area_rect / (N * np.pi))
    
    # Binary search on radius with relaxation
    r_low = 0.0
    r_high = r_max_est * 0.99
    
    best_xs, best_ys, best_r = None, None, 0.0
    
    # First, get a valid configuration at small radius
    xs_init, ys_init = _make_grid_init(W, H, N)
    r_small = r_max_est * 0.1
    rs = np.full(N, r_small)
    xs, ys, valid = _relax_positions(xs_init, ys_init, rs, W, H, max_iter=200)
    if not valid:
        # Try again with smaller
        r_small = r_max_est * 0.05
        rs = np.full(N, r_small)
        xs, ys, valid = _relax_positions(xs_init, ys_init, rs, W, H, max_iter=300)
    
    if not valid:
        return None, None, None, W, H, False
    
    best_xs, best_ys, best_r = xs.copy(), ys.copy(), r_small
    
    # Now binary search / iterative increase
    for it in range(60):
        if time.time() - start_time > max_time:
            break
        
        # Try to increase radius
        r_new = best_r * 1.03 + r_max_est * 0.001
        if r_new >= r_high:
            r_new = r_high
        
        rs = np.full(N, r_new)
        xs_try, ys_try, valid = _relax_positions(best_xs, best_ys, rs, W, H, max_iter=400)
        
        if valid:
            best_xs, best_ys, best_r = xs_try, ys_try, r_new
        else:
            # Try slightly smaller
            r_mid = best_r + (r_new - best_r) * 0.5
            if r_mid > best_r * 1.005:
                rs = np.full(N, r_mid)
                xs_try, ys_try, valid = _relax_positions(best_xs, best_ys, rs, W, H, max_iter=300)
                if valid:
                    best_xs, best_ys, best_r = xs_try, ys_try, r_mid
            else:
                break
    
    if _check_validity(best_xs, best_ys, np.full(N, best_r), W, H):
        return best_xs, best_ys, np.full(N, best_r), W, H, True
    return None, None, None, W, H, False


def _lbbfgs_optimize(xs_init, ys_init, W, H, max_time=30.0):
    """
    Use L-BFGS-B to optimize sum of radii with penalty for violations.
    Variables: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
    """
    start_time = time.time()
    
    def objective(v):
        if time.time() - start_time > max_time:
            return 1e10
        xs_v = v[0::3]
        ys_v = v[1::3]
        rs_v = v[2::3]
        
        # Negative sum of radii (we minimize)
        obj = -np.sum(rs_v)
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs_v[i] - xs_v[j]
                dy = ys_v[i] - ys_v[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = rs_v[i] + rs_v[j]
                if d < min_d:
                    obj += 100.0 * (min_d - d) ** 2
        
        # Penalty for walls
        for i in range(N):
            if xs_v[i] < rs_v[i]:
                obj += 100.0 * (rs_v[i] - xs_v[i]) ** 2
            if xs_v[i] > W - rs_v[i]:
                obj += 100.0 * (xs_v[i] - (W - rs_v[i])) ** 2
            if ys_v[i] < rs_v[i]:
                obj += 100.0 * (rs_v[i] - ys_v[i]) ** 2
            if ys_v[i] > H - rs_v[i]:
                obj += 100.0 * (ys_v[i] - (H - rs_v[i])) ** 2
        
        return obj
    
    # Initial guess
    r0 = 0.05 * min(W, H)
    xs0, ys0 = _make_grid_init(W, H, N)
    v0 = np.zeros(3 * N)
    v0[0::3] = xs0
    v0[1::3] = ys0
    v0[2::3] = r0
    
    # Bounds: x in [0, W], y in [0, H], r in [0, r_max]
    r_max = 0.5 * min(W, H)
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
        bounds.append((0, r_max))
    
    try:
        result = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10})
        
        if result.success and time.time() - start_time < max_time:
            xs_opt = result.x[0::3]
            ys_opt = result.x[1::3]
            rs_opt = result.x[2::3]
            
            if _check_validity(xs_opt, ys_opt, rs_opt, W, H):
                return xs_opt, ys_opt, rs_opt, True
    except Exception:
        pass
    
    return None, None, None, False


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_sum_r = 0.0
    best_config = None
    best_rect = None
    
    # Try multiple aspect ratios
    d_values = [-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45]
    
    for d in d_values:
        if time.time() > deadline - 25:
            break
        
        xs, ys, rs, W, H, success = _try_packing(d, max_time=8.0)
        
        if success:
            total_r = np.sum(rs)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.column_stack((xs, ys, rs))
                best_rect = (W, H)
    
    # If time remains, try L-BFGS-B on the best aspect ratio
    if best_rect is not None and time.time() < deadline - 40:
        W, H = best_rect
        xs_init, ys_init = best_config[:, 0].copy(), best_config[:, 1].copy()
        
        xs_opt, ys_opt, rs_opt, success = _lbbfgs_optimize(xs_init, ys_init, W, H, max_time=30.0)
        
        if success:
            total_r = np.sum(rs_opt)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.column_stack((xs_opt, ys_opt, rs_opt))
                best_rect = (W, H)
    
    # Try a few more random initializations with L-BFGS-B if time permits
    if time.time() < deadline - 50:
        for d in [0.0, -0.2, 0.2]:
            if time.time() > deadline - 45:
                break
            W = 1.0 + d
            H = 1.0 - d
            if W <= 0.1 or H <= 0.1:
                continue
            
            np.random.seed(42)
            for trial in range(3):
                if time.time() > deadline - 45:
                    break
                
                # Random initialization
                r0 = 0.04 * min(W, H)
                xs_rand = np.random.uniform(r0, W - r0, N)
                ys_rand = np.random.uniform(r0, H - r0, N)
                
                xs_opt, ys_opt, rs_opt, success = _lbbfgs_optimize(xs_rand, ys_rand, W, H, max_time=12.0)
                
                if success:
                    total_r = np.sum(rs_opt)
                    if total_r > best_sum_r:
                        best_sum_r = total_r
                        best_config = np.column_stack((xs_opt, ys_opt, rs_opt))
                        best_rect = (W, H)
    
    # If we found a valid configuration, return it
    if best_config is not None:
        # Final validation
        xs = best_config[:, 0]
        ys = best_config[:, 1]
        rs = best_config[:, 2]
        W, H = best_rect
        
        if _check_validity(xs, ys, rs, W, H):
            return best_config
        else:
            # Try to repair
            xs_rep, ys_rep, valid = _relax_positions(xs, ys, rs, W, H, max_iter=200)
            if valid and _check_validity(xs_rep, ys_rep, rs, W, H):
                return np.column_stack((xs_rep, ys_rep, rs))
    
    # Fallback: simple grid in unit square
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
