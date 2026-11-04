"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _smooth_objective(v, n, beta=50.0):
    """
    Smooth objective: -sum(r) + penalty for violations.
    v = [x_1..x_n, y_1..y_n, r_1..r_n]
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    # Wall violations: r_i > min(x_i, 1-x_i, y_i, 1-y_i)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = np.maximum(0.0, r - wall_min)
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)
    
    # Pairwise violations: dist_ij < r_i + r_j
    # Vectorized: compute all pairwise distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = np.maximum(0.0, r_sum - dist)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, 1)
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol[iu]) / beta)
    
    return -np.sum(r) + 100.0 * (wall_pen + pair_pen)


def _smooth_grad(v, n, beta=50.0):
    """
    Gradient of the smooth objective.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    grad = np.zeros(3 * n)
    
    # Wall penalty gradient
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    # Softplus derivative: sigmoid(beta * viol)
    wall_sig = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    
    # d(wall_pen)/d(r_i) = wall_sig_i
    grad[2*n:3*n] += 100.0 * wall_sig
    
    # d(wall_pen)/d(x_i): if x_i is the min component, -wall_sig_i
    # Simplified: assume x_i or 1-x_i or y_i or 1-y_i is the min
    # For gradient, we approximate by checking which constraint is active
    # More robust: use subgradient
    for i in range(n):
        if wall_viol[i] > 0:
            # Find which wall is active
            candidates = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
            min_idx = np.argmin(candidates)
            if min_idx == 0:
                grad[i] += -100.0 * wall_sig[i]  # d(x_i)/dx_i = 1, d(x_i)/dr_i = -1 already handled
            elif min_idx == 1:
                grad[i] += 100.0 * wall_sig[i]   # d(1-x_i)/dx_i = -1
            elif min_idx == 2:
                grad[n + i] += -100.0 * wall_sig[i]
            elif min_idx == 3:
                grad[n + i] += 100.0 * wall_sig[i]
    
    # Pairwise penalty gradient
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    viol = r_sum - dist
    
    # Softplus derivative
    sig = 1.0 / (1.0 + np.exp(-beta * viol))
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    sig_iu = sig[iu]
    viol_iu = viol[iu]
    
    # d(pair_pen)/d(r_i) = sum_j sig_ij
    for i in range(n):
        mask = (iu[0] == i) | (iu[1] == i)
        grad[2*n + i] += 100.0 * np.sum(sig_iu[mask])
    
    # d(pair_pen)/d(x_i) = sum_j sig_ij * (x_i - x_j) / dist_ij
    for i in range(n):
        mask_i = (iu[0] == i)
        mask_j = (iu[1] == i)
        
        # As i in first index
        if np.any(mask_i):
            js = iu[1][mask_i]
            d = dist[iu[0][mask_i], iu[1][mask_i]]
            d = np.maximum(d, 1e-12)
            grad[i] += 100.0 * np.sum(sig_iu[mask_i] * (x[i] - x[js]) / d)
            grad[n + i] += 100.0 * np.sum(sig_iu[mask_i] * (y[i] - y[js]) / d)
        
        # As j in second index
        if np.any(mask_j):
            is_ = iu[0][mask_j]
            d = dist[iu[0][mask_j], iu[1][mask_j]]
            d = np.maximum(d, 1e-12)
            grad[i] += 100.0 * np.sum(sig_iu[mask_j] * (x[i] - x[is_]) / d * -1)
            grad[n + i] += 100.0 * np.sum(sig_iu[mask_j] * (y[i] - y[is_]) / d * -1)
    
    # Gradient of -sum(r)
    grad[2*n:3*n] -= 1.0
    
    return grad


def _gradient_polish(xy, r, deadline, beta=50.0):
    """
    Gradient-based polish using L-BFGS-B with vectorized gradient.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))
    
    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, beta)
    
    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n, beta)
    
    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9,
                                   'maxfun': 20000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        
        # Clip to valid range
        for i in range(n):
            new_r[i] = min(new_r[i], new_xy[i, 0], 1.0 - new_xy[i, 0], new_xy[i, 1], 1.0 - new_xy[i, 1])
            new_r[i] = max(new_r[i], 0.001)
            new_xy[i, 0] = np.clip(new_xy[i, 0], new_r[i], 1.0 - new_r[i])
            new_xy[i, 1] = np.clip(new_xy[i, 1], new_r[i], 1.0 - new_r[i])
        
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
    
    return xy, r


def _sa_packing(n, deadline, initial_xy=None, initial_r=None, T0=0.01, T_end=0.0001, alpha=0.999):
    """
    Simulated annealing for circle packing.
    """
    rng = np.random.RandomState(42)
    
    if initial_xy is not None:
        xy = initial_xy.copy()
        r = initial_r.copy()
    else:
        # Random initialization
        xy = rng.uniform(0.1, 0.9, (n, 2))
        r = rng.uniform(0.02, 0.05, n)
    
    T = T0
    best_xy = xy.copy()
    best_r = r.copy()
    best_total = 0.0
    
    # Initial evaluation
    total, valid = _eval_packing(xy, r)
    if valid:
        best_total = total
    else:
        best_total = -np.sum(r)  # Penalize invalid
    
    step_size = 0.05
    
    while time.time() < deadline:
        # Pick a random circle
        i = rng.randint(n)
        
        # Random perturbation
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        dr = rng.normal(0, step_size * 0.5)
        
        new_xy = xy.copy()
        new_r = r.copy()
        new_xy[i, 0] += dx
        new_xy[i, 1] += dy
        new_r[i] += dr
        
        # Keep in bounds
        new_xy[i, 0] = np.clip(new_xy[i, 0], new_r[i], 1.0 - new_r[i])
        new_xy[i, 1] = np.clip(new_xy[i, 1], new_r[i], 1.0 - new_r[i])
        new_r[i] = np.clip(new_r[i], 0.001, 0.5)
        
        new_total, new_valid = _eval_packing(new_xy, new_r)
        if not new_valid:
            new_total = -np.sum(new_r)
        
        delta = new_total - best_total
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / T):
            xy = new_xy
            r = new_r
            if new_total > best_total:
                best_total = new_total
                best_xy = xy.copy()
                best_r = r.copy()
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
        
        # Occasionally restart with larger step
        if rng.random() < 0.01:
            step_size = rng.uniform(0.02, 0.1)
    
    return best_xy, best_r, best_total


def _best_fit_local_search(xy, r, deadline, grid_size=50):
    """
    Local search: iteratively move each circle to best position/radius.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
    
    max_sweeps = 5
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]
            
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()
            
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]
            
            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total, best_valid


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n):
    """Random initialization."""
    rng = np.random.RandomState(42)
    xy = rng.uniform(0.1, 0.9, (n, 2))
    r = rng.uniform(0.02, 0.05, n)
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random init + local search
    if time.time() < deadline - 30:
        xy, r = _random_init(N)
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=50)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Simulated annealing
    if time.time() < deadline - 40:
        sa_xy, sa_r, sa_total = _sa_packing(N, deadline - 20)
        if sa_total > best_total:
            # Refine with local search
            out_xy, out_r, total, valid = _best_fit_local_search(sa_xy, sa_r, deadline - 10, grid_size=50)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r

    # Phase 4: Gradient polish on best result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 25:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline - 10, beta=50.0)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

        # Phase 5: Second local search pass
        if best_valid and best_xy is not None and time.time() < deadline - 15:
            ls2_xy, ls2_r, ls2_total, ls2_valid = _best_fit_local_search(
                best_xy.copy(), best_r.copy(), deadline, grid_size=80)
            if ls2_valid and ls2_total > best_total + 1e-10:
                best_total = ls2_total
                best_xy = ls2_xy
                best_r = ls2_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
