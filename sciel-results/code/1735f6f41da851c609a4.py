"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _make_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _feasible_grow(pos, radii, grow, pairs):
    """Check if we can grow all radii by 'grow' amount."""
    n = len(pos)
    new_radii = radii + grow
    for i in range(n):
        if new_radii[i] > 0.5:
            return False, new_radii
        if pos[i, 0] < new_radii[i] - 1e-9 or pos[i, 0] > 1 - new_radii[i] + 1e-9:
            return False, new_radii
        if pos[i, 1] < new_radii[i] - 1e-9 or pos[i, 1] > 1 - new_radii[i] + 1e-9:
            return False, new_radii
    for i, j in pairs:
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < new_radii[i] + new_radii[j] - 1e-9:
            return False, new_radii
    return True, new_radii


def _clamp_boundaries(pos, radii):
    for i in range(len(pos)):
        pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
        pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))


def _relax(pos, radii, pairs, max_iters=2000, time_budget=30.0, start_time=None):
    """Iterative overlap resolution with boundary clamping."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        _clamp_boundaries(pos, radii)
        
        changed = False
        for i, j in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                changed = True
                if dist < 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = (min_dist - dist) / 2
                    pos[i, 0] -= push * np.cos(angle)
                    pos[i, 1] -= push * np.sin(angle)
                    pos[j, 0] += push * np.cos(angle)
                    pos[j, 1] += push * np.sin(angle)
                else:
                    push = (min_dist - dist) / 2
                    nx = dx / dist
                    ny = dy / dist
                    pos[i, 0] -= push * nx
                    pos[i, 1] -= push * ny
                    pos[j, 0] += push * nx
                    pos[j, 1] += push * ny
        
        _clamp_boundaries(pos, radii)
        
        if not changed:
            break
    
    return pos, radii


def _grow_radii(pos, radii, pairs, time_budget=5.0, start_time=None):
    """Try to grow radii as much as possible."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    grow = 0.002
    max_grow_attempts = 500
    
    for _ in range(max_grow_attempts):
        if time.time() - start_time > time_budget:
            break
        
        feasible, new_radii = _feasible_grow(pos, radii, grow, pairs)
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=200,
                                  time_budget=min(1.0, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
        else:
            lo, hi = 0.0, grow
            for _ in range(20):
                mid = (lo + hi) / 2
                feasible_mid, _ = _feasible_grow(pos, radii, mid, pairs)
                if feasible_mid:
                    lo = mid
                else:
                    hi = mid
            if lo > 1e-6:
                _, new_radii = _feasible_grow(pos, radii, lo, pairs)
                radii = new_radii
                pos, radii = _relax(pos, radii, pairs, max_iters=200,
                                      time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                      start_time=start_time)
            grow *= 0.7
            if grow < 1e-6:
                break
    
    return pos, radii


def _local_search(pos, radii, pairs, time_budget=20.0, start_time=None):
    """Local search with random moves and growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    while time.time() - start_time < time_budget:
        feasible, new_radii = _feasible_grow(pos, radii, 0.001, pairs)
        if feasible:
            radii = new_radii
            pos, radii = _relax(pos, radii, pairs, max_iters=100,
                                  time_budget=min(0.5, time_budget - (time.time() - start_time)),
                                  start_time=start_time)
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
            continue
        
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            idx = rng.randint(n)
            old_pos = pos[idx].copy()
            step_size = 0.01
            pos[idx] += rng.normal(0, step_size, size=2)
            pos[idx] = np.clip(pos[idx], radii[idx], 1 - radii[idx])
            
            valid = True
            if pos[idx, 0] < radii[idx] - 1e-9 or pos[idx, 0] > 1 - radii[idx] + 1e-9:
                valid = False
            if pos[idx, 1] < radii[idx] - 1e-9 or pos[idx, 1] > 1 - radii[idx] + 1e-9:
                valid = False
            if valid:
                for j in range(n):
                    if j == idx:
                        continue
                    dx = pos[j, 0] - pos[idx, 0]
                    dy = pos[j, 1] - pos[idx, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[idx] + radii[j] - 1e-9:
                        valid = False
                        break
            
            if valid:
                feasible2, new_r = _feasible_grow(pos, radii, 0.0005, pairs)
                if feasible2:
                    radii = new_r
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
            else:
                pos[idx] = old_pos
    
    return best_pos, best_radii


def _softplus_penalty(pos, radii, pairs, beta=100.0):
    """
    Smooth log-sum-exp (soft-plus) surrogate for the penalty.
    Returns (penalty_value, gradient w.r.t. flat vector [x0,y0,r0, x1,y1,r1, ...]).
    """
    n = len(pos)
    flat = np.zeros(3 * n)
    for i in range(n):
        flat[3*i] = pos[i, 0]
        flat[3*i+1] = pos[i, 1]
        flat[3*i+2] = radii[i]
    
    penalty = 0.0
    grad = np.zeros(3 * n)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    for i, j in pairs:
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[i] + radii[j]
        viol = min_dist - dist
        if viol > 0:
            # Soft-plus: log(1 + exp(beta * viol)) / beta
            # For large beta*viol, approx viol
            bv = beta * viol
            if bv > 50:
                s = viol
                ds_dviol = 1.0
            else:
                s = np.log1p(np.exp(bv)) / beta
                ds_dviol = 1.0 / (1.0 + np.exp(-bv))
            penalty += s
            # d(s)/d(pos_i) = ds_dviol * d(viol)/d(pos_i)
            # viol = min_dist - dist
            # d(dist)/d(pos_i_x) = -(dx)/dist
            # d(viol)/d(pos_i_x) = dx/dist
            if dist > 1e-12:
                nx = dx / dist
                ny = dy / dist
            else:
                nx, ny = 1.0, 0.0
            # grad w.r.t. pos_i: ds_dviol * (nx, ny) for x,y of i... wait
            # d(viol)/d(pos_i_x) = d(min_dist - dist)/d(pos_i_x) = -d(dist)/d(pos_i_x) = -(-dx/dist) = dx/dist = nx
            # So grad[3*i] += ds_dviol * nx, grad[3*i+1] += ds_dviol * ny
            # d(viol)/d(pos_j_x) = -nx
            grad[3*i] += ds_dviol * nx
            grad[3*i+1] += ds_dviol * ny
            grad[3*j] -= ds_dviol * nx
            grad[3*j+1] -= ds_dviol * ny
            # d(viol)/d(r_i) = 1, d(viol)/d(r_j) = 1
            grad[3*i+2] += ds_dviol
            grad[3*j+2] += ds_dviol
    
    # Boundary constraints
    for i in range(n):
        r = radii[i]
        # x >= r
        viol = r - pos[i, 0]
        if viol > 0:
            bv = beta * viol
            if bv > 50:
                s = viol
                ds_dviol = 1.0
            else:
                s = np.log1p(np.exp(bv)) / beta
                ds_dviol = 1.0 / (1.0 + np.exp(-bv))
            penalty += s
            grad[3*i] -= ds_dviol  # d(viol)/d(x_i) = -1
            grad[3*i+2] += ds_dviol  # d(viol)/d(r_i) = 1
        # x <= 1 - r
        viol = pos[i, 0] - (1 - r)
        if viol > 0:
            bv = beta * viol
            if bv > 50:
                s = viol
                ds_dviol = 1.0
            else:
                s = np.log1p(np.exp(bv)) / beta
                ds_dviol = 1.0 / (1.0 + np.exp(-bv))
            penalty += s
            grad[3*i] += ds_dviol  # d(viol)/d(x_i) = 1
            grad[3*i+2] -= ds_dviol  # d(viol)/d(r_i) = -1
        # y >= r
        viol = r - pos[i, 1]
        if viol > 0:
            bv = beta * viol
            if bv > 50:
                s = viol
                ds_dviol = 1.0
            else:
                s = np.log1p(np.exp(bv)) / beta
                ds_dviol = 1.0 / (1.0 + np.exp(-bv))
            penalty += s
            grad[3*i+1] -= ds_dviol
            grad[3*i+2] += ds_dviol
        # y <= 1 - r
        viol = pos[i, 1] - (1 - r)
        if viol > 0:
            bv = beta * viol
            if bv > 50:
                s = viol
                ds_dviol = 1.0
            else:
                s = np.log1p(np.exp(bv)) / beta
                ds_dviol = 1.0 / (1.0 + np.exp(-bv))
            penalty += s
            grad[3*i+1] += ds_dviol
            grad[3*i+2] -= ds_dviol
    
    return penalty, grad


def _scipy_optimize(pos0, radii0, pairs, time_budget=10.0, start_time=None, beta=100.0):
    """
    Use scipy L-BFGS-B to minimize penalty - sum(radii) with soft-plus surrogate.
    """
    if start_time is None:
        start_time = time.time()
    n = len(pos0)
    
    def objective_and_grad(flat):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(flat)
        pos = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            pos[i, 0] = flat[3*i]
            pos[i, 1] = flat[3*i+1]
            radii[i] = flat[3*i+2]
        penalty, pgrad = _softplus_penalty(pos, radii, pairs, beta=beta)
        # We want to minimize: penalty - sum(radii)
        val = penalty - np.sum(radii)
        grad = pgrad.copy()
        for i in range(n):
            grad[3*i+2] -= 1.0  # d(-sum(radii))/d(r_i) = -1
        return val, grad
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pos0[i, 0]
        x0[3*i+1] = pos0[i, 1]
        x0[3*i+2] = radii0[i]
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.49))  # x
        bounds.append((0.01, 0.49))  # y
        bounds.append((0.01, 0.49))  # r
    
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                          jac=True, bounds=bounds,
                          options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-6,
                                   'maxls': 20})
    except Exception:
        return pos0.copy(), radii0.copy()
    
    if time.time() - start_time > time_budget:
        return pos0.copy(), radii0.copy()
    
    pos_new = np.zeros((n, 2))
    radii_new = np.zeros(n)
    for i in range(n):
        pos_new[i, 0] = result.x[3*i]
        pos_new[i, 1] = result.x[3*i+1]
        radii_new[i] = result.x[3*i+2]
    
    # Clamp
    _clamp_boundaries(pos_new, radii_new)
    
    return pos_new, radii_new


def _build_hex(n):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.17
            y = (row + 0.5) * 0.149
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    return pos, radii


def _build_grid(n):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=0):
    """Build a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii


def _build_corner(n):
    """Place circles at corners and edges for a structured start."""
    pos_list = []
    r_corner = 0.12
    pos_list.append([r_corner, r_corner])
    pos_list.append([1 - r_corner, r_corner])
    pos_list.append([r_corner, 1 - r_corner])
    pos_list.append([1 - r_corner, 1 - r_corner])
    
    r_edge = 0.1
    pos_list.append([0.5, r_edge])
    pos_list.append([0.5, 1 - r_edge])
    pos_list.append([r_edge, 0.5])
    pos_list.append([1 - r_edge, 0.5])
    
    remaining = n - len(pos_list)
    if remaining > 0:
        side = int(np.ceil(np.sqrt(remaining)))
        for j in range(side):
            for i in range(side):
                if len(pos_list) >= n:
                    break
                x = 0.25 + (i + 0.5) * 0.5 / side
                y = 0.25 + (j + 0.5) * 0.5 / side
                pos_list.append([x, y])
    
    pos = np.array(pos_list[:n])
    radii = np.full(n, 0.08)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    pairs = _make_pairs(N)
    
    # Multiple initial configurations
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))
    
    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))
    
    for seed in [42, 123, 456, 789, 1000]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    
    # Process each configuration
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        remaining = deadline - elapsed - 5
        
        # Phase 1: Relax overlaps
        time_alloc = min(5.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _relax(pos, radii, pairs, max_iters=2000,
                                 time_budget=time_alloc, start_time=start_time)
        
        # Phase 2: Grow radii
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(5.0, remaining * 0.2)
        if time_alloc > 1:
            pos, radii = _grow_radii(pos, radii, pairs, time_budget=time_alloc,
                                      start_time=start_time)
        
        # Phase 3: Scipy optimization with soft-plus penalty
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(15.0, remaining * 0.4)
        if time_alloc > 2:
            # Use increasing beta for progressive sharpening
            betas = [50.0, 100.0, 200.0, 500.0]
            for b in betas:
                if time.time() - start_time > deadline - 5:
                    break
                t_remaining = deadline - time.time() - 5
                if t_remaining < 2:
                    break
                pos, radii = _scipy_optimize(pos, radii, pairs,
                                              time_budget=min(3.0, t_remaining),
                                              start_time=start_time, beta=b)
                _clamp_boundaries(pos, radii)
        
        # Phase 4: Local search polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(10.0, remaining * 0.5)
        if time_alloc > 1:
            pos, radii = _local_search(pos, radii, pairs, time_budget=time_alloc,
                                        start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final polish if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        if remaining > 5:
            # Try scipy optimization on best with high beta
            backup_pos = best_pos.copy()
            backup_radii = best_radii.copy()
            pos2, radii2 = _scipy_optimize(best_pos, best_radii, pairs,
                                            time_budget=min(10.0, remaining * 0.6),
                                            start_time=start_time, beta=500.0)
            _clamp_boundaries(pos2, radii2)
            if _check_validity(pos2, radii2):
                new_score = np.sum(radii2)
                if new_score > best_score:
                    best_pos = pos2
                    best_radii = radii2
                    best_score = new_score
            
            # Try local search on best
            elapsed = time.time() - start_time
            remaining = deadline - elapsed - 3
            if remaining > 3:
                pos3, radii3 = _local_search(best_pos, best_radii, pairs,
                                              time_budget=min(5.0, remaining),
                                              start_time=start_time)
                if _check_validity(pos3, radii3):
                    new_score = np.sum(radii3)
                    if new_score > best_score:
                        best_pos = pos3
                        best_radii = radii3
                        best_score = new_score
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
