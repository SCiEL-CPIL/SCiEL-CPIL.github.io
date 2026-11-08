"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    # Boundary checks
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Pairwise overlap check (vectorized)
    ii, jj = np.triu_indices(n, k=1)
    dx = pos[ii, 0] - pos[jj, 0]
    dy = pos[ii, 1] - pos[jj, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < radii[ii] + radii[jj] - tol):
        return False
    return True


def _build_grid(n, jitter=0.0):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.45 / side
    radii = np.full(n, r)
    return pos, radii


def _build_hex(n, jitter=0.0):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.55 / cols
            y = (row + 0.5) * 0.5 / (int(np.ceil(n / cols)) + 1)
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.22
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=None):
    """Build a random initial configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.08)
    return pos, radii


def _penalty_objective(x, n, beta=50.0, lam=100.0):
    """
    Penalty-based objective for circle packing.
    x: flattened array [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    Returns: -sum(r) + lam * sum(softplus(violations))
    """
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Main term: -sum(radii)
    obj = -np.sum(radii)
    
    # Boundary violations: x < r, x > 1-r, y < r, y > 1-r
    # softplus(v) where v > 0 means violation
    b_left = radii - pos[:, 0]
    b_right = pos[:, 0] + radii - 1.0
    b_bottom = radii - pos[:, 1]
    b_top = pos[:, 1] + radii - 1.0
    
    # Pairwise overlap violations: r_i + r_j - dist_ij
    ii, jj = np.triu_indices(n, k=1)
    dx = pos[ii, 0] - pos[jj, 0]
    dy = pos[ii, 1] - pos[jj, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    overlap = radii[ii] + radii[jj] - dist
    
    # Softplus for smooth max(0, v)
    def softplus(v):
        v = np.clip(v, -50, 50)
        return np.log1p(np.exp(beta * v)) / beta
    
    penalty = (np.sum(softplus(b_left)) + np.sum(softplus(b_right)) +
               np.sum(softplus(b_bottom)) + np.sum(softplus(b_top)) +
               np.sum(softplus(overlap)))
    
    return obj + lam * penalty


def _penalty_gradient(x, n, beta=50.0, lam=100.0):
    """
    Gradient of penalty objective.
    """
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    grad = np.zeros_like(x)
    
    # d(-sum(r))/dr = -1
    grad[2*n:] = -1.0
    
    # Boundary gradients
    def softplus_grad(v):
        v = np.clip(v, -50, 50)
        return 1.0 / (1.0 + np.exp(-beta * v))
    
    b_left = radii - pos[:, 0]
    b_right = pos[:, 0] + radii - 1.0
    b_bottom = radii - pos[:, 1]
    b_top = pos[:, 1] + radii - 1.0
    
    # Gradients for boundaries
    gl = softplus_grad(b_left)
    gr = softplus_grad(b_right)
    gb = softplus_grad(b_bottom)
    gt = softplus_grad(b_top)
    
    # d(b_left)/dx_i = -1, d(b_left)/dr_i = 1
    grad[:n, 0] += -lam * gl
    grad[n:2*n, 0] += lam * gr  # d(b_right)/dx_i = 1
    grad[2*n:] += lam * (gl + gr + gb + gt)
    
    grad[:n, 1] += -lam * gb
    grad[n:2*n, 1] += lam * gt
    
    # Pairwise overlap gradients
    ii, jj = np.triu_indices(n, k=1)
    dx = pos[ii, 0] - pos[jj, 0]
    dy = pos[ii, 1] - pos[jj, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    overlap = radii[ii] + radii[jj] - dist
    
    go = softplus_grad(overlap)
    
    # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
    # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        if dist[k] > 1e-10:
            nx = dx[k] / dist[k]
            ny = dy[k] / dist[k]
        else:
            nx, ny = 0.0, 0.0
        
        g = lam * go[k]
        grad[i, 0] += -g * nx
        grad[j, 0] += g * nx
        grad[i, 1] += -g * ny
        grad[j, 1] += g * ny
        grad[2*n + i] += g
        grad[2*n + j] += g
    
    return grad


def _scipy_optimize(pos, radii, time_budget=30.0, start_time=None):
    """Optimize using L-BFGS-B with penalty method."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    
    # Flatten: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds: x in [0, 1], y in [0, 1], r in [0.01, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.01, 0.5))  # r
    
    def objective(x):
        return _penalty_objective(x, n, beta=50.0, lam=100.0)
    
    def gradient(x):
        return _penalty_gradient(x, n, beta=50.0, lam=100.0)
    
    # Track best valid solution
    best_x = x0.copy()
    best_score = np.sum(radii)
    
    def callback(xk):
        nonlocal best_x, best_score
        if time.time() - start_time > time_budget:
            raise RuntimeError("Time budget exceeded")
        
        # Check if this is a valid configuration
        pos_k = xk[:2*n].reshape(n, 2)
        radii_k = xk[2*n:]
        
        if _check_validity(pos_k, radii_k):
            score = np.sum(radii_k)
            if score > best_score:
                best_score = score
                best_x = xk.copy()
    
    try:
        result = minimize(
            objective, x0, jac=gradient, method='L-BFGS-B',
            bounds=bounds, callback=callback,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                     'maxfun': 2000}
        )
    except (RuntimeError, Exception):
        pass
    
    # Use best valid solution found
    pos_opt = best_x[:2*n].reshape(n, 2)
    radii_opt = best_x[2*n:]
    
    # Final validation
    if _check_validity(pos_opt, radii_opt):
        return pos_opt, radii_opt
    
    # Fallback to original if optimization failed
    return pos, radii


def _relax_and_grow(pos, radii, max_iters=5000, time_budget=30.0, start_time=None):
    """Physics-based relaxation with simultaneous radius growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    # Precompute pair indices
    ii, jj = np.triu_indices(n, k=1)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Check boundary violations (vectorized)
        pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
        pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)
        
        # Resolve overlaps (vectorized)
        dx = pos[jj, 0] - pos[ii, 0]
        dy = pos[jj, 1] - pos[ii, 1]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        min_dist = radii[ii] + radii[jj]
        overlap = min_dist - dist
        
        mask = overlap > 0
        if np.any(mask):
            push = overlap[mask] / 2
            nx = dx[mask] / dist[mask]
            ny = dy[mask] / dist[mask]
            
            ii_m = ii[mask]
            jj_m = jj[mask]
            
            pos[ii_m, 0] -= push * nx
            pos[ii_m, 1] -= push * ny
            pos[jj_m, 0] += push * nx
            pos[jj_m, 1] += push * ny
        
        # Re-check boundaries
        pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
        pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)
        
        # Try to grow radii
        if it % 50 == 0:
            grow_amount = 0.001
            new_radii = radii + grow_amount
            # Check if growth is feasible
            feasible = True
            if np.any(new_radii > 0.5) or np.any(new_radii < 0.01):
                feasible = False
            elif np.any(pos[:, 0] < new_radii) or np.any(pos[:, 0] > 1 - new_radii):
                feasible = False
            elif np.any(pos[:, 1] < new_radii) or np.any(pos[:, 1] > 1 - new_radii):
                feasible = False
            else:
                dx = pos[ii, 0] - pos[jj, 0]
                dy = pos[ii, 1] - pos[jj, 1]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                if np.any(dist < new_radii[ii] + new_radii[jj] - 1e-9):
                    feasible = False
            
            if feasible:
                radii = new_radii
    
    return pos, radii


def _local_search(pos, radii, time_budget=10.0, start_time=None):
    """Local search: try small moves to improve packing."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    ii, jj = np.triu_indices(n, k=1)
    
    while time.time() - start_time < time_budget:
        # Try to grow all radii slightly
        grow = 0.0005
        new_radii = radii + grow
        feasible = True
        if np.any(new_radii > 0.5):
            feasible = False
        elif np.any(pos[:, 0] < new_radii - 1e-9) or np.any(pos[:, 0] > 1 - new_radii + 1e-9):
            feasible = False
        elif np.any(pos[:, 1] < new_radii - 1e-9) or np.any(pos[:, 1] > 1 - new_radii + 1e-9):
            feasible = False
        else:
            dx = pos[ii, 0] - pos[jj, 0]
            dy = pos[ii, 1] - pos[jj, 1]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            if np.any(dist < new_radii[ii] + new_radii[jj] - 1e-9):
                feasible = False
        
        if feasible:
            radii = new_radii
            score = np.sum(radii)
            if score > best_score:
                best_score = score
            continue
        
        # Try small random moves
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            idx = np.random.randint(n)
            old_pos = pos[idx].copy()
            pos[idx] += np.random.normal(0, 0.005, size=2)
            pos[idx] = np.clip(pos[idx], 0, 1)
            
            # Check validity
            valid = True
            if pos[idx, 0] < radii[idx] - 1e-9 or pos[idx, 0] > 1 - radii[idx] + 1e-9:
                valid = False
            if pos[idx, 1] < radii[idx] - 1e-9 or pos[idx, 1] > 1 - radii[idx] + 1e-9:
                valid = False
            if valid:
                # Check against all others
                others = np.setdiff1d(np.arange(n), idx)
                dx = pos[others, 0] - pos[idx, 0]
                dy = pos[others, 1] - pos[idx, 1]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                if np.any(dist < radii[idx] + radii[others] - 1e-9):
                    valid = False
            
            if not valid:
                pos[idx] = old_pos
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Multiple initial configurations
    configs = []
    
    # Grid
    pos, radii = _build_grid(N, jitter=0.0)
    configs.append((pos, radii))
    
    # Grid with jitter
    pos, radii = _build_grid(N, jitter=0.02)
    configs.append((pos, radii))
    
    # Hexagonal
    pos, radii = _build_hex(N, jitter=0.0)
    configs.append((pos, radii))
    
    # Hexagonal with jitter
    pos, radii = _build_hex(N, jitter=0.02)
    configs.append((pos, radii))
    
    # Random starts
    for seed in [42, 123, 456]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos, radii))
    
    # Process each configuration
    for ci, (pos, radii) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining = deadline - elapsed
        
        # First: relax and grow to get a valid starting point
        relax_budget = min(15.0, remaining * 0.4)
        pos, radii = _relax_and_grow(pos, radii, max_iters=2000,
                                      time_budget=relax_budget,
                                      start_time=start_time)
        
        # Then: scipy optimization
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            scipy_budget = min(25.0, remaining * 0.6)
            pos, radii = _scipy_optimize(pos, radii, time_budget=scipy_budget,
                                          start_time=start_time)
        
        # Final local search
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 3:
            pos, radii = _local_search(pos, radii, time_budget=min(5.0, remaining - 1),
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
        remaining = deadline - elapsed
        if remaining > 5:
            best_pos, best_radii = _local_search(best_pos, best_radii,
                                                   time_budget=min(10.0, remaining - 2),
                                                   start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
    
    # Return best result
    if best_pos is None:
        # Fallback: simple grid
        pos, radii = _build_grid(N, jitter=0.0)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
