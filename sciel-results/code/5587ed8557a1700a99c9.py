"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Pairwise overlap check - vectorized
    pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(pos_diff**2, axis=2))
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist[mask] < radii_sum[mask] - tol):
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


def _build_corner_heavy(n, seed=None):
    """Build a configuration with circles near corners and edges."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Place some circles in corners, some in edge midpoints, rest random
    pos_list = []
    radii_list = []
    
    # Corners
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for cx, cy in corners:
        if len(pos_list) < n:
            pos_list.append([cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)])
            radii_list.append(0.12)
    
    # Edge midpoints
    edges = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
    for ex, ey in edges:
        if len(pos_list) < n:
            pos_list.append([ex + rng.uniform(-0.03, 0.03), ey + rng.uniform(-0.03, 0.03)])
            radii_list.append(0.1)
    
    # Fill rest randomly
    while len(pos_list) < n:
        pos_list.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
        radii_list.append(0.08)
    
    pos = np.array(pos_list[:n])
    radii = np.array(radii_list[:n])
    pos = np.clip(pos, 0.01, 0.99)
    return pos, radii


def _smooth_max_0(x, k=50.0):
    """Smooth approximation of max(0, x) using softplus-like function."""
    return np.where(x > 0, x + np.log1p(np.exp(-k * x)) / k, np.log1p(np.exp(k * x)) / k)


def _penalty_objective(pos, radii, k=50.0):
    """
    Differentiable penalty-based objective for circle packing.
    Minimizes: -sum(radii) + penalty * (boundary_viol + overlap_viol)
    """
    n = len(pos)
    
    # Base objective: negative sum of radii
    obj = -np.sum(radii)
    
    # Boundary violations (smooth)
    b_left = _smooth_max_0(radii - pos[:, 0], k)
    b_right = _smooth_max_0(pos[:, 0] - (1 - radii), k)
    b_bottom = _smooth_max_0(radii - pos[:, 1], k)
    b_top = _smooth_max_0(pos[:, 1] - (1 - radii), k)
    boundary_viol = np.sum(b_left**2) + np.sum(b_right**2) + np.sum(b_bottom**2) + np.sum(b_top**2)
    
    # Pairwise overlap violations (smooth)
    pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(pos_diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    diff = _smooth_max_0(radii_sum - dist, k)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap_viol = np.sum(diff[mask]**2)
    
    penalty_weight = 500.0
    obj += penalty_weight * (boundary_viol + overlap_viol)
    
    return obj


def _penalty_gradient(pos, radii, k=50.0):
    """
    Gradient of penalty-based objective.
    Returns grad_pos (n,2) and grad_radii (n,).
    """
    n = len(pos)
    grad_pos = np.zeros((n, 2))
    grad_radii = np.zeros(n)
    
    # Base gradient: -1 for each radius
    grad_radii[:] = -1.0
    
    penalty_weight = 500.0
    
    # Boundary gradients (smooth)
    # Left: viol = smooth_max(0, r-x). d/dx = -sigmoid(k*(r-x)) if we use logistic, but let's use derivative of our smooth_max
    # Our smooth_max_0(x, k) = where(x>0, x + log1p(exp(-kx))/k, log1p(exp(kx))/k)
    # For x > 0: d/dx = 1 - exp(-kx)/(1+exp(-kx)) = exp(kx)/(1+exp(kx)) = sigmoid(kx)
    # For x <= 0: d/dx = k*exp(kx)/(1+exp(kx)) = k*sigmoid(kx)
    # Actually, let's just use the subgradient approximation: d/dx of max(0,x) is 1 if x>0 else 0
    # With smoothing, it's approximately sigmoid(kx)
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    # Left boundary: viol = smooth_max(0, r - x)
    x_left = radii - pos[:, 0]
    sig_left = sigmoid(k * x_left)
    # d/dx of viol = -sig_left, d/dr of viol = sig_left
    grad_pos[:, 0] -= penalty_weight * 2 * sig_left * _smooth_max_0(x_left, k)
    grad_radii += penalty_weight * 2 * sig_left * _smooth_max_0(x_left, k)
    
    # Right boundary: viol = smooth_max(0, x - (1-r))
    x_right = pos[:, 0] - (1 - radii)
    sig_right = sigmoid(k * x_right)
    grad_pos[:, 0] += penalty_weight * 2 * sig_right * _smooth_max_0(x_right, k)
    grad_radii -= penalty_weight * 2 * sig_right * _smooth_max_0(x_right, k)
    
    # Bottom boundary: viol = smooth_max(0, r - y)
    x_bottom = radii - pos[:, 1]
    sig_bottom = sigmoid(k * x_bottom)
    grad_pos[:, 1] -= penalty_weight * 2 * sig_bottom * _smooth_max_0(x_bottom, k)
    grad_radii += penalty_weight * 2 * sig_bottom * _smooth_max_0(x_bottom, k)
    
    # Top boundary: viol = smooth_max(0, y - (1-r))
    x_top = pos[:, 1] - (1 - radii)
    sig_top = sigmoid(k * x_top)
    grad_pos[:, 1] += penalty_weight * 2 * sig_top * _smooth_max_0(x_top, k)
    grad_radii -= penalty_weight * 2 * sig_top * _smooth_max_0(x_top, k)
    
    # Pairwise overlap gradients
    pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(pos_diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    x_overlap = radii_sum - dist
    sig_overlap = sigmoid(k * x_overlap)
    viol_overlap = _smooth_max_0(x_overlap, k)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Only consider active pairs
    active = mask & (x_overlap > -0.01)  # Include near-boundary pairs
    
    if np.any(active):
        active_indices = np.argwhere(active)
        for i, j in active_indices:
            d = dist[i, j]
            sig_val = sig_overlap[i, j]
            viol_val = viol_overlap[i, j]
            
            if viol_val < 1e-10:
                continue
                
            factor = -2 * penalty_weight * sig_val * viol_val / d
            
            grad_pos[i, 0] += factor * pos_diff[i, j, 0]
            grad_pos[i, 1] += factor * pos_diff[i, j, 1]
            
            grad_pos[j, 0] -= factor * pos_diff[i, j, 0]
            grad_pos[j, 1] -= factor * pos_diff[i, j, 1]
            
            r_factor = 2 * penalty_weight * sig_val * viol_val
            grad_radii[i] += r_factor
            grad_radii[j] += r_factor
    
    return grad_pos, grad_radii


def _adam_optimize(pos, radii, time_budget=30.0, start_time=None, k=50.0):
    """
    Adam optimizer for the smooth penalty-based objective.
    """
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    
    pos = pos.copy()
    radii = radii.copy()
    
    # Adam state
    m_pos = np.zeros_like(pos)
    v_pos = np.zeros_like(pos)
    m_rad = np.zeros_like(radii)
    v_rad = np.zeros_like(radii)
    
    t = 0
    lr_pos = 0.03
    lr_rad = 0.02
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    last_obj = _penalty_objective(pos, radii, k)
    
    while time.time() - start_time < time_budget:
        t += 1
        obj = _penalty_objective(pos, radii, k)
        grad_pos, grad_rad = _penalty_gradient(pos, radii, k)
        
        # Update Adam state
        m_pos = beta1 * m_pos + (1 - beta1) * grad_pos
        v_pos = beta2 * v_pos + (1 - beta2) * (grad_pos ** 2)
        m_rad = beta1 * m_rad + (1 - beta1) * grad_rad
        v_rad = beta2 * v_rad + (1 - beta2) * (grad_rad ** 2)
        
        m_pos_hat = m_pos / (1 - beta1 ** t)
        v_pos_hat = v_pos / (1 - beta2 ** t)
        m_rad_hat = m_rad / (1 - beta1 ** t)
        v_rad_hat = v_rad / (1 - beta2 ** t)
        
        # Update positions and radii
        pos -= lr_pos * m_pos_hat / (np.sqrt(v_pos_hat) + epsilon)
        radii -= lr_rad * m_rad_hat / (np.sqrt(v_rad_hat) + epsilon)
        
        # Clip to valid bounds
        pos = np.clip(pos, 0.001, 0.999)
        radii = np.clip(radii, 0.005, 0.45)
        
        # Decay learning rate
        if t % 1000 == 0:
            lr_pos *= 0.95
            lr_rad *= 0.95
        
        # Stagnation detection
        if abs(last_obj - obj) < 1e-12:
            if t % 200 == 0:
                pos += np.random.normal(0, 0.002, size=pos.shape)
                radii += np.random.normal(0, 0.001, size=radii.shape)
                pos = np.clip(pos, 0.001, 0.999)
                radii = np.clip(radii, 0.005, 0.45)
                lr_pos *= 1.2
                lr_rad *= 1.2
        last_obj = obj
    
    return pos, radii


def _relax_and_grow(pos, radii, max_iters=3000, time_budget=20.0, start_time=None):
    """Physics-based relaxation with simultaneous radius growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Check boundary violations and fix
        for i in range(n):
            if pos[i, 0] < radii[i]:
                pos[i, 0] = radii[i]
            if pos[i, 0] > 1 - radii[i]:
                pos[i, 0] = 1 - radii[i]
            if pos[i, 1] < radii[i]:
                pos[i, 1] = radii[i]
            if pos[i, 1] > 1 - radii[i]:
                pos[i, 1] = 1 - radii[i]
        
        # Resolve overlaps
        for i, j in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                if dist < 1e-10:
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
        
        # Re-check boundaries
        for i in range(n):
            if pos[i, 0] < radii[i]:
                pos[i, 0] = radii[i]
            if pos[i, 0] > 1 - radii[i]:
                pos[i, 0] = 1 - radii[i]
            if pos[i, 1] < radii[i]:
                pos[i, 1] = radii[i]
            if pos[i, 1] > 1 - radii[i]:
                pos[i, 1] = 1 - radii[i]
        
        # Try to grow radii
        if it % 5 == 0:
            # Binary search for max feasible uniform growth
            lo, hi = 0.0, 0.05
            best_growth = 0.0
            
            for _ in range(8):
                mid = (lo + hi) / 2
                new_radii = radii + mid
                
                if np.any(new_radii > 0.45):
                    hi = mid
                    continue
                
                if np.any(pos[:, 0] < new_radii) or np.any(pos[:, 0] > 1 - new_radii):
                    hi = mid
                    continue
                if np.any(pos[:, 1] < new_radii) or np.any(pos[:, 1] > 1 - new_radii):
                    hi = mid
                    continue
                
                pos_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(pos_diff**2, axis=2))
                radii_sum = new_radii[:, np.newaxis] + new_radii[np.newaxis, :]
                
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.any(dist[mask] < radii_sum[mask] - 1e-9):
                    hi = mid
                else:
                    lo = mid
                    best_growth = mid
            
            if best_growth > 1e-6:
                radii += best_growth
    
    return pos, radii


def _local_search(pos, radii, time_budget=15.0, start_time=None):
    """Local search with random moves and radius growth attempts."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid():
        if np.any(pos[:, 0] < radii - 1e-9) or np.any(pos[:, 0] > 1 - radii + 1e-9):
            return False
        if np.any(pos[:, 1] < radii - 1e-9) or np.any(pos[:, 1] > 1 - radii + 1e-9):
            return False
        for i, j in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False
        return True
    
    while time.time() - start_time < time_budget:
        # Try to grow all radii slightly
        grow = 0.0003
        new_radii = radii + grow
        feasible = True
        for i in range(n):
            if new_radii[i] > 0.45:
                feasible = False
                break
            if pos[i, 0] < new_radii[i] - 1e-9 or pos[i, 0] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
            if pos[i, 1] < new_radii[i] - 1e-9 or pos[i, 1] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
        if feasible:
            for i, j in pairs:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < new_radii[i] + new_radii[j] - 1e-9:
                    feasible = False
                    break
        if feasible:
            radii = new_radii
            score = np.sum(radii)
            if score > best_score:
                best_score = score
            continue
        
        # Try small random moves
        for _ in range(50):
            if time.time() - start_time > time_budget:
                break
            idx = np.random.randint(n)
            old_pos = pos[idx].copy()
            pos[idx] += np.random.normal(0, 0.003, size=2)
            pos[idx] = np.clip(pos[idx], 0, 1)
            
            if not is_valid():
                pos[idx] = old_pos
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    def get_fallback():
        side = int(np.ceil(np.sqrt(N)))
        pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                        for j in range(side) for i in range(side)][:N])
        radii = np.full(N, 0.45 / side)
        return pos, radii
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Grid
    pos, radii = _build_grid(N, jitter=0.0)
    configs.append((pos, radii))
    
    # 2. Hexagonal
    pos, radii = _build_hex(N, jitter=0.0)
    configs.append((pos, radii))
    
    # 3. Random
    pos, radii = _build_random(N, seed=42)
    configs.append((pos, radii))
    
    # 4. Corner-heavy
    pos, radii = _build_corner_heavy(N, seed=123)
    configs.append((pos, radii))
    
    # Process each configuration
    for ci, (pos_init, radii_init) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining = deadline - elapsed
        if remaining < 15:
            break
        
        pos = pos_init.copy()
        radii = radii_init.copy()
        
        # Step 1: Relax and grow
        budget1 = min(8.0, remaining * 0.3)
        pos, radii = _relax_and_grow(pos, radii, max_iters=2000,
                                      time_budget=budget1, start_time=start_time)
        
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Step 2: Adam optimization on the smooth penalty objective
        remaining = deadline - (time.time() - start_time)
        if remaining > 20:
            budget2 = min(30.0, remaining * 0.5)
            pos_adam, radii_adam = _adam_optimize(pos, radii,
                                                   time_budget=budget2,
                                                   start_time=start_time, k=50.0)
            
            if _check_validity(pos_adam, radii_adam):
                score = np.sum(radii_adam)
                if score > best_score:
                    best_score = score
                    best_pos = pos_adam.copy()
                    best_radii = radii_adam.copy()
        
        # Step 3: Local search polish
        remaining = deadline - (time.time() - start_time)
        if remaining > 10 and best_pos is not None:
            budget3 = min(15.0, remaining - 5)
            best_pos, best_radii = _local_search(best_pos, best_radii,
                                                   time_budget=budget3,
                                                   start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
    
    # Final check
    if best_pos is None:
        best_pos, best_radii = get_fallback()
    
    if not _check_validity(best_pos, best_radii):
        best_pos, best_radii = get_fallback()
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr

# EVOLVE-BLOCK-END
