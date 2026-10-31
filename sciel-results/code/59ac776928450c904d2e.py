"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-8):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _objective(x, penalty_weight=5000.0):
    """
    Objective: minimize negative sum of radii + smooth penalty for violations.
    Uses a high penalty weight to strictly enforce constraints.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-5)
    
    penalty = 0.0
    
    # Boundary penalties (smooth quadratic)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    # Only consider upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Overlap is when dist_sq < min_dist_sq
    # We penalize the square of the violation amount
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    
    # Normalize by min_dist_sq to make penalty scale-invariant
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-8)))
    
    return -np.sum(radii) + penalty_weight * penalty

def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.098
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
            
    # Place 26th circle in a gap, e.g., center of a 2x2 block if possible, or just perturb
    # For simplicity, place it in a corner-ish spot
    pos[25, 0] = 0.5
    pos[25, 1] = 0.5
    radii[25] = 0.08
    
    return pos, radii

def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    y_spacing = r * np.sqrt(3)
    
    # 5 rows: 5, 4, 5, 4, 5 = 23 circles. Plus 3 more?
    # Let's do 5, 4, 5, 4, 4 = 22? No, N=26.
    # 6, 5, 6, 5, 4 = 26.
    
    rows = [6, 5, 6, 5, 4]
    idx = 0
    y_pos = 0.1
    for row_idx, count in enumerate(rows):
        y = 0.1 + row_idx * y_spacing
        x_start = (1.0 - (count - 1) * 2 * r) / 2.0
        for i in range(count):
            pos[idx, 0] = x_start + i * 2 * r
            pos[idx, 1] = y
            radii[idx] = r
            idx += 1
            
    return pos, radii

def _initial_corner_heavy():
    """
    Place large circles in corners and smaller ones in the center.
    This often yields a better sum of radii because corner circles can be larger.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i, (cx, cy) in enumerate(corners):
        pos[i, 0] = cx
        pos[i, 1] = cy
        radii[i] = 0.12
        
    # 4 edges (midpoints)
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        idx = 4 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = 0.08
        
    # Remaining 18 circles in the center area
    # Arrange in a 4x4 grid in the center, plus 2 extra
    center_r = 0.06
    center_x_start = 0.25
    center_y_start = 0.25
    spacing = 0.15
    
    idx = 8
    for i in range(4):
        for j in range(4):
            pos[idx, 0] = center_x_start + i * spacing
            pos[idx, 1] = center_y_start + j * spacing
            radii[idx] = center_r
            idx += 1
            
    # 2 extra circles
    pos[idx, 0] = 0.5
    pos[idx, 1] = 0.5
    radii[idx] = 0.05
    idx += 1
    
    pos[idx, 0] = 0.5
    pos[idx, 1] = 0.3
    radii[idx] = 0.05
    
    return pos, radii

def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii

def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        
        for i in range(N):
            # Boundary check
            max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r_bound * 0.999:
                radii[i] = max_r_bound * 0.999
                valid = False
            if radii[i] < 1e-5:
                radii[i] = 1e-5
                valid = False
                
        # Pairwise check - shrink the smaller circle preferentially
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink proportionally to radius, with slight bias toward smaller
                    total_r = radii[i] + radii[j]
                    if total_r < 1e-12:
                        continue
                    excess = min_dist - dist
                    # Split excess: shrink each by its share of the total
                    share_i = radii[i] / total_r
                    share_j = radii[j] / total_r
                    shrink_i = excess * share_i * 1.01
                    shrink_j = excess * share_j * 1.01
                    radii[i] = max(radii[i] - shrink_i, 1e-5)
                    radii[j] = max(radii[j] - shrink_j, 1e-5)
                    valid = False
                    
        if valid:
            break
            
    return pos, radii

def _grow_and_push(pos, radii, deadline, n_iter=5000):
    """
    Grow-and-Push local search.
    Iteratively tries to grow each circle to its maximum possible radius.
    If blocked, it pushes the blocking objects to create space.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(12345)
    
    for iteration in range(n_iter):
        if time.time() > deadline - 2:
            break
            
        # Pick a random circle to grow
        i = rng.randint(N)
        
        # Calculate max radius for i given current positions of others
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        
        dx = pos[:, 0] - pos[i, 0]
        dy = pos[:, 1] - pos[i, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        limits = dist - radii
        limits[i] = np.inf
        
        min_pair_limit = np.min(limits)
        max_r = min(max_r, min_pair_limit)
        max_r = max(max_r, 1e-5)
        
        if max_r > radii[i] + 1e-7:
            radii[i] = max_r
        else:
            # Blocked. Identify blocker and push.
            blocking_j = np.argmin(limits)
            
            if blocking_j != i and limits[blocking_j] <= radii[i] + 1e-6:
                dx_ij = pos[blocking_j, 0] - pos[i, 0]
                dy_ij = pos[blocking_j, 1] - pos[i, 1]
                dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                
                if dist_ij < 1e-9:
                    dx_ij = rng.normal(0, 0.01)
                    dy_ij = rng.normal(0, 0.01)
                    dist_ij = np.sqrt(dx_ij*dx_ij + dy_ij*dy_ij)
                
                push_mag = 0.002 + rng.uniform(0, 0.005)
                nx = dx_ij / dist_ij
                ny = dy_ij / dist_ij
                
                pos[blocking_j, 0] += nx * push_mag
                pos[blocking_j, 1] += ny * push_mag
                pos[i, 0] -= nx * push_mag * 0.5
                pos[i, 1] -= ny * push_mag * 0.5
                
                pos[blocking_j] = np.clip(pos[blocking_j], 0, 1)
                pos[i] = np.clip(pos[i], 0, 1)
            else:
                # Nudge randomly if stuck
                pos[i, 0] += rng.uniform(-0.002, 0.002)
                pos[i, 1] += rng.uniform(-0.002, 0.002)
                pos[i] = np.clip(pos[i], 0, 1)
                
        pos = np.clip(pos, 0, 1)
        
        # Quick validity check and update
        if _is_valid(pos, radii, tol=1e-7):
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # Repair
            pos_pp, radii_pp = _post_process(pos, radii)
            pos = pos_pp
            radii = radii_pp
            current_score = np.sum(radii)
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
    return best_pos, best_radii, best_score

def _polish(pos, radii, deadline):
    """High-precision local search using L-BFGS-B and SLSQP."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. L-BFGS-B with tighter tolerances
    if time.time() < deadline - 5:
        try:
            result = minimize(lambda x: _objective(x, 10000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 50000})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # 2. SLSQP (often better for tight constraints)
    if time.time() < deadline - 2:
        try:
            result = minimize(lambda x: _objective(x, 10000.0), x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-16, 'maxiter': 1000})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
            
    # 3. Final L-BFGS-B pass from the best found
    if time.time() < deadline - 1:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(lambda x: _objective(x, 20000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
            
    return best_pos, best_radii, best_score

def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # 1. Grow and Push
    if time.time() < deadline - 10:
        gp_pos, gp_radii, gp_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=3000)
        if gp_score > best_score + 1e-12:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
            
    # 2. Gradient Polish
    if time.time() < deadline - 5:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
            
    # 3. Second Grow and Push
    if time.time() < deadline - 2:
        gp_pos, gp_radii, gp_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=2000)
        if gp_score > best_score + 1e-12:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
            
    # 4. Final Polish
    if time.time() < deadline - 1:
        pol_pos, pol_radii, pol_score = _polish(best_pos, best_radii, deadline)
        if pol_score > best_score + 1e-12:
            best_score = pol_score
            best_pos = pol_pos.copy()
            best_radii = pol_radii.copy()
            
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    init_configs = []
    
    # 1. Grid
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    # 2. Hexagonal
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    # 3. Corner Heavy
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    # 4. Random
    for seed in [42, 123]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
        
    # 5. Perturbed Grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # 6. Perturbed Hex
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # 7. Perturbed Corner Heavy
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each config
    # Order: Corner heavy is often good, then Hex, then Grid
    config_order = [2, 6, 1, 5, 0, 4, 3]
    
    for idx in config_order:
        if time.time() > deadline - 8:
            break
        if idx >= len(init_configs):
            continue
            
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
            
    # Fallback
    if best_pos is None or best_score < 0:
        pos, radii = _initial_grid()
        best_pos, best_radii = _post_process(pos, radii)
        best_score = np.sum(best_radii)
        
    # Final high-precision polish on the best found
    if time.time() < deadline - 1:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # One more polish if time permits
    if time.time() < deadline - 0.5:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
