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
    Uses vectorized computation and high penalty weight.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-5)
    
    penalty = 0.0
    
    # Boundary penalties (smooth quadratic, vectorized)
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
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * np.maximum(min_dist_sq[mask], 1e-8)))
    
    return -np.sum(radii) + penalty_weight * penalty


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.098
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
            
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
    
    rows = [6, 5, 6, 5, 4]
    idx = 0
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
    """Place large circles in corners and smaller ones in the center."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i, (cx, cy) in enumerate(corners):
        pos[i, 0] = cx
        pos[i, 1] = cy
        radii[i] = 0.12
        
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        idx = 4 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = 0.08
        
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
    """Iteratively shrink circles to ensure validity with proportional shrinking."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        
        for i in range(N):
            max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r_bound * 0.999:
                radii[i] = max_r_bound * 0.999
                valid = False
            if radii[i] < 1e-5:
                radii[i] = 1e-5
                valid = False
                
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    total_r = radii[i] + radii[j]
                    if total_r < 1e-12:
                        continue
                    excess = min_dist - dist
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
        dist = np.sqrt(dx * dx + dy * dy)
        
        for j in range(N):
            if j == i:
                continue
            if dist[j] < 1e-12:
                continue
            max_r = min(max_r, dist[j] - radii[j])
        
        if max_r <= radii[i]:
            continue
            
        # Try to grow circle i
        new_r = max_r * 0.999
        if new_r <= radii[i] + 1e-10:
            continue
            
        # Check if growing creates overlaps with neighbors
        old_r = radii[i]
        radii[i] = new_r
        
        # Push back any circles that now overlap
        pushed = False
        for j in range(N):
            if j == i:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist and dist > 1e-12:
                # Push circle j away
                push_dist = min_dist - dist
                push_x = (dx / dist) * push_dist
                push_y = (dy / dist) * push_dist
                
                # Weight push toward smaller circles
                weight = radii[j] / (radii[i] + radii[j])
                pos[j, 0] += push_x * weight
                pos[j, 1] += push_y * weight
                pushed = True
        
        # Check boundary violations after pushing
        boundary_violation = False
        for j in range(N):
            if pos[j, 0] < radii[j] or pos[j, 0] > 1.0 - radii[j] or \
               pos[j, 1] < radii[j] or pos[j, 1] > 1.0 - radii[j]:
                boundary_violation = True
                break
        
        if boundary_violation:
            # Revert
            radii[i] = old_r
            # Revert positions (simplified - just check validity)
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            continue
        
        # Check validity
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score + 1e-12:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        else:
            # Revert
            radii[i] = old_r
    
    return best_pos, best_radii, best_score


def _polish(pos, radii, deadline):
    """High-precision local search with multiple passes to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Multiple L-BFGS-B passes with decreasing tolerances
    tolerances = [(1e-12, 1e-8), (1e-14, 1e-10), (1e-16, 1e-12)]
    for ftol, gtol in tolerances:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': ftol, 'gtol': gtol})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # SLSQP polish with tighter bounds
    if time.time() < deadline - 3:
        try:
            tight_bounds = []
            for i in range(N):
                x_lo = max(0.0, best_pos[i, 0] - 0.05)
                x_hi = min(1.0, best_pos[i, 0] + 0.05)
                y_lo = max(0.0, best_pos[i, 1] - 0.05)
                y_hi = min(1.0, best_pos[i, 1] + 0.05)
                r_lo = max(1e-5, best_radii[i] * 0.8)
                r_hi = min(0.5, best_radii[i] * 1.2)
                tight_bounds.append((x_lo, x_hi))
                tight_bounds.append((y_lo, y_hi))
            for i in range(N):
                tight_bounds.append((r_lo, r_hi))
            
            result = minimize(_objective, x0, method='SLSQP', bounds=tight_bounds,
                              options={'maxiter': 2000, 'ftol': 1e-16, 'eps': 1e-12})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # Final Powell pass
    if time.time() < deadline - 2:
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-14})
            
            if result.fun < _objective(x0) - 1e-15:
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
    
    # Final grow-and-push refinement
    if time.time() < deadline - 2:
        best_pos, best_radii, best_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=2000)
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration with time-aware allocation."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.25))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # L-BFGS-B first
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # SLSQP intermediate pass
    if time.time() < deadline - 7:
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-14, 'eps': 1e-10})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Powell pass
    if time.time() < deadline - 6:
        try:
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 800, 'ftol': 1e-13})
            
            if result.fun < _objective(x0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Grow-and-push
    if time.time() < deadline - 5:
        best_pos, best_radii, best_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=3000)
    
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def _differential_evolution_run(deadline):
    """Run differential evolution for global search with time monitoring."""
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.25))
    
    de_deadline = deadline - 25
    
    try:
        result = differential_evolution(
            _objective, bounds,
            maxiter=80, popsize=12,
            tol=1e-7, mutation=(0.5, 1.5), recombination=0.7,
            seed=42, polish=False,
            updating='deferred',
            callback=lambda xk, convergence: (time.time() > de_deadline)
        )
        
        if result.fun < 0:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp, score
    except Exception:
        pass
    
    return None, None, -np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    config_order = [0, 1, 2, 5, 6, 3, 4]
    for idx in config_order:
        if time.time() > deadline - 20:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    if time.time() < deadline - 30:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None and de_score > best_score + 1e-12:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    if best_pos is None or best_score < 0:
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx2 = i * 5 + j
                if idx2 < N:
                    pos[idx2, 0] = (i + 0.5) / 5.0
                    pos[idx2, 1] = (j + 0.5) / 5.0
                    radii[idx2] = 0.09
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    if time.time() < deadline - 2:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
