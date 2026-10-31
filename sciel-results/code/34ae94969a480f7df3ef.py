"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
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


def _objective(x):
    """Objective: minimize negative sum of radii + smooth penalty for violations.
    
    Uses log-sum-exp for smooth gradients on binding constraints.
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = np.maximum(x[2*n:], 1e-4)
    
    # Boundary violations: how much each circle sticks out
    # For circle i: violations are max(0, r_i - x_i), max(0, x_i - (1-r_i)), etc.
    # Smooth via soft-max of violations
    boundary_viol = np.zeros(4*n)
    for i in range(n):
        boundary_viol[4*i]   = max(0.0, radii[i] - pos[i, 0])
        boundary_viol[4*i+1] = max(0.0, pos[i, 0] - (1.0 - radii[i]))
        boundary_viol[4*i+2] = max(0.0, radii[i] - pos[i, 1])
        boundary_viol[4*i+3] = max(0.0, pos[i, 1] - (1.0 - radii[i]))
    
    # Pairwise violations
    m = n*(n-1)//2
    pair_viol = np.zeros(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            pair_viol[idx] = max(0.0, min_dist - dist)
            idx += 1
    
    # Log-sum-exp penalty: smooth, differentiable, focuses on binding constraints
    # penalty = sum(log(1 + exp(violation / tau))) * tau  -- this is smooth
    # For small violations, ~violation; for large, ~violation + tau*log(1)
    tau = 0.01
    def _lse_penalty(v):
        """Smooth sum of violations via log-sum-exp."""
        if np.max(v) < 1e-10:
            return 0.0
        v_scaled = v / tau
        v_max = np.max(v_scaled)
        # log-sum-exp for stability
        lse = v_max + np.log(np.sum(np.exp(v_scaled - v_max)))
        return lse * tau
    
    boundary_pen = _lse_penalty(boundary_viol)
    pair_pen = _lse_penalty(pair_viol)
    
    return -np.sum(radii) + 50.0 * boundary_pen + 50.0 * pair_pen


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.09
    
    pos[25, 0] = 0.1
    pos[25, 1] = 0.1
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.09
    for i in range(5):
        idx = i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1
        radii[idx] = r
    
    dy = r * np.sqrt(3)
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + dy
        radii[idx] = r
    
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 2 * dy
        radii[idx] = r
    
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + 3 * dy
        radii[idx] = r
    
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 4 * dy
        radii[idx] = r
    
    pos[25, 0] = 0.5
    pos[25, 1] = 0.95
    radii[25] = 0.05
    
    return pos, radii


def _initial_center_ring():
    """Center circle surrounded by a ring of circles."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Center circle
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.12
    
    # Ring of 12 circles around center
    ring_r = 0.08
    center_r = 0.12
    ring_dist = center_r + ring_r  # distance from center to ring centers
    
    for i in range(12):
        idx = 1 + i
        angle = 2 * np.pi * i / 12
        pos[idx, 0] = 0.5 + ring_dist * np.cos(angle)
        pos[idx, 1] = 0.5 + ring_dist * np.sin(angle)
        radii[idx] = ring_r
    
    # 13 more circles in corners and edges
    corner_r = 0.06
    corners = [
        (0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
        (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5),
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.5)  # This will overlap with center, will be fixed
    ]
    for i in range(13):
        idx = 13 + i
        pos[idx, 0] = corners[i][0]
        pos[idx, 1] = corners[i][1]
        radii[idx] = corner_r
    
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
    
    for iteration in range(500):
        valid = True
        
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.999, 1e-4)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.999)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.999)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.999)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.999
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _grow_and_push(pos, radii, deadline):
    """Iteratively grow each circle and push neighbors to make room.
    
    This is a local search that tries to increase the sum of radii by
    growing one circle at a time and adjusting positions of overlapping circles.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    n = N
    rng = np.random.RandomState(7)
    
    for outer_iter in range(200):
        if time.time() > deadline - 15:
            break
        
        # Pick a random circle to try to grow
        i = rng.randint(n)
        
        # Try to grow circle i by a small amount
        # First, compute how much we can grow without violating constraints
        max_growth = 0.05  # max we'll try
        
        # Check boundary limits
        max_growth = min(max_growth, pos[i, 0] - radii[i])
        max_growth = min(max_growth, 1.0 - pos[i, 0] - radii[i])
        max_growth = min(max_growth, pos[i, 1] - radii[i])
        max_growth = min(max_growth, 1.0 - pos[i, 1] - radii[i])
        
        if max_growth < 1e-6:
            continue
        
        # Try growing by a fraction of max
        growth = max_growth * 0.5
        new_radii = radii.copy()
        new_radii[i] += growth
        
        # Check if this causes overlaps; if so, push neighbors
        new_pos = pos.copy()
        feasible = True
        
        for j in range(n):
            if j == i:
                continue
            dx = new_pos[i, 0] - new_pos[j, 0]
            dy = new_pos[i, 1] - new_pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = new_radii[i] + new_radii[j]
            
            if dist < min_dist - 1e-9:
                # Push j away from i
                if dist < 1e-10:
                    # Random direction
                    angle = rng.uniform(0, 2 * np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    dist = 1.0
                else:
                    dx /= dist
                    dy /= dist
                
                push = (min_dist - dist) * 0.5
                new_pos[j, 0] += dx * push
                new_pos[j, 1] += dy * push
        
        # Check if new_pos is still valid (within bounds, no overlaps)
        valid = True
        for k in range(n):
            if new_pos[k, 0] < new_radii[k] or new_pos[k, 0] > 1.0 - new_radii[k]:
                valid = False
                break
            if new_pos[k, 1] < new_radii[k] or new_pos[k, 1] > 1.0 - new_radii[k]:
                valid = False
                break
            if valid:
                for l in range(k + 1, n):
                    dx = new_pos[k, 0] - new_pos[l, 0]
                    dy = new_pos[k, 1] - new_pos[l, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_radii[k] + new_radii[l] - 1e-9:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            new_score = np.sum(new_radii)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos = new_pos
                radii = new_radii
    
    return best_pos, best_radii


def _polish(pos, radii, deadline):
    """High-precision local search to squeeze out more sum of radii.
    
    Uses multiple optimization passes with tightening tolerances.
    """
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
    
    # Pass 1: L-BFGS-B with moderate tolerance
    if time.time() < deadline - 20:
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Pass 2: SLSQP with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-16})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Pass 3: Final L-BFGS-B with very tight tolerance
    if time.time() < deadline - 5:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
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
    
    # L-BFGS-B first (fast gradient-based)
    if time.time() < deadline - 15:
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-9})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Grow and push: try to expand circles locally
    if time.time() < deadline - 15:
        gp_pos, gp_radii = _grow_and_push(best_pos, best_radii, deadline)
        gp_score = np.sum(gp_radii)
        if gp_score > best_score:
            best_score = gp_score
            best_pos = gp_pos.copy()
            best_radii = gp_radii.copy()
    
    # Powell (derivative-free, good for fine-tuning)
    if time.time() < deadline - 15:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(_objective, x0, method='Powell', bounds=bounds,
                              options={'maxiter': 800, 'ftol': 1e-13})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 8:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def _differential_evolution_run(deadline):
    """Run differential evolution for global search."""
    # Define bounds for DE
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
    for i in range(N):
        bounds.append((0.01, 0.25))  # radius
    
    # Use a subset of the time for DE
    de_deadline = deadline - 20
    
    try:
        result = differential_evolution(
            _objective, bounds,
            maxiter=100, popsize=15,
            tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
            seed=42, polish=False,
            updating='deferred'
        )
        
        if result.fun < 0:  # Valid improvement
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
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed hexagonal
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Corner-heavy initial config
    pos_ch = np.zeros((N, 2))
    radii_ch = np.zeros(N)
    # Place circles in corners and edges with larger radii
    corner_positions = [
        (0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88),
        (0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5),
        (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7),
        (0.5, 0.5), (0.2, 0.5), (0.8, 0.5), (0.5, 0.2),
        (0.5, 0.8), (0.2, 0.2), (0.8, 0.8), (0.2, 0.8),
        (0.8, 0.2), (0.35, 0.5), (0.65, 0.5), (0.5, 0.35),
        (0.5, 0.65), (0.4, 0.6)
    ]
    for i in range(N):
        pos_ch[i, 0] = corner_positions[i][0]
        pos_ch[i, 1] = corner_positions[i][1]
        radii_ch[i] = 0.08
    init_configs.append((pos_ch, radii_ch))
    
    # Run optimization from each initial config
    # Use more time per config since we have fewer configs
    time_per_config = min(20.0, (deadline - start_time - 20) / max(1, len(init_configs)))
    
    for idx, (pos, radii) in enumerate(init_configs):
        if time.time() > deadline - 20:
            break
        
        config_deadline = min(deadline - 10, time.time() + time_per_config)
        pos_opt, radii_opt, score = _optimize_from(pos, radii, config_deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Try differential evolution if time permits
    if time.time() < deadline - 30:
        de_pos, de_radii, de_score = _differential_evolution_run(deadline)
        if de_pos is not None and de_score > best_score:
            best_score = de_score
            best_pos = de_pos.copy()
            best_radii = de_radii.copy()
    
    # Fallback if nothing valid found
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
    
    # Final post-processing and polish
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 3:
        best_pos, best_radii, _ = _polish(best_pos, best_radii, deadline)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
