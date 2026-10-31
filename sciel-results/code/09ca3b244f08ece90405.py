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


def _objective(x, beta=50.0):
    """Smooth objective: minimize negative sum of radii + soft penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Soft penalty for going outside square using log-sum-exp style
    penalty = 0.0
    for i in range(n):
        # Left boundary violation
        v = radii[i] - pos[i, 0]
        if v > 0:
            penalty += v * v
        # Right boundary violation
        v = pos[i, 0] - (1.0 - radii[i])
        if v > 0:
            penalty += v * v
        # Bottom boundary violation
        v = radii[i] - pos[i, 1]
        if v > 0:
            penalty += v * v
        # Top boundary violation
        v = pos[i, 1] - (1.0 - radii[i])
        if v > 0:
            penalty += v * v
    
    # Soft penalty for overlapping using smooth approximation
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            # Smooth penalty: use max(0, min_dist - dist)^2 but smoothed
            violation = min_dist - dist
            if violation > 0:
                # Quadratic penalty with smooth transition
                penalty += violation * violation
            elif violation > -0.01:
                # Small smooth region for gradient
                penalty += 0.0 * violation  # zero but keeps differentiability
    
    return -np.sum(radii) + beta * penalty


def _objective_smooth(x, beta=50.0, smooth_eps=0.005):
    """Fully smooth objective using soft-min via log-sum-exp for overlaps."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Boundary penalty (smooth)
    penalty = 0.0
    for i in range(n):
        for coord in range(2):
            # Distance to left/bottom boundary
            d = pos[i, coord] - radii[i]
            if d < 0:
                penalty += d * d
            # Distance to right/top boundary
            d = (1.0 - pos[i, coord]) - radii[i]
            if d < 0:
                penalty += d * d
    
    # Overlap penalty using smooth max
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            # Smooth approximation of max(0, min_dist - sqrt(dist_sq))^2
            # Use: if dist_sq < min_dist^2, penalty = (min_dist - sqrt(dist_sq))^2
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                violation = min_dist - dist
                if violation > 0:
                    penalty += violation * violation
    
    return -np.sum(radii) + beta * penalty


def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    
    if strategy == 'grid':
        # 5x5 grid, use 25 cells, add one more
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        # Add 26th circle in a gap
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
        return pos, radii
    
    elif strategy == 'hexagonal':
        # Hexagonal packing pattern
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        r = 0.1
        rows = 5
        for row in range(rows):
            y = 0.1 + row * r * np.sqrt(3)
            if row % 2 == 0:
                xs = np.arange(5) * 2 * r + r
            else:
                xs = np.arange(5) * 2 * r + 2 * r
            for col, x in enumerate(xs):
                idx = row * 5 + col
                if idx < N:
                    pos[idx] = [x, y]
                    radii[idx] = r
        return pos, radii
    
    elif strategy == 'center_ring':
        # One center, rings around
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        pos[0] = [0.5, 0.5]
        radii[0] = 0.15
        
        # Inner ring: 8 circles
        r_inner = 0.12
        dist_inner = 0.3
        for i in range(8):
            angle = 2 * np.pi * i / 8
            pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
            radii[i+1] = r_inner
        
        # Outer ring: 17 circles
        r_outer = 0.08
        dist_outer = 0.42
        for i in range(17):
            angle = 2 * np.pi * i / 17
            pos[i+9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
            radii[i+9] = r_outer
        
        return pos, radii
    
    elif strategy == 'random':
        # Random initial positions
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.1, size=N)
        return pos, radii
    
    else:
        # Default: slightly perturbed grid
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        # Check boundaries
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(0.001, pos[i, 0] * 0.98)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 0]) * 0.98))
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], max(0.001, pos[i, 1] * 0.98))
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 1]) * 0.98))
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both proportionally
                    scale = (dist / (min_dist + 1e-12)) * 0.98
                    radii[i] = max(0.001, radii[i] * scale)
                    radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _polish(pos, radii, deadline=None):
    """High-precision polish using L-BFGS-B with tight tolerances."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    try:
        result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if result.fun < _objective_smooth(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    # Try Nelder-Mead polish
    try:
        result_nm = minimize(_objective_smooth, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-12})
        if result_nm.fun < _objective_smooth(x0):
            x_opt = result_nm.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _force_directed_anneal(pos, radii, n_iter=500, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Initialize with small random velocities
    vel = np.zeros_like(pos)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        temp = 0.02 * np.exp(-0.002 * it)
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces (push circles inward if too close to boundary)
        for i in range(N):
            # Left
            if pos[i, 0] < radii[i] + 0.01:
                forces[i, 0] += 5.0 * (radii[i] + 0.01 - pos[i, 0])
            # Right
            if pos[i, 0] > 1.0 - radii[i] - 0.01:
                forces[i, 0] -= 5.0 * (pos[i, 0] - (1.0 - radii[i] - 0.01))
            # Bottom
            if pos[i, 1] < radii[i] + 0.01:
                forces[i, 1] += 5.0 * (radii[i] + 0.01 - pos[i, 1])
            # Top
            if pos[i, 1] > 1.0 - radii[i] - 0.01:
                forces[i, 1] -= 5.0 * (pos[i, 1] - (1.0 - radii[i] - 0.01))
        
        # Repulsion forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    # Strong repulsion
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.5 and dist > 1e-6:
                    # Weak repulsion to encourage separation
                    force_mag = 0.5 * (min_dist * 1.5 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update positions with velocity damping and random perturbation
        vel = 0.9 * vel + 0.1 * forces
        pos += vel * 0.01 + np.random.normal(0, temp, size=pos.shape)
        
        # Clamp to valid range
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        # Try to grow radii slightly
        for i in range(N):
            # Check if we can grow this circle
            can_grow = True
            max_grow = 0.001
            # Boundary constraints
            max_grow = min(max_grow, pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            # Overlap constraints
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0001:
                radii[i] += max_grow * 0.5
        
        # Check validity and update best
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['grid', 'hexagonal', 'center_ring', 'random']
    
    # Phase 1: Multi-start local optimization
    for strategy in strategies:
        if time.time() > deadline - 60:
            break
        
        for seed in range(4):
            if time.time() > deadline - 55:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            
            # Combine into single vector
            x0 = np.concatenate([pos.flatten(), radii])
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((0.005, 0.995))
                bounds.append((0.005, 0.995))
            for i in range(N):
                bounds.append((0.005, 0.35))
            
            # Try L-BFGS-B with smooth objective
            try:
                result = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                if result.fun < _objective_smooth(x0):
                    x_opt = result.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    # Post-process to ensure validity
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
            
            # Try Nelder-Mead
            if time.time() > deadline - 50:
                break
            
            try:
                x_start = np.concatenate([pos.flatten(), radii])
                result_nm = minimize(_objective_smooth, x_start, method='Nelder-Mead',
                                     options={'maxiter': 1000, 'fatol': 1e-10})
                if result_nm.fun < _objective_smooth(x_start):
                    x_opt = result_nm.x
                    pos_opt = x_opt[:2*N].reshape(N, 2)
                    radii_opt = x_opt[2*N:]
                    
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    score = np.sum(radii_pp)
                    
                    if _is_valid(pos_pp, radii_pp) and score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Phase 2: Force-directed annealing from best result
    if best_pos is not None and time.time() < deadline - 30:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=800, deadline=deadline - 25)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_pol, radii_pol = _polish(best_pos, best_radii, deadline=deadline - 2)
            score_pol = np.sum(radii_pol)
            if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
        except Exception:
            pass
    
    # If we didn't find a valid solution, fall back to simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
