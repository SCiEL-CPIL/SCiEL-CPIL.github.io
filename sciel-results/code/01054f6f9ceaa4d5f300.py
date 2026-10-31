"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Smooth differentiable penalty objective (soft-max collisions) optimized by L-BFGS-B from diverse starts.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, k=100.0):
    """Numerically stable softplus approximation of max(0, x)."""
    return np.logaddexp(0, k * x) / k

def _soft_max(x, k=50.0):
    """Smooth approximation of max(x)."""
    if len(x) == 0:
        return 0.0
    return np.logaddexp.reduce(k * x) / k

def _objective(x, penalty_weight=100.0):
    """
    Objective function for L-BFGS-B.
    x: flattened array [x0, y0, x1, y1, ..., x25, y25, r0, r1, ..., r25]
    Returns: -sum(radii) + penalty * (boundary_violations + collision_violations)
    """
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # 1. Boundary violations
    # dist from center to wall must be >= radius
    # violation_i = max(0, r_i - min(x_i, 1-x_i, y_i, 1-y_i))
    walls = np.minimum(
        np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
        np.minimum(pos[:, 1], 1.0 - pos[:, 1])
    )
    boundary_violations = _soft_max(radii - walls, k=50.0)
    
    # 2. Collision violations
    # dist(i, j) >= r_i + r_j
    # violation_ij = max(0, r_i + r_j - dist(i, j))
    # We compute all pairwise distances and sum the soft violations
    # To be efficient, we can use a vectorized approach or loop.
    # Given N=26, N^2/2 = 325 pairs. Looping is fine in Python if not too slow, 
    # but vectorized is better for gradient.
    # Let's use a vectorized calculation for the penalty term.
    
    # Pairwise distances
    # pos is (n, 2)
    # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] -> (n, n, 2)
    # dist = np.linalg.norm(diff, axis=2) -> (n, n)
    # sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :] -> (n, n)
    # violations = sum_radii - dist
    # We only care about i < j.
    
    # For speed and memory, we can just compute the upper triangle.
    # But _soft_max works on an array.
    # Let's build the array of violations for i < j.
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dists = np.sqrt(dx*dx + dy*dy)
    sum_r = radii[i_idx] + radii[j_idx]
    violations = sum_r - dists
    
    collision_violations = _soft_max(violations, k=50.0)
    
    penalty = penalty_weight * (boundary_violations + collision_violations)
    
    # Maximize sum of radii -> Minimize -sum(radii)
    return -np.sum(radii) + penalty

def _objective_smooth(x, penalty_weight=100.0):
    """
    Wrapper for minimize.
    """
    return _objective(x, penalty_weight)

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(radii < 0):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    for _ in range(300):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    k = (dist / (sum_r + 1e-12)) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
        if valid:
            break
    return pos, radii

def _initial_best_fit():
    """
    Initialize circles at grid points and set radii to the maximum possible
    given only the other circles' initial positions (ignoring their radii initially,
    or using a small epsilon). This creates a "best fit" starting point where
    circles are pushed against each other and walls.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid
    points = []
    for i in range(5):
        for j in range(5):
            points.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0])
    
    # Use first 25 points, leave center for a smaller one? 
    # Actually, let's just use a random subset or the first 25.
    # Better: Use a hexagonal packing approximation or just grid.
    # Let's use the 5x5 grid minus the center for 24, and 2 randoms?
    # Or just 25 grid points + 1 center?
    
    # Let's use 25 grid points (5x5) and 1 at center (0.5, 0.5) is already in grid.
    # So we need 26 points.
    # 5x5 = 25. Add one more?
    # Let's just use 25 grid points and shift them slightly or use a different grid.
    # 6x5 = 30 points. Pick 26?
    
    # Simpler: Random initialization with "best fit" radii calculation.
    rng = np.random.RandomState(42)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    
    # Calculate max radius for each circle given others' positions (assuming others have r=0)
    # This is a lower bound on the true max radius, but gives a good start.
    for i in range(N):
        x, y = pos[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(N):
            if i == j: continue
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Assume other circles have radius 0 for this initial estimate
            max_r = min(max_r, dist)
        radii[i] = max(max_r * 0.9, 0.01)
        
    return pos, radii

def _initial_hexagonal():
    """
    Hexagonal packing initialization.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Generate hexagonal points
    # Spacing s. r = s / (2 * sqrt(3))
    # Let's try to fit 26 circles.
    # Rows:
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle
    # Total 26.
    
    s = 0.18 # spacing
    r = s / (2 * np.sqrt(3))
    
    k = 0
    for row in range(6):
        if row == 5:
            num = 1
        else:
            num = 5
            
        x_offset = (row % 2) * s / 2
        for col in range(num):
            x = x_offset + col * s + 0.1
            y = row * s * np.sqrt(3) / 2 + 0.1
            if x < 1.0 and y < 1.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            if k >= N:
                break
        if k >= N:
            break
            
    return pos, radii

def _initial_corner_biased():
    """
    Place large circles in corners, smaller in center.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corners
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.14
        
    # 4 edges (midpoints)
    edges = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
    for i in range(4, 8):
        pos[i] = edges[i-4]
        radii[i] = 0.12
        
    # 4 inner corners
    inner = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    for i in range(8, 12):
        pos[i] = inner[i-8]
        radii[i] = 0.10
        
    # Remaining 14 random
    rng = np.random.RandomState(123)
    for i in range(12, N):
        pos[i] = rng.uniform(0.2, 0.8, 2)
        radii[i] = 0.05
        
    return pos, radii

def _initial_grid():
    """
    Simple grid.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 6x5 grid = 30 points. Take 26.
    points = []
    for i in range(6):
        for j in range(5):
            points.append([(i + 0.5) / 6.0, (j + 0.5) / 5.0])
            
    # Shuffle and take 26? Or just first 26.
    # Let's take a subset that covers the square well.
    # Actually, 5x5 grid = 25 points. Add one more at (0.5, 0.5) shifted?
    # Let's just use 5x5 grid and one random.
    for i in range(5):
        for j in range(5):
            pos[i*5+j] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[i*5+j] = 0.1
    pos[25] = [0.5, 0.5]
    radii[25] = 0.05
            
    return pos, radii

def _initial_random(seed):
    """
    Random initialization.
    """
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.15, N)
    return pos, radii

def _force_directed_relax(pos, radii, n_iters=1000, deadline=None):
    """
    Simple force-directed relaxation to push circles apart and grow them.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        forces = np.zeros_like(pos)
        
        # Repulsion between circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dx, dy = 1e-6, 1e-6
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                # Desired distance
                desired = radii[i] + radii[j]
                
                if dist < desired:
                    # Push apart
                    f = (desired - dist) / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                    # Grow radii slightly
                    radii[i] += 0.001
                    radii[j] += 0.001
                    
        # Attraction to walls (grow towards walls)
        for i in range(N):
            # Push towards nearest wall
            x, y = pos[i]
            # Force towards left wall if close
            if x < radii[i]:
                forces[i, 0] -= (radii[i] - x) * 10
            if x > 1 - radii[i]:
                forces[i, 0] += (x - (1 - radii[i])) * 10
            if y < radii[i]:
                forces[i, 1] -= (radii[i] - y) * 10
            if y > 1 - radii[i]:
                forces[i, 1] += (y - (1 - radii[i])) * 10
                
        # Update positions
        pos += forces * 0.1
        pos = np.clip(pos, 0.01, 0.99)
        
        # Post-process to ensure validity
        pos, radii = _post_process(pos, radii)
        
    return pos, radii

def _constrained_optimize(pos, radii, deadline):
    """
    Use L-BFGS-B with penalty method.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
        
    # We will run L-BFGS-B with a high penalty weight to enforce constraints.
    # Then we might do a second pass with lower penalty to maximize sum.
    
    # Start with high penalty
    res = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 100, 'ftol': 1e-10})
    
    if res.success or res.fun < _objective_smooth(x0):
        x = res.x
        pos_out = x[:2*N].reshape(N, 2)
        radii_out = np.maximum(x[2*N:], 1e-6)
        pos_out, radii_out = _post_process(pos_out, radii_out)
        
        # If valid, try to maximize sum with lower penalty (just maximize sum, 
        # but L-BFGS-B doesn't handle constraints well without penalty.
        # So we stick with penalty.
        # We can try increasing penalty weight to tighten constraints.
        
        # Second pass with higher penalty
        res2 = minimize(lambda x: _objective(x, penalty_weight=500.0), x, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 100, 'ftol': 1e-10})
        if res2.success or res2.fun < res.fun:
            x = res2.x
            pos_out = x[:2*N].reshape(N, 2)
            radii_out = np.maximum(x[2*N:], 1e-6)
            pos_out, radii_out = _post_process(pos_out, radii_out)
            
        return pos_out, radii_out
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    inits = [
        _initial_best_fit(),
        _initial_hexagonal(),
        _initial_corner_biased(),
        _initial_grid(),
        _initial_random(1),
        _initial_random(2),
        _initial_random(3),
    ]

    best_pos, best_radii = inits[0]
    best_score = np.sum(best_radii)

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        pos_init, rad_init = _post_process(pos_init, rad_init)
        
        # Force directed relaxation to improve initial state
        pos_fd, rad_fd = _force_directed_relax(pos_init, rad_init, n_iters=200, deadline=deadline)
        score_fd = np.sum(rad_fd)
        
        # L-BFGS-B optimization
        pos_opt, rad_opt = _constrained_optimize(pos_fd, rad_fd, deadline)
        pos_opt, rad_opt = _post_process(pos_opt, rad_opt)
        score_opt = np.sum(rad_opt)
        
        # Choose best of FD and Optimized
        if score_opt > score_fd:
            pos_curr, rad_curr = pos_opt, rad_opt
            score_curr = score_opt
        else:
            pos_curr, rad_curr = pos_fd, rad_fd
            score_curr = score_fd
            
        if score_curr > best_score:
            best_score = score_curr
            best_pos = pos_curr.copy()
            best_radii = rad_curr.copy()

    # Final polish on best result with SLSQP (hard constraints) if time permits
    if time.time() < deadline - 10:
        # We can try SLSQP on the best result found by L-BFGS-B
        # But SLSQP is slow. Let's stick with L-BFGS-B penalty.
        # Maybe one more L-BFGS-B pass with very high penalty?
        pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        # Fallback to a simple valid config
        pos, radii = _initial_grid()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
