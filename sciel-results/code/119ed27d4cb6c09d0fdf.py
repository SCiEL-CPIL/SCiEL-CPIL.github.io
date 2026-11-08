"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _gradient_optimize(pos0, radii0, time_budget=30.0, start_time=None):
    """
    Gradient-based optimization using smooth penalty functions.
    Maximizes sum(radii) subject to smooth constraint violations.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(pos0)
    pos = pos0.copy()
    radii = radii0.copy()
    
    # Penalty weight for constraint violations
    lambda_overlap = 50.0
    lambda_boundary = 50.0
    
    # Learning rates
    lr_pos = 0.001
    lr_rad = 0.0001
    
    # Momentum
    vel_pos = np.zeros_like(pos)
    vel_rad = np.zeros_like(radii)
    beta = 0.9
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Compute gradients
        grad_pos = np.zeros_like(pos)
        grad_rad = np.zeros_like(radii)
        
        # Objective: maximize sum(radii)
        grad_rad[:] = 1.0
        
        # Penalty for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Smooth penalty: (min_dist - dist)^2
                    penalty = min_dist - dist
                    if dist < 1e-10:
                        # Random direction if coincident
                        angle = np.random.uniform(0, 2 * np.pi)
                        nx, ny = np.cos(angle), np.sin(angle)
                    else:
                        nx, ny = dx / dist, dy / dist
                    
                    # Gradient of penalty w.r.t. positions
                    grad_pos[i, 0] += lambda_overlap * penalty * nx
                    grad_pos[i, 1] += lambda_overlap * penalty * ny
                    grad_pos[j, 0] -= lambda_overlap * penalty * nx
                    grad_pos[j, 1] -= lambda_overlap * penalty * ny
                    
                    # Gradient w.r.t. radii
                    grad_rad[i] -= lambda_overlap * penalty
                    grad_rad[j] -= lambda_overlap * penalty
        
        # Penalty for boundary violations
        for i in range(n):
            r = radii[i]
            # Left boundary
            if pos[i, 0] < r:
                penalty = r - pos[i, 0]
                grad_pos[i, 0] += lambda_boundary * penalty
                grad_rad[i] -= lambda_boundary * penalty
            # Right boundary
            if pos[i, 0] > 1 - r:
                penalty = pos[i, 0] - (1 - r)
                grad_pos[i, 0] -= lambda_boundary * penalty
                grad_rad[i] -= lambda_boundary * penalty
            # Bottom boundary
            if pos[i, 1] < r:
                penalty = r - pos[i, 1]
                grad_pos[i, 1] += lambda_boundary * penalty
                grad_rad[i] -= lambda_boundary * penalty
            # Top boundary
            if pos[i, 1] > 1 - r:
                penalty = pos[i, 1] - (1 - r)
                grad_pos[i, 1] -= lambda_boundary * penalty
                grad_rad[i] -= lambda_boundary * penalty
        
        # Update with momentum
        vel_pos = beta * vel_pos + (1 - beta) * grad_pos
        vel_rad = beta * vel_rad + (1 - beta) * grad_rad
        
        pos += lr_pos * vel_pos
        radii += lr_rad * vel_rad
        
        # Clamp to valid ranges
        for i in range(n):
            radii[i] = max(0.01, min(0.49, radii[i]))
            pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
            pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))
        
        # Track best valid solution
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Adaptive learning rate: decrease if no improvement
        if it % 100 == 0:
            lr_pos *= 0.95
            lr_rad *= 0.95
        
        # Early stopping if gradients are very small
        if np.max(np.abs(grad_pos)) < 1e-8 and np.max(np.abs(grad_rad)) < 1e-8:
            break
    
    return best_pos, best_radii


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
    # Corners
    pos_list.append([r_corner, r_corner])
    pos_list.append([1 - r_corner, r_corner])
    pos_list.append([r_corner, 1 - r_corner])
    pos_list.append([1 - r_corner, 1 - r_corner])
    
    # Edge midpoints
    r_edge = 0.1
    pos_list.append([0.5, r_edge])
    pos_list.append([0.5, 1 - r_edge])
    pos_list.append([r_edge, 0.5])
    pos_list.append([1 - r_edge, 0.5])
    
    # Fill remaining in a grid pattern in the center
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
    deadline = 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Multiple initial configurations
    configs = []
    
    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))
    
    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))
    
    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))
    
    # Process each configuration with gradient optimization
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        remaining = deadline - elapsed - 2
        time_alloc = min(25.0, remaining * 0.7)
        
        pos, radii = _gradient_optimize(pos, radii, time_budget=time_alloc,
                                         start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    # Final polish with gradient if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 2
        if remaining > 5:
            best_pos, best_radii = _gradient_optimize(best_pos, best_radii,
                                                       time_budget=min(15.0, remaining),
                                                       start_time=start_time)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
    
    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
