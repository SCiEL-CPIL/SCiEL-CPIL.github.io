"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Smooth differentiable penalty objective (soft-max collisions) optimized by L-BFGS-B from diverse starts.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

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

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0, radii - pos[:, 0]) + np.maximum(0, radii - (1.0 - pos[:, 0])) + \
         np.maximum(0, radii - pos[:, 1]) + np.maximum(0, radii - (1.0 - pos[:, 1]))
    bv = np.sum(bv)
    
    # Circle-circle violations
    cc_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = radii[i] + radii[j] - dist
            cc_viol += _soft_max(viol, beta)
    
    return -np.sum(radii) + penalty_weight * (bv + cc_viol)

def _objective_grad(x, penalty_weight=100.0, beta=50.0):
    """Gradient of the objective."""
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    grad = np.zeros_like(x)
    
    # Gradient of -sum(radii)
    grad[2*N:] = -1.0
    
    # Gradient of boundary penalty
    # d/dx_i (r_i - x_i) if x_i < r_i -> -1
    # d/dx_i (r_i - (1-x_i)) if x_i > 1-r_i -> 1
    # Similarly for y
    for i in range(N):
        r = radii[i]
        x, y = pos[i]
        if x < r:
            grad[2*i] -= penalty_weight
        if x > 1.0 - r:
            grad[2*i] += penalty_weight
        if y < r:
            grad[2*i+1] -= penalty_weight
        if y > 1.0 - r:
            grad[2*i+1] += penalty_weight
        
        # Gradient of r in boundary terms
        if x < r:
            grad[2*N+i] += penalty_weight
        if x > 1.0 - r:
            grad[2*N+i] += penalty_weight
        if y < r:
            grad[2*N+i] += penalty_weight
        if y > 1.0 - r:
            grad[2*N+i] += penalty_weight

    # Gradient of circle-circle penalty
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-8:
                continue
            viol = radii[i] + radii[j] - dist
            if viol > 0:
                # Derivative of soft_max(viol) is sigmoid(beta * viol)
                sig = 1.0 / (1.0 + np.exp(-beta * viol))
                # d(viol)/dx_i = -dx/dist
                # d(viol)/dx_j = dx/dist
                # d(viol)/dr_i = 1
                # d(viol)/dr_j = 1
                factor = penalty_weight * sig
                grad[2*i] += factor * (-dx / dist)
                grad[2*i+1] += factor * (-dy / dist)
                grad[2*j] += factor * (dx / dist)
                grad[2*j+1] += factor * (dy / dist)
                grad[2*N+i] += factor
                grad[2*N+j] += factor
    return grad

def _init_hexagonal():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # Approximate hexagonal grid
    r0 = 0.1
    dx = 2 * r0 * np.sqrt(3) / 2
    dy = 2 * r0
    k = 0
    for row in range(5):
        for col in range(5):
            if k >= N:
                break
            x = 0.1 + col * dx + (row % 2) * dx / 2
            y = 0.1 + row * dy
            if x < 0.95 and y < 0.95:
                pos[k] = [x, y]
                radii[k] = r0
                k += 1
            if k >= N:
                break
    # Fill remaining
    rng = np.random.RandomState(1)
    while k < N:
        pos[k] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
        radii[k] = 0.05
        k += 1
    return pos, radii

def _init_grid():
    """5x5 grid minus center, plus center small."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            if k >= N - 1:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.09
            k += 1
    pos[k] = [0.5, 0.5]
    radii[k] = 0.03
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.1, N)
    return pos, radii

def _init_corner_biased():
    """Corner + edge biased initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    rng = np.random.RandomState(42)
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.15
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5),
             (0.3, 0.08), (0.7, 0.08), (0.3, 0.92), (0.7, 0.92),
             (0.08, 0.3), (0.08, 0.7), (0.92, 0.3), (0.92, 0.7)]
    for i in range(4, N):
        if i - 4 < len(edges):
            pos[i] = edges[i - 4]
            radii[i] = 0.10
        else:
            pos[i] = [rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)]
            radii[i] = 0.05
    return pos, radii

def _lbgfbs_b_optimize(pos, radii, deadline, seed=0):
    """L-BFGS-B optimization from a single start."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
        
    # Adaptive penalty weight
    penalty_weight = 100.0
    beta = 50.0
    
    try:
        result = minimize(
            _objective, x0, method='L-BFGS-B', jac=_objective_grad,
            args=(penalty_weight, beta), bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if result.success or result.fun < _objective(x0, penalty_weight, beta) - 1e-8:
            x = result.x
            pos_out = x[:2 * N].reshape(N, 2)
            radii_out = np.maximum(x[2 * N:], 1e-6)
            return pos_out, radii_out
    except Exception:
        pass
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corner_biased(),
        _init_random(1),
        _init_random(2),
        _init_random(3)
    ]

    best_pos, best_radii = inits[0]
    best_score = np.sum(best_radii)

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Post-process to ensure validity before optimization
        pos_pp, rad_pp = _post_process(pos_init, rad_init)
        pos_pp, rad_pp = _lbgfbs_b_optimize(pos_pp, rad_pp, deadline, seed=idx)
        pos_final, rad_final = _post_process(pos_pp, rad_pp)
        
        score = np.sum(rad_final)
        if _is_valid(pos_final, rad_final) and score > best_score:
            best_score = score
            best_pos = pos_final.copy()
            best_radii = rad_final.copy()

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
