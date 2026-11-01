import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _score(c):
    """Compute the score (radius) for a given configuration c (N, 2)."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_x = np.min(c[:, 0]) + 0.5
    max_x = 0.5 - np.max(c[:, 0])
    min_y = np.min(c[:, 1]) + 0.5
    max_y = 0.5 - np.max(c[:, 1])
    min_wall_dist = np.min([min_x, max_x, min_y, max_y])
    
    return np.min([min_pair_dist / 2.0, min_wall_dist])

def _compute_radius(c):
    """Compute the maximum valid radius for configuration c."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_x = np.min(c[:, 0]) + 0.5
    max_x = 0.5 - np.max(c[:, 0])
    min_y = np.min(c[:, 1]) + 0.5
    max_y = 0.5 - np.max(c[:, 1])
    min_wall_dist = np.min([min_x, max_x, min_y, max_y])
    
    return np.min([min_pair_dist / 2.0, min_wall_dist])

def _hex_init(r_init=0.035):
    """Generate hexagonal lattice initialization."""
    points = []
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    i = 0
    while len(points) < N + 50:
        y = -0.5 + r_init + i * dy
        if y > 0.5 - r_init:
            break
        j = 0
        while True:
            x = -0.5 + r_init + j * dx + (i % 2) * (dx / 2.0)
            if x > 0.5 - r_init:
                break
            points.append([x, y])
            j += 1
        i += 1
    
    points = np.array(points[:N])
    
    if len(points) < N:
        extra = np.random.uniform(-0.4, 0.4, size=(N - len(points), 2))
        points = np.vstack([points, extra])
    
    return points

def _gradient_step(c, r, delta=1e-4):
    """Compute gradient for one step of gradient ascent."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_x = c[:, 0] + 0.5
    max_x = 0.5 - c[:, 0]
    min_y = c[:, 1] + 0.5
    max_y = 0.5 - c[:, 1]
    
    grad = np.zeros_like(c)
    
    # Pair constraints
    mask = dists < (2.0 * r + delta)
    idx_i, idx_j = np.where(mask)
    idx_i = idx_i[idx_i < idx_j]
    idx_j = idx_j[idx_i < idx_j]
    
    for i, j in zip(idx_i, idx_j):
        d = dists[i, j]
        if d < 1e-9:
            vec = np.random.randn(2)
            vec /= (np.linalg.norm(vec) + 1e-9)
        else:
            vec = (c[i] - c[j]) / d
        grad[i] += vec
        grad[j] -= vec
    
    # Wall constraints
    mask_l = min_x < (r + delta)
    grad[mask_l, 0] += 1.0
    mask_r = max_x < (r + delta)
    grad[mask_r, 0] -= 1.0
    mask_b = min_y < (r + delta)
    grad[mask_b, 1] += 1.0
    mask_t = max_y < (r + delta)
    grad[mask_t, 1] -= 1.0
    
    # Normalize per point
    norms = np.linalg.norm(grad, axis=1)
    mask_nonzero = norms > 1e-9
    grad[mask_nonzero] /= norms[mask_nonzero, None]
    
    return grad

def _line_search(c, grad, r, alpha_max, max_trials=5):
    """Simple backtracking line search."""
    for trial in range(max_trials):
        alpha = alpha_max / (2 ** trial)
        c_new = c + alpha * grad
        c_new[:, 0] = np.clip(c_new[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        c_new[:, 1] = np.clip(c_new[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        r_new = _compute_radius(c_new)
        if r_new >= r - 1e-12:
            return c_new, r_new, alpha
    return c, r, alpha_max / (2 ** (max_trials - 1))

def _optimize(c, alpha_init, max_steps, deadline, delta_init=1e-4):
    """Main optimization loop with line search."""
    r = _compute_radius(c)
    best_r = r
    best_c = c.copy()
    alpha = alpha_init
    delta = delta_init
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        grad = _gradient_step(c, r, delta)
        
        # Check if we're stuck
        if np.allclose(grad, 0, atol=1e-12):
            break
        
        c_new, r_new, alpha_used = _line_search(c, grad, r, alpha)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        
        # Update
        c = c_new
        r = r_new
        alpha = alpha_used * 1.5  # Try slightly larger next time
        
        # Adapt delta
        if step % 300 == 299:
            delta *= 0.7
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Phase 1: Coarse optimization
    c = _hex_init(0.035)
    c, r = _optimize(c, alpha_init=5e-3, max_steps=2000, deadline=deadline)
    
    if time.time() > deadline - 15:
        return c
    
    # Phase 2: Medium optimization
    c, r = _optimize(c, alpha_init=1e-3, max_steps=2000, deadline=deadline)
    
    if time.time() > deadline - 8:
        return c
    
    # Phase 3: Fine optimization
    c, r = _optimize(c, alpha_init=1e-4, max_steps=2000, deadline=deadline)
    
    if time.time() > deadline - 3:
        return c
    
    # Phase 4: Ultra-fine polish
    c, r = _optimize(c, alpha_init=1e-5, max_steps=1000, deadline=deadline, delta_init=1e-6)
    
    return c

if __name__ == "__main__":
    res = construct()
    print("Score:", _score(res))
