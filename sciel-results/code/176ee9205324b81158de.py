import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _score(c):
    """Compute the score (radius) for a given configuration c (N, 2)."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
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

def _gradient_ascent(points, alpha=1e-3, max_steps=3000, deadline=None, delta=1e-4, adaptive=True):
    """
    Gradient ascent on the minimum distance (radius) objective.
    Uses active set method with adaptive step size.
    """
    c = points.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Calculate distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        min_x = c[:, 0] + 0.5
        max_x = 0.5 - c[:, 0]
        min_y = c[:, 1] + 0.5
        max_y = 0.5 - c[:, 1]
        
        # Current radius
        min_pair = np.min(dists) / 2.0
        min_wall = np.min([np.min(min_x), np.min(max_x), np.min(min_y), np.min(max_y)])
        r = np.min([min_pair, min_wall])
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Compute gradient
        grad = np.zeros_like(c)
        
        # Pair constraints - active if dist < 2*r + delta
        mask = dists < (2.0 * r + delta)
        # Extract upper triangle indices to avoid double counting and broadcasting errors
        idx_i, idx_j = np.where(np.triu(mask, k=1))
        
        for i, j in zip(idx_i, idx_j):
            d = dists[i, j]
            if d < 1e-9:
                # Random perturbation for coincident points
                vec = np.random.randn(2)
                vec /= np.linalg.norm(vec) + 1e-9
            else:
                vec = (c[i] - c[j]) / d
            grad[i] += vec
            grad[j] -= vec
        
        # Wall constraints - active if dist < r + delta
        mask_l = min_x < (r + delta)
        grad[mask_l, 0] += 1.0
        
        mask_r = max_x < (r + delta)
        grad[mask_r, 0] -= 1.0
        
        mask_b = min_y < (r + delta)
        grad[mask_b, 1] += 1.0
        
        mask_t = max_y < (r + delta)
        grad[mask_t, 1] -= 1.0
        
        # Normalize gradients per point
        norms = np.linalg.norm(grad, axis=1)
        mask_nonzero = norms > 1e-9
        grad[mask_nonzero] /= norms[mask_nonzero, None]
        
        # Adaptive step size
        if adaptive:
            # Try a few step sizes and pick the best
            step_sizes = [alpha * 0.5, alpha, alpha * 2.0]
            best_step_r = r
            best_step_c = c.copy()
            
            for ss in step_sizes:
                c_trial = c + ss * grad
                c_trial[:, 0] = np.clip(c_trial[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
                c_trial[:, 1] = np.clip(c_trial[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
                trial_r = _compute_radius(c_trial)
                if trial_r > best_step_r:
                    best_step_r = trial_r
                    best_step_c = c_trial.copy()
            
            c = best_step_c
        else:
            c = c + alpha * grad
            c[:, 0] = np.clip(c[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            c[:, 1] = np.clip(c[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Reduce delta over time for tighter convergence
        if step % 500 == 499:
            delta *= 0.7
    
    return best_c, best_r

def _polish(points, max_steps=2000, deadline=None):
    """
    High-precision polish phase with very small steps and tight tolerances.
    """
    c = points.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    delta = 1e-6
    alpha = 1e-5
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_x = c[:, 0] + 0.5
        max_x = 0.5 - c[:, 0]
        min_y = c[:, 1] + 0.5
        max_y = 0.5 - c[:, 1]
        
        min_pair = np.min(dists) / 2.0
        min_wall = np.min([np.min(min_x), np.min(max_x), np.min(min_y), np.min(max_y)])
        r = np.min([min_pair, min_wall])
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        grad = np.zeros_like(c)
        
        mask = dists < (2.0 * r + delta)
        # Extract upper triangle indices to avoid double counting and broadcasting errors
        idx_i, idx_j = np.where(np.triu(mask, k=1))
        
        for i, j in zip(idx_i, idx_j):
            d = dists[i, j]
            if d < 1e-9:
                vec = np.random.randn(2)
                vec /= np.linalg.norm(vec) + 1e-9
            else:
                vec = (c[i] - c[j]) / d
            grad[i] += vec
            grad[j] -= vec
        
        mask_l = min_x < (r + delta)
        grad[mask_l, 0] += 1.0
        mask_r = max_x < (r + delta)
        grad[mask_r, 0] -= 1.0
        mask_b = min_y < (r + delta)
        grad[mask_b, 1] += 1.0
        mask_t = max_y < (r + delta)
        grad[mask_t, 1] -= 1.0
        
        norms = np.linalg.norm(grad, axis=1)
        mask_nonzero = norms > 1e-9
        grad[mask_nonzero] /= norms[mask_nonzero, None]
        
        c = c + alpha * grad
        c[:, 0] = np.clip(c[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        c[:, 1] = np.clip(c[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Reduce step size gradually
        if step % 500 == 499:
            alpha *= 0.8
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    c, r = _gradient_ascent(_hex_init(0.035), alpha=5e-3, max_steps=2000, deadline=deadline)
    
    if time.time() > deadline - 10:
        return c
    
    # 2. Second pass with smaller alpha
    c, r = _gradient_ascent(c, alpha=1e-3, max_steps=2000, deadline=deadline)
    
    if time.time() > deadline - 5:
        return c
    
    # 3. Final high-precision polish
    c, r = _polish(c, max_steps=3000, deadline=deadline)
    
    return c

if __name__ == "__main__":
    res = construct()
    print("Score:", _score(res))
