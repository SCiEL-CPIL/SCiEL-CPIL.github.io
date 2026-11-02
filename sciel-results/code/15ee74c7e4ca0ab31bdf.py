"""Improved packer for 50 equal circles in the unit square [-1/2, 1/2]^2."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # seconds, safety margin before 120s kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - _elapsed()


def _score(pos):
    """Compute the largest common radius for given centers."""
    # pos shape (N, 2)
    # pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # distance to walls
    walls = 0.5 - np.abs(pos)
    min_wall = np.min(walls)
    return min(min_pair, min_wall)


def _neg_score(pos):
    return -_score(pos)


def _neg_score_grad(pos):
    """Gradient of negative score (i.e., gradient of -min(pair_dist/2, wall_dist))."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    walls = 0.5 - np.abs(pos)
    min_wall = np.min(walls)
    min_pair = np.min(dists) / 2.0
    
    r = min(min_pair, min_wall)
    
    grad = np.zeros_like(pos)
    
    if min_pair <= min_wall:
        # The bottleneck is a pair
        idx = np.argmin(dists)
        i, j = idx // n, idx % n
        d = dists[i, j]
        if d < 1e-12:
            # Degenerate: push apart
            grad[i] += np.array([1.0, 0.0])
            grad[j] -= np.array([1.0, 0.0])
        else:
            diff_ij = pos[i] - pos[j]
            unit = diff_ij / d
            grad[i] -= unit / 2.0
            grad[j] += unit / 2.0
    else:
        # The bottleneck is a wall
        idx = np.argmin(walls)
        i = idx // 2
        k = idx % 2
        # wall constraint: 0.5 - |pos[i,k]| is minimized
        # gradient of -(0.5 - |pos[i,k]|) w.r.t. pos[i,k]
        # d/dx (|x|) = sign(x)
        sign = np.sign(pos[i, k])
        if sign == 0:
            sign = 1.0
        grad[i, k] += sign  # we want to maximize the wall distance, so push away from wall
    
    return -grad  # gradient of -score


def _hexagonal_init():
    """Generate a hexagonal close-packing initial configuration."""
    # 50 circles: try 7x7 grid with hex offset, take first 50
    # Or better: use a known good arrangement
    # Let's try a 7-column layout with hexagonal offsets
    cols = 7
    rows_needed = int(np.ceil(N / cols))
    
    # Hexagonal spacing
    # We want to fit in [-0.5, 0.5]^2
    # Let's parameterize by radius and optimize
    # For now, generate a reasonable hexagonal pattern
    
    # Use a 7x8 grid with hex offset, take 50 points
    x_coords = []
    y_coords = []
    
    # Try to create a hexagonal pattern that fits
    # 7 columns, alternating rows offset
    n_cols = 7
    n_rows = int(np.ceil(N / n_cols))
    
    # Estimate spacing
    # Width = (n_cols - 1) * dx, Height = (n_rows - 1) * dy
    # For hex: dx = 2*r, dy = sqrt(3)*r
    # Let's just place them in a grid first and let optimization fix it
    
    points = []
    for row in range(n_rows):
        for col in range(n_cols):
            if len(points) >= N:
                break
            x = (col - (n_cols - 1) / 2.0) * 0.14
            y = (row - (n_rows - 1) / 2.0) * 0.14
            # Hex offset for odd rows
            if row % 2 == 1:
                x += 0.07
            points.append([x, y])
    
    return np.array(points[:N])


def _random_init(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    # Random points in [-0.4, 0.4]^2 to leave room for walls
    return rng.uniform(-0.4, 0.4, size=(N, 2))


def _local_search(pos, max_iters=500):
    """Simple gradient-based local search."""
    best_pos = pos.copy()
    best_score = _score(pos)
    
    step_size = 0.01
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        score = _score(pos)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
        
        # Compute gradient
        grad = _neg_score_grad(pos)
        
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            break
        grad = grad / norm
        
        # Take a step
        new_pos = pos - step_size * grad
        
        # Check if it's better
        new_score = _score(new_pos)
        if new_score > score:
            pos = new_pos
            # Reduce step size slightly
            step_size *= 0.99
        else:
            # Reduce step size and try again
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_pos


def _polish(pos):
    """High-precision polish using L-BFGS-B with smooth surrogate."""
    # Use a soft-min surrogate for smoother optimization
    def soft_neg_score(x):
        pos = x.reshape(N, 2)
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_terms = dists / 2.0
        
        # Wall distances
        walls = 0.5 - np.abs(pos)
        
        # Soft-min of all terms
        all_terms = np.concatenate([pair_terms.flatten(), walls.flatten()])
        
        # Use log-sum-exp for soft-min: -logsumexp(-x)
        x = -all_terms
        x_max = np.max(x)
        if x_max > 500:  # Avoid overflow
            x_max = 500
        log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
        soft_min = -log_sum_exp
        
        return soft_min
    
    # L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        soft_neg_score,
        pos.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    polished_pos = result.x.reshape(N, 2)
    return polished_pos


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_pos = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = []
    
    # Hexagonal initialization
    inits.append(_hexagonal_init())
    
    # Random initializations
    for seed in range(10):
        if _remaining() < 10.0:
            break
        inits.append(_random_init(seed))
    
    # Also try a simple grid as fallback
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    for init_pos in inits:
        if _remaining() < 10.0:
            break
        
        # Local search phase
        ls_pos = _local_search(init_pos, max_iters=300)
        ls_score = _score(ls_pos)
        
        if ls_score > best_score:
            best_score = ls_score
            best_pos = ls_pos.copy()
        
        # Polish phase if time permits
        if _remaining() > 15.0:
            try:
                polished = _polish(ls_pos)
                pol_score = _score(polished)
                
                if pol_score > best_score:
                    best_score = pol_score
                    best_pos = polished.copy()
            except Exception:
                pass
    
    # Final check: ensure validity
    if best_pos is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Ensure all values are finite and within bounds
    best_pos = np.clip(best_pos, -0.5, 0.5)
    best_pos[~np.isfinite(best_pos)] = 0.0
    
    return best_pos
