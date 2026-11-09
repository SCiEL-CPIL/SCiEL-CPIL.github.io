"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 140.0  # safety margin before 150s kill


def _score(pts):
    """Compute the radius: min of half min-pair-dist and min distance to boundary."""
    n = len(pts)
    # Pairwise distances
    dx = pts[:, 0][:, None] - pts[:, 0][None, :]
    dy = pts[:, 1][:, None] - pts[:, 1][None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    # Distance to boundary
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    min_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top)).min()
    return min(min_pair, min_bound)


def _make_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _make_honeycomb():
    """Honeycomb-like initial config."""
    pts = np.zeros((N, 2))
    # Approximate hexagonal packing
    spacing = 0.12
    row = 0
    idx = 0
    y = -0.45
    while idx < N and y < 0.5:
        x_offset = (spacing * 0.5) if (row % 2 == 1) else 0.0
        x = -0.45 + x_offset
        col = 0
        while x < 0.5 and idx < N:
            pts[idx, 0] = x
            pts[idx, 1] = y
            idx += 1
            x += spacing
            col += 1
        y += spacing * 0.866
        row += 1
    return pts


def _make_random():
    """Random points in the square."""
    return np.random.uniform(-0.4, 0.4, (N, 2))


def _anneal(pts, time_budget, rng):
    """Simulated annealing to maximize the minimum radius."""
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t_start = time.time()
    t = 0.05  # initial temperature
    cooling = 0.9995
    step_size = 0.05
    
    while time.time() - t_start < time_budget:
        # Pick a random point to move
        i = rng.randint(0, N - 1)
        dx = (rng.random() - 0.5) * 2 * step_size
        dy = (rng.random() - 0.5) * 2 * step_size
        old_x, old_y = pts[i, 0], pts[i, 1]
        pts[i, 0] = old_x + dx
        pts[i, 1] = old_y + dy
        
        # Clamp to square
        pts[i, 0] = np.clip(pts[i, 0], -0.5, 0.5)
        pts[i, 1] = np.clip(pts[i, 1], -0.5, 0.5)
        
        new_score = _score(pts)
        
        if new_score >= current_score or rng.random() < np.exp((new_score - current_score) / t):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            pts[i, 0] = old_x
            pts[i, 1] = old_y
        
        t *= cooling
        step_size *= 0.9999
        if step_size < 0.001:
            step_size = 0.001
    
    return best_pts, best_score


def _polish(pts, time_budget, rng):
    """Local polishing: small random perturbations that must improve score."""
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t_start = time.time()
    step_size = 0.01
    
    while time.time() - t_start < time_budget:
        i = rng.randint(0, N - 1)
        dx = (rng.random() - 0.5) * 2 * step_size
        dy = (rng.random() - 0.5) * 2 * step_size
        old_x, old_y = pts[i, 0], pts[i, 1]
        pts[i, 0] = np.clip(old_x + dx, -0.5, 0.5)
        pts[i, 1] = np.clip(old_y + dy, -0.5, 0.5)
        
        new_score = _score(pts)
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            pts[i, 0] = old_x
            pts[i, 1] = old_y
        
        # Occasionally reduce step size
        if rng.random() < 0.01:
            step_size *= 0.95
            if step_size < 0.0005:
                step_size = 0.0005
    
    return best_pts, best_score


def construct():
    t_global_start = time.time()
    rng = random.Random(42)
    np.random.seed(42)
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_make_grid())
    inits.append(_make_honeycomb())
    inits.append(_make_random())
    inits.append(_make_random())
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Anneal each initial config
    n_inits = len(inits)
    for i, init in enumerate(inits):
        remaining = DEADLINE - (time.time() - t_global_start)
        if remaining < 5:
            break
        # Allocate time proportionally, with more time for later inits
        time_budget = min(remaining * 0.3, max(2.0, 8.0 - i * 1.5))
        pts, score = _anneal(init, time_budget, rng)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        return _make_grid()
    
    # Phase 2: Polish the best result
    remaining = DEADLINE - (time.time() - t_global_start)
    if remaining > 3:
        polish_time = min(remaining * 0.5, 20.0)
        best_pts, best_score = _polish(best_pts, polish_time, rng)
    
    # Phase 3: Final micro-polish
    remaining = DEADLINE - (time.time() - t_global_start)
    if remaining > 2:
        polish_time = min(remaining * 0.6, 10.0)
        best_pts, best_score = _polish(best_pts, polish_time, rng)
    
    # Ensure validity: clamp to square
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    # Final check
    final_score = _score(best_pts)
    if final_score <= 0 or not np.all(np.isfinite(best_pts)):
        return _make_grid()
    
    return best_pts
# EVOLVE-BLOCK-END
