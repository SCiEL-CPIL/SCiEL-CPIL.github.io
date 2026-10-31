"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 140  # seconds, safety margin before 150s kill


def _elapsed_ok(start_time):
    return (time.time() - start_time) < DEADLINE


def _evaluate(circles, W):
    """
    circles: (N, 3) array of [x, y, r]
    W: width of rectangle, H = 2 - W
    Returns: (sum_radii, penalty)
    penalty = sum of overlap violations + boundary violations
    """
    H = 2.0 - W
    n = circles.shape[0]
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Boundary penalties
    # Circle must be inside [0, W] x [0, H]
    boundary_pen = 0.0
    boundary_pen += np.sum(np.maximum(0, r - x))  # left
    boundary_pen += np.sum(np.maximum(0, r - (W - x)))  # right
    boundary_pen += np.sum(np.maximum(0, r - y))  # bottom
    boundary_pen += np.sum(np.maximum(0, r - (H - y)))  # top

    # Overlap penalties: for each pair, if dist < r_i + r_j, penalty = (r_i + r_j - dist)
    # Vectorized pairwise
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = r[:, None] + r[None, :]
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    overlaps = np.maximum(0, min_dist[iu] - dist[iu])
    overlap_pen = np.sum(overlaps)

    total_pen = boundary_pen + overlap_pen
    sum_r = np.sum(r)
    return sum_r, total_pen


def _objective(circles, W):
    """Return negative sum_radii + penalty (to minimize)"""
    sum_r, pen = _evaluate(circles, W)
    return -(sum_r) + 10.0 * pen


def _initial_packing(W, seed=42):
    """Grid-like initial packing"""
    H = 2.0 - W
    rng = np.random.RandomState(seed)
    
    # Try to fit circles in a grid
    # Estimate number of circles per row
    # For 21 circles, try 5x5 grid (25 cells, use 21)
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / side * 0.45
            cells.append([cx, cy, r])
    
    cells = np.array(cells[:N])
    # Add small random perturbation
    cells[:, 0] += rng.uniform(-0.01, 0.01, N)
    cells[:, 1] += rng.uniform(-0.01, 0.01, N)
    # Clamp radii to positive
    cells[:, 2] = np.maximum(0.01, cells[:, 2])
    return cells


def _sa_optimize(W, start_time, max_iter=50000):
    """Simulated annealing for fixed aspect ratio W"""
    H = 2.0 - W
    circles = _initial_packing(W, seed=int(np.random.randint(0, 1000000)))
    
    # Initial evaluation
    sum_r, pen = _evaluate(circles, W)
    energy = _objective(circles, W)
    best_energy = energy
    best_circles = circles.copy()
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.9995
    
    rng = np.random.RandomState(int(np.random.randint(0, 1000000)))
    
    for it in range(max_iter):
        if not _elapsed_ok(start_time):
            break
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move a circle
            idx = rng.randint(N)
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            new_circles = circles.copy()
            new_circles[idx, 0] = np.clip(new_circles[idx, 0] + dx, 0, W)
            new_circles[idx, 1] = np.clip(new_circles[idx, 1] + dy, 0, H)
        elif move_type < 0.7:
            # Change a radius
            idx = rng.randint(N)
            dr = rng.normal(0, 0.02)
            new_circles = circles.copy()
            new_circles[idx, 2] = np.maximum(0.005, new_circles[idx, 2] + dr)
        elif move_type < 0.9:
            # Move multiple circles slightly
            for idx in rng.choice(N, size=3, replace=False):
                dx = rng.normal(0, 0.03)
                dy = rng.normal(0, 0.03)
                circles[idx, 0] = np.clip(circles[idx, 0] + dx, 0, W)
                circles[idx, 1] = np.clip(circles[idx, 1] + dy, 0, H)
            new_circles = circles
        else:
            # Perturb all radii slightly
            new_circles = circles.copy()
            new_circles[:, 2] = np.maximum(0.005, new_circles[:, 2] + rng.normal(0, 0.01, N))
        
        new_energy = _objective(new_circles, W)
        delta = new_energy - energy
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            circles = new_circles
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_circles = circles.copy()
        
        T = max(T * cooling, T_min)
    
    return best_circles, best_energy


def _optimize_aspect_ratio(start_time):
    """Optimize over aspect ratio using multiple starts"""
    best_overall = None
    best_score = -np.inf
    
    # Try a range of aspect ratios
    # W from 0.5 to 1.5 (symmetric around 1.0)
    W_values = np.linspace(0.6, 1.4, 9)
    
    for W in W_values:
        if not _elapsed_ok(start_time):
            break
        
        # Run SA for this aspect ratio
        circles, energy = _sa_optimize(W, start_time, max_iter=20000)
        
        # Check validity
        sum_r, pen = _evaluate(circles, W)
        if pen < 1e-6:  # valid
            score = sum_r
            if score > best_score:
                best_score = score
                best_overall = (W, circles)
        
        # Also try with slightly different seeds
        if _elapsed_ok(start_time):
            circles2, energy2 = _sa_optimize(W, start_time, max_iter=15000)
            sum_r2, pen2 = _evaluate(circles2, W)
            if pen2 < 1e-6:
                score2 = sum_r2
                if score2 > best_score:
                    best_score = score2
                    best_overall = (W, circles2)
    
    return best_overall, best_score


def construct():
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    best_result, best_score = _optimize_aspect_ratio(start_time)
    
    if best_result is None:
        # Fallback: grid packing with W=1
        W = 1.0
        circles = _initial_packing(W, seed=42)
        # Clamp to ensure validity
        H = 2.0 - W
        for i in range(N):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
        return circles
    
    W, circles = best_result
    H = 2.0 - W
    
    # Final clamping to ensure validity
    for i in range(N):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
    
    # Verify final result
    sum_r, pen = _evaluate(circles, W)
    if pen > 1e-4:
        # Fallback to grid
        W = 1.0
        circles = _initial_packing(W, seed=42)
        H = 2.0 - W
        for i in range(N):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], W - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], H - circles[i, 2])
    
    return circles
# EVOLVE-BLOCK-END
