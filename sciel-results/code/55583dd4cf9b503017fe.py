"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree


def _compute_score_fast(centers, tree=None):
    """
    Compute the radius r = min(min_pair_dist/2, min_wall_dist).
    Uses KD-tree for efficient nearest-neighbor queries.
    """
    n = len(centers)
    if n < 2:
        return 0.5
    
    # Build or reuse KD-tree
    if tree is None:
        tree = KDTree(centers)
    
    # Find nearest neighbor for each point (excluding self)
    # query k=2 to get self + nearest
    dists, _ = tree.query(centers, k=2)
    # dists[:, 1] is the distance to the nearest other point
    min_pair = np.min(dists[:, 1]) / 2.0
    
    # Wall distances
    min_wall = np.min(0.5 - np.abs(centers))
    
    return min(min_pair, min_wall)


def _simulated_annealing(centers_init, n_iterations, T_start, T_end, base_step, rng):
    """
    Run simulated annealing from an initial configuration.
    Returns (best_centers, best_score).
    """
    centers = centers_init.copy()
    
    # Build initial KD-tree
    tree = KDTree(centers)
    
    current_score = _compute_score_fast(centers, tree)
    best_centers = centers.copy()
    best_score = current_score
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    step_size = base_step
    no_improve_count = 0
    max_no_improve = 3000
    
    for it in range(n_iterations):
        T = T_start * (alpha ** it)
        
        # Pick a random center to perturb
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Random perturbation
        centers[i] += rng.normal(0, step_size, 2)
        
        # Clip to stay within bounds
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        # Update KD-tree incrementally: remove old point, add new point
        # For efficiency, rebuild tree periodically or use a simple approach
        # Here we rebuild every few iterations for simplicity, but check score incrementally
        # Actually, for 50 points, rebuilding is fast enough
        new_tree = KDTree(centers)
        new_score = _compute_score_fast(centers, new_tree)
        
        if new_score >= current_score:
            current_score = new_score
            tree = new_tree
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta / T > -100:  # Avoid underflow
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    tree = new_tree
                    no_improve_count = 0
                else:
                    # Reject: revert
                    centers[i] = old_pos
                    no_improve_count += 1
            else:
                # Reject: revert
                centers[i] = old_pos
                no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > max_no_improve:
            step_size *= 1.15
            no_improve_count = 0
        elif it % 1000 == 0 and it > 0:
            step_size *= 0.97
    
    return best_centers, best_score


def _hexagonal_init(rng, jitter=0.0):
    """
    Generate a hexagonal-like initial configuration with optional jitter.
    """
    centers = np.zeros((N, 2))
    
    # Hexagonal packing: rows of circles
    # For 50 circles, try different row configurations
    # e.g., rows of 7, 8, 7, 8, 7, 7 = 44... not quite
    # Let's use a grid-like hexagonal arrangement
    
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    
    idx = 0
    y = -0.48
    row = 0
    while idx < N and y <= 0.48:
        x_offset = 0.0 if row % 2 == 0 else spacing / 2
        x = -0.48 + x_offset
        while x <= 0.48 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += spacing
        y += row_spacing
        row += 1
    
    # Fill remaining if any
    if idx < N:
        remaining = N - idx
        side = int(np.ceil(np.sqrt(remaining)))
        ticks = (np.arange(side) + 0.5) / side - 0.5
        for i in range(remaining):
            centers[idx + i] = [ticks[i % side], ticks[i // side]]
    
    # Add jitter if specified
    if jitter > 0:
        centers += rng.normal(0, jitter, (N, 2))
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers


def _random_init(rng):
    """
    Generate a random initial configuration.
    """
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    return centers


def _grid_init(rng, jitter=0.0):
    """
    Generate a grid initial configuration with optional jitter.
    """
    # 7x8 grid = 56, use 50
    rows = 7
    cols = 8
    centers = np.zeros((N, 2))
    
    x_ticks = np.linspace(-0.45, 0.45, cols)
    y_ticks = np.linspace(-0.45, 0.45, rows)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    
    # Add jitter if specified
    if jitter > 0:
        centers += rng.normal(0, jitter, (N, 2))
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers


def construct():
    """
    Use simulated annealing with multiple restarts to maximize the minimum
    distance between circles and from walls for 50 equal circles in a unit
    square [-0.5, 0.5]^2.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|)).
    We want to maximize r.
    """
    rng = np.random.default_rng(42)
    
    best_overall = None
    best_overall_score = -np.inf
    
    # Multiple restarts with different initializations and parameters
    n_restarts = 3
    
    for restart in range(n_restarts):
        # Choose initialization strategy
        if restart == 0:
            # Hexagonal with small jitter
            centers_init = _hexagonal_init(rng, jitter=0.005)
            n_iters = 80000
            T_start = 0.015
            T_end = 1e-9
            base_step = 0.03
        elif restart == 1:
            # Grid with jitter
            centers_init = _grid_init(rng, jitter=0.01)
            n_iters = 80000
            T_start = 0.015
            T_end = 1e-9
            base_step = 0.035
        else:
            # Hexagonal with more jitter
            centers_init = _hexagonal_init(rng, jitter=0.02)
            n_iters = 80000
            T_start = 0.018
            T_end = 1e-9
            base_step = 0.04
        
        best_centers, best_score = _simulated_annealing(
            centers_init, n_iters, T_start, T_end, base_step, rng
        )
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers
    
    return best_overall
# EVOLVE-BLOCK-END
