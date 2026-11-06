"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

DEADLINE = 140.0  # Safety margin within 150s limit


def _compute_radius(centres):
    """Compute the maximum common radius for a given set of centres."""
    if len(centres) < 2:
        return 0.0
    # Minimum distance to walls
    wall_min = min(0.5 - np.min(centres[:, 0]), 0.5 - np.max(centres[:, 0]),
                   0.5 - np.min(centres[:, 1]), 0.5 - np.max(centres[:, 1]))
    # Minimum pairwise distance / 2
    dists = pdist(centres)
    min_pair_dist = np.min(dists) if len(dists) > 0 else 1.0
    return min(wall_min, min_pair_dist / 2.0)


def _hex_grid_init():
    """Generate a hexagonal grid initialization for ~200 points."""
    dx = np.sqrt(2.0 / (N * np.sqrt(3)))
    dy = dx * np.sqrt(3) / 2.0

    pts = []
    x_start = -0.5 + dx / 2.0
    y_start = -0.5 + dy / 2.0

    row = 0
    y = y_start
    while y <= 0.5 - dy / 2.0 + 1e-6:
        offset = 0.0 if row % 2 == 0 else dx / 2.0
        x = x_start + offset
        while x <= 0.5 - dx / 2.0 + 1e-6:
            pts.append([x, y])
            x += dx
        row += 1
        y += dy

    pts = np.array(pts)

    if len(pts) > N:
        centers = np.mean(pts, axis=0)
        dists_to_center = np.linalg.norm(pts - centers, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.4, 0.4, size=2)])

    return pts


def _random_initial():
    """Create random initial placement inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9


def _vectorized_soft_optimize(centres, r_target, deadline, max_iter=500):
    """
    Vectorized gradient descent with smooth hinge loss on pairwise distances
    and wall distances. Uses all 200 points simultaneously via NumPy vectorization.
    """
    n = len(centres)
    pts = centres.copy().astype(np.float64)
    pts_flat = pts.flatten().copy()
    
    # Pre-compute index pairs for vectorized operations
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Learning rate schedule
    base_lr = 0.0005
    decay = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute pairwise differences
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        dists = np.sqrt(dx * dx + dy * dy)
        dists_safe = np.maximum(dists, 1e-12)
        
        # Smooth hinge loss: penalty when dist < 2*r_target
        # Use a smooth approximation: max(0, 2r - d)^2
        violations = np.maximum(0, 2.0 * r_target - dists)
        
        # Gradient contribution from pairwise constraints
        grad = np.zeros_like(pts)
        # d/dx_i of (2r - d)^2 = -2(2r-d) * (x_i - x_j)/d
        # For point i: -2 * violations * (dx/d)
        # For point j: +2 * violations * (dx/d)
        factor = 2.0 * violations / dists_safe
        grad[i_idx, 0] -= factor * dx
        grad[i_idx, 1] -= factor * dy
        grad[j_idx, 0] += factor * dx
        grad[j_idx, 1] += factor * dy
        
        # Wall constraints: want dist_to_wall >= r_target
        # dist_to_wall = min(x+0.5, 0.5-x, y+0.5, 0.5-y)
        # Smooth approximation: use soft min via log-sum-exp or just handle each wall
        # For simplicity, use the closest wall and apply gradient
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_left = x + 0.5
        d_right = 0.5 - x
        d_bottom = y + 0.5
        d_top = 0.5 - y
        
        # Soft minimum: use a smooth approximation
        # For each point, the wall distance is the min of the four
        # We'll use a soft version: -log(exp(-a*d1)+exp(-a*d2)+exp(-a*d3)+exp(-a*d4))/a for large a
        # But simpler: just use the actual min and apply gradient to the active wall
        wall_dists = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        wall_violations = np.maximum(0, r_target - wall_dists)
        
        # Gradient from wall constraints
        for i in range(n):
            if wall_violations[i] > 0:
                # Determine which wall is closest
                min_d = min(d_left[i], d_right[i], d_bottom[i], d_top[i])
                if min_d == d_left[i]:
                    grad[i, 0] += 2.0 * wall_violations[i]
                elif min_d == d_right[i]:
                    grad[i, 0] -= 2.0 * wall_violations[i]
                elif min_d == d_bottom[i]:
                    grad[i, 1] += 2.0 * wall_violations[i]
                else:
                    grad[i, 1] -= 2.0 * wall_violations[i]
        
        # Update positions with adaptive learning rate
        lr = base_lr * (decay ** it)
        pts -= lr * grad
        
        # Clamp to square
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Early stopping if no violations
        if np.max(violations) < 1e-10 and np.max(wall_violations) < 1e-10:
            break
    
    return pts


def _local_refinement(centres, deadline):
    """
    Local refinement: focus on bottleneck pairs and wall contacts.
    Iteratively adjust the most constrained points.
    """
    current = centres.copy()
    current_radius = _compute_radius(current)
    
    for iteration in range(300):
        if time.time() > deadline - 1:
            break
        
        # Compute all pairwise distances and wall distances
        dists = pdist(current)
        dist_matrix = squareform(dists)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - np.abs(current[:, 0]), 0.5 - np.abs(current[:, 1])),
            0.5
        )
        
        # Find the bottleneck
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall is bottleneck
            idx = np.argmin(wall_dists)
            x, y = current[idx]
            d_left = x - (-0.5)
            d_right = 0.5 - x
            d_bottom = y - (-0.5)
            d_top = 0.5 - y
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            step = 0.0002
            if min_d == d_left:
                current[idx, 0] += step
            elif min_d == d_right:
                current[idx, 0] -= step
            elif min_d == d_bottom:
                current[idx, 1] += step
            else:
                current[idx, 1] -= step
            
            # Clamp
            current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_radius = _compute_radius(current)
            if new_radius > current_radius:
                current_radius = new_radius
            else:
                # Revert
                if min_d == d_left:
                    current[idx, 0] -= step
                elif min_d == d_right:
                    current[idx, 0] += step
                elif min_d == d_bottom:
                    current[idx, 1] -= step
                else:
                    current[idx, 1] += step
                current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
                current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        else:
            # Pair is bottleneck
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            d = current[i] - current[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-12
                dist = np.linalg.norm(d)
            else:
                d = d / dist
            
            # Move each point by a small amount
            step = 0.0002
            current[i] += step * d
            current[j] -= step * d
            
            # Clamp
            current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_radius = _compute_radius(current)
            if new_radius > current_radius:
                current_radius = new_radius
            else:
                # Revert
                current[i] -= step * d
                current[j] += step * d
                current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
                current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
    
    return current


def _adaptive_radius_search(centres, start_r, end_r, deadline):
    """
    Binary search on radius: find the largest r such that we can place all circles
    with minimum distance 2r and wall distance r.
    """
    low = start_r
    high = end_r
    
    best_centres = centres.copy()
    best_radius = _compute_radius(best_centres)
    
    # First, try to achieve the target radius
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        
        mid = (low + high) / 2.0
        optimized = _vectorized_soft_optimize(centres, mid, deadline - 5)
        achieved_radius = _compute_radius(optimized)
        
        if achieved_radius >= mid * 0.99:
            # Can achieve this radius, try larger
            low = mid
            best_centres = optimized
            best_radius = achieved_radius
        else:
            # Cannot achieve, try smaller
            high = mid
        
        # Update best if improved
        if achieved_radius > best_radius:
            best_radius = achieved_radius
            best_centres = optimized
    
    return best_centres, best_radius


def construct():
    """Construct 200 circle centres in the unit square to maximize common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    # Start with hex grid initialization
    best_centres = _hex_grid_init()
    best_radius = _compute_radius(best_centres)
    
    # Multiple restarts with different initializations
    num_restarts = 4
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Different initialization strategies
        if restart == 0:
            centres = _hex_grid_init()
        elif restart == 1:
            centres = _hex_grid_init() + np.random.randn(N, 2) * 0.01
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 2:
            centres = _random_initial()
        else:
            centres = best_centres + np.random.randn(N, 2) * 0.005
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        
        # Adaptive radius search
        current_radius = _compute_radius(centres)
        # Start from current radius, try to go higher
        start_r = max(0.025, current_radius * 0.9)
        end_r = min(0.04, current_radius * 1.2)
        
        refined, refined_radius = _adaptive_radius_search(centres, start_r, end_r, deadline - 10)
        
        if refined_radius > best_radius:
            best_radius = refined_radius
            best_centres = refined.copy()
        
        # Local refinement if time permits
        if time.time() < deadline - 5:
            polished = _local_refinement(refined, deadline - 5)
            polished_radius = _compute_radius(polished)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centres = polished.copy()
    
    # Final verification: ensure all centres are strictly inside
    best_centres[:, 0] = np.clip(best_centres[:, 0], -0.5, 0.5)
    best_centres[:, 1] = np.clip(best_centres[:, 1], -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
