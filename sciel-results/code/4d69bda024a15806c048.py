"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

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


def _energy(centres, r_target=0.03):
    """Energy function: penalize pairs closer than 2*r_target and wall violations."""
    x = centres[:, 0]
    y = centres[:, 1]
    
    # Pairwise repulsion for close pairs
    dists = pdist(centres)
    # Only penalize pairs that are too close
    close_mask = dists < 2.0 * r_target
    if np.any(close_mask):
        close_dists = dists[close_mask]
        close_dists = np.maximum(close_dists, 1e-10)
        pair_energy = np.sum((2.0 * r_target - close_dists) ** 2)
    else:
        pair_energy = 0.0
    
    # Wall repulsion
    dx_min = x - (-0.5)
    dx_max = 0.5 - x
    dy_min = y - (-0.5)
    dy_max = 0.5 - y
    
    dist_to_walls = np.minimum(
        np.minimum(dx_min, dx_max),
        np.minimum(dy_min, dy_max)
    )
    dist_to_walls = np.maximum(dist_to_walls, 1e-10)
    
    # Penalize circles too close to walls
    wall_violation = np.maximum(0, r_target - dist_to_walls)
    wall_energy = np.sum(wall_violation ** 2)
    
    return pair_energy + wall_energy


def _hex_grid_init():
    """Generate a hexagonal grid initialization for ~200 points."""
    # Hexagonal packing: horizontal spacing dx, vertical spacing dy = dx * sqrt(3)/2
    # For N points in a unit square, estimate dx
    # Area per point ~ dx * dy = dx^2 * sqrt(3)/2
    # N * dx^2 * sqrt(3)/2 ≈ 1 => dx ≈ sqrt(2/(N*sqrt(3)))
    dx = np.sqrt(2.0 / (N * np.sqrt(3)))
    dy = dx * np.sqrt(3) / 2.0
    
    pts = []
    # Start from bottom-left corner with offset
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
    
    # Adjust to get exactly N points
    if len(pts) > N:
        # Keep the most central N points
        centers = np.mean(pts, axis=0)
        dists_to_center = np.linalg.norm(pts - centers, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        # Add random points
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.4, 0.4, size=2)])
    
    return pts


def _initial_grid():
    """Create initial square grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Create random initial placement inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9


def _optimize_with_repulsion(centres, r_target, deadline):
    """Optimize using repulsion-based energy minimization with L-BFGS-B."""
    start_time = time.time()
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        c = x.reshape(-1, 2)
        return _energy(c, r_target)
    
    try:
        result = minimize(
            objective,
            centres.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < np.inf:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centres


def _local_refinement(centres, deadline):
    """Local refinement: focus on bottleneck pairs and wall contacts."""
    start_time = time.time()
    current = centres.copy()
    current_radius = _compute_radius(current)
    
    for iteration in range(50):
        if time.time() > deadline - 2:
            break
        
        # Find the minimum distance
        dists = pdist(current)
        min_idx = np.argmin(dists)
        
        # Convert flat index to pair index
        # pdist uses condensed form: index = i*N - i*(i+1)/2 + (j-i-1) for i<j
        # Simpler: use scipy's squareform or just work with the matrix
        from scipy.spatial.distance import squareform
        dist_matrix = squareform(dists)
        np.fill_diagonal(dist_matrix, np.inf)
        
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        min_dist = dist_matrix[i, j]
        
        # Also check wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - np.abs(current[:, 0]), 0.5 - np.abs(current[:, 1])),
            0.5
        )
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # If wall is the bottleneck, push that point inward
        if min_wall_dist < min_dist / 2.0:
            # Push point away from nearest wall
            x, y = current[min_wall_idx]
            # Determine which wall is closest
            d_left = x - (-0.5)
            d_right = 0.5 - x
            d_bottom = y - (-0.5)
            d_top = 0.5 - y
            
            min_wall_d = min(d_left, d_right, d_bottom, d_top)
            if min_wall_d == d_left:
                current[min_wall_idx, 0] += 0.001
            elif min_wall_d == d_right:
                current[min_wall_idx, 0] -= 0.001
            elif min_wall_d == d_bottom:
                current[min_wall_idx, 1] += 0.001
            else:
                current[min_wall_idx, 1] -= 0.001
        else:
            # Push the closest pair apart
            d = current[i] - current[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(2) * 1e-10
                dist = np.linalg.norm(d)
            else:
                d = d / dist
            
            # Move each point by a small amount
            step = 0.001
            current[i] += step * d
            current[j] -= step * d
        
        # Clamp to square
        current[:, 0] = np.clip(current[:, 0], -0.5, 0.5)
        current[:, 1] = np.clip(current[:, 1], -0.5, 0.5)
        
        new_radius = _compute_radius(current)
        if new_radius > current_radius:
            current_radius = new_radius
        else:
            # Revert if it didn't help
            current[i] -= step * d if min_wall_dist >= min_dist / 2.0 else 0
            current[j] += step * d if min_wall_dist >= min_dist / 2.0 else 0
            # Re-clamp
            current[:, 0] = np.clip(current[:, 0], -0.5, 0.5)
            current[:, 1] = np.clip(current[:, 1], -0.5, 0.5)
    
    return current


def construct():
    """Construct 200 circle centres in the unit square to maximize common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_centres = _hex_grid_init()
    best_radius = _compute_radius(best_centres)
    
    np.random.seed(42)
    
    # Strategy: multiple restarts with different initializations
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() > deadline - 15:  # Stop if less than 15s left
            break
        
        # Different initialization strategies
        if restart == 0:
            centres = _hex_grid_init()
        elif restart == 1:
            centres = _initial_grid()
        elif restart == 2:
            centres = _random_initial()
        elif restart == 3:
            # Perturbed hex grid
            centres = _hex_grid_init() + np.random.randn(N, 2) * 0.005
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 4:
            # Perturbed best solution
            centres = best_centres + np.random.randn(N, 2) * 0.01
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        else:
            centres = _random_initial()
        
        # Optimize using repulsion-based method with adaptive r_target
        # Start with a conservative r_target and increase
        r_target = max(0.025, best_radius * 0.95)
        
        if time.time() < deadline - 10:
            centres = _optimize_with_repulsion(centres, r_target, deadline - 10)
        
        # Check radius
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
        
        # Local refinement if time permits
        if time.time() < deadline - 5 and current_radius > best_radius * 0.99:
            refined = _local_refinement(centres, deadline - 5)
            refined_radius = _compute_radius(refined)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centres = refined.copy()
    
    # Final verification: ensure all centres are strictly inside
    best_centres[:, 0] = np.clip(best_centres[:, 0], -0.5, 0.5)
    best_centres[:, 1] = np.clip(best_centres[:, 1], -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
