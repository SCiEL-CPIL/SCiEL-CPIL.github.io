"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

DEADLINE = 138.0  # Safety margin within 150s limit


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


def _smooth_energy_and_grad(centres, r_target, beta=80.0):
    """
    Smooth surrogate energy with explicit gradient.
    Energy = sum over close pairs of (2r - d)^2 + sum over wall violations of (r - wd)^2
    """
    x = centres[:, 0]
    y = centres[:, 1]
    
    # Pairwise distances
    dists = pdist(centres)
    # Gradient for pairwise: for each pair (i,j), d_ij = |c_i - c_j|
    # d/dc_i of (2r - d)^2 = -2(2r - d) * (c_i - c_j)/d
    # We need the full gradient vector
    
    # Compute distance matrix
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Pairwise penalty and gradient
    violations = np.maximum(0, 2.0 * r_target - dist_mat)
    pair_penalty = np.sum(violations ** 2)
    
    # Gradient for pairwise term
    # d/dc_i sum_j (2r - d_ij)^2 for d_ij < 2r
    # = sum_j 2*(2r - d_ij) * (c_i - c_j)/d_ij  (for d_ij < 2r)
    # Note: (c_i - c_j)/d_ij is the unit vector from j to i
    
    grad = np.zeros_like(centres)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist_mat[i, j]
            if d < 2.0 * r_target and d > 1e-12:
                # Gradient contribution to c_i
                grad[i] += 2.0 * (2.0 * r_target - d) * (centres[i] - centres[j]) / d
    
    # Wall distances and penalties
    dx_left = x - (-0.5)
    dx_right = 0.5 - x
    dy_bottom = y - (-0.5)
    dy_top = 0.5 - y
    
    wall_dists = np.minimum(
        np.minimum(dx_left, dx_right),
        np.minimum(dy_bottom, dy_top)
    )
    wall_violations = np.maximum(0, r_target - wall_dists)
    wall_penalty = np.sum(wall_violations ** 2)
    
    # Wall gradient
    for i in range(N):
        wd = wall_dists[i]
        if wd < r_target:
            # Which wall is closest?
            if dx_left[i] == wd:
                grad[i, 0] += 2.0 * (r_target - wd) * (-1.0)
            elif dx_right[i] == wd:
                grad[i, 0] += 2.0 * (r_target - wd) * (1.0)
            elif dy_bottom[i] == wd:
                grad[i, 1] += 2.0 * (r_target - wd) * (-1.0)
            elif dy_top[i] == wd:
                grad[i, 1] += 2.0 * (r_target - wd) * (1.0)
    
    return pair_penalty + wall_penalty, grad


def _hex_grid_init():
    """Generate a hexagonal grid initialization for ~200 points."""
    # Target radius ~ 0.0366, so spacing ~ 2*r = 0.0732
    # Hex grid with spacing s: points per row ~ 1/s, rows ~ 1/(s*sqrt(3)/2)
    # Total ~ 1/(s^2 * sqrt(3)/2) = 2/(s^2 * sqrt(3))
    # For 200 points: s = sqrt(2/(200*sqrt(3))) = sqrt(1/(100*sqrt(3)))
    s = np.sqrt(1.0 / (N * np.sqrt(3.0) / 2.0))
    dy = s * np.sqrt(3.0) / 2.0
    
    pts = []
    row = 0
    y = -0.5 + dy / 2.0
    while y <= 0.5 - dy / 2.0 + 1e-6:
        offset = 0.0 if row % 2 == 0 else s / 2.0
        x = -0.5 + s / 2.0 + offset
        while x <= 0.5 - s / 2.0 + 1e-6:
            pts.append([x, y])
            x += s
        row += 1
        y += dy
    
    pts = np.array(pts)
    
    if len(pts) > N:
        # Remove farthest from center
        center = np.mean(pts, axis=0)
        dists_to_center = np.linalg.norm(pts - center, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        # Add random points
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, size=2)])
    
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


def _optimize_for_radius(centres, r_target, deadline, max_iter=300):
    """
    Optimize centres to achieve radius >= r_target using L-BFGS-B with explicit gradients.
    """
    bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (2 * N)
    
    def objective(x):
        if time.time() > deadline:
            return np.inf, np.zeros_like(x)
        c = x.reshape(-1, 2)
        val, g = _smooth_energy_and_grad(c, r_target)
        return val, g.flatten()
    
    try:
        result = minimize(
            objective,
            centres.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 5, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.fun < np.inf:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centres


def _polish_centres(centres, deadline, max_iter=200):
    """
    High-precision polish: iteratively adjust the bottleneck constraint.
    """
    current = centres.copy()
    current_radius = _compute_radius(current)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 0.5:
            break
        
        # Compute all pairwise distances and wall distances
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_mat, np.inf)
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - np.abs(current[:, 0]), 0.5 - np.abs(current[:, 1])),
            0.5
        )
        
        # Find the bottleneck
        min_pair_dist = np.min(dist_mat)
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
            # Try multiple step sizes
            improved = False
            for step in [0.001, 0.0005, 0.0002, 0.0001]:
                orig = current[idx].copy()
                if min_d == d_left:
                    current[idx, 0] += step
                elif min_d == d_right:
                    current[idx, 0] -= step
                elif min_d == d_bottom:
                    current[idx, 1] += step
                else:
                    current[idx, 1] -= step
                
                current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
                current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
                
                new_radius = _compute_radius(current)
                if new_radius > current_radius:
                    current_radius = new_radius
                    improved = True
                    break
                else:
                    current[idx] = orig
                    current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
                    current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
            
            if not improved:
                break
        else:
            # Pair is bottleneck
            i, j = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            d = current[i] - current[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-12
                dist = np.linalg.norm(d)
            d = d / dist
            
            improved = False
            for step in [0.001, 0.0005, 0.0002, 0.0001]:
                orig_i = current[i].copy()
                orig_j = current[j].copy()
                current[i] += step * d
                current[j] -= step * d
                
                current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
                current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
                
                new_radius = _compute_radius(current)
                if new_radius > current_radius:
                    current_radius = new_radius
                    improved = True
                    break
                else:
                    current[i] = orig_i
                    current[j] = orig_j
                    current[:, 0] = np.clip(current[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
                    current[:, 1] = np.clip(current[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
            
            if not improved:
                break
    
    return current


def _iterative_radius_increase(centres, deadline):
    """
    Binary search style iterative radius increase.
    Start from current radius, try to increase it in small steps.
    """
    current = centres.copy()
    current_radius = _compute_radius(current)
    
    # Try to increase radius by small increments
    r_step = 0.00005
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 2:
            break
        
        r_target = current_radius + r_step
        if r_target > 0.036612798904:
            r_target = 0.036612798904
        
        # Try to optimize for this target
        optimized = _optimize_for_radius(current, r_target, deadline - 2, max_iter=150)
        new_radius = _compute_radius(optimized)
        
        if new_radius > current_radius + 1e-10:
            current = optimized
            current_radius = new_radius
            # Keep the step size
        else:
            # Reduce step size
            r_step *= 0.5
            if r_step < 1e-7:
                break
    
    return current, current_radius


def construct():
    """Construct 200 circle centres in the unit square to maximize common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    best_centres = _hex_grid_init()
    best_radius = _compute_radius(best_centres)
    
    # Phase 1: Multiple restarts with different initializations
    initializations = [
        _hex_grid_init(),
        _initial_grid(),
        _random_initial(),
        _hex_grid_init() + np.random.randn(N, 2) * 0.005,
        _hex_grid_init() + np.random.randn(N, 2) * 0.01,
        best_centres + np.random.randn(N, 2) * 0.003,
    ]
    
    for init_idx, centres in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
        centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        
        # Adaptive target
        r_target = min(0.036612798904, max(0.033, best_radius * 1.01))
        
        if time.time() < deadline - 20:
            centres = _optimize_for_radius(centres, r_target, deadline - 20, max_iter=250)
        
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
    
    # Phase 2: Iterative radius increase
    if time.time() < deadline - 10:
        polished, polished_radius = _iterative_radius_increase(best_centres, deadline - 5)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centres = polished
    
    # Phase 3: Final polish
    if time.time() < deadline - 3:
        final_polished = _polish_centres(best_centres, deadline - 1)
        final_radius = _compute_radius(final_polished)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centres = final_polished
    
    # Final verification: ensure all centres are strictly inside
    best_centres[:, 0] = np.clip(best_centres[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
    best_centres[:, 1] = np.clip(best_centres[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centres
# EVOLVE-BLOCK-END
