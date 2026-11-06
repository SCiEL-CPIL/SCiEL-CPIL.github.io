"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(pts):
    """Compute the maximum radius for a set of points."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distances to walls
    # Square is [-0.5, 0.5]
    dists_wall = np.minimum(np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]), 
                            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1]))
    min_wall = np.min(dists_wall)
    
    return min(min_pair / 2.0, min_wall)

def _energy_and_grad(x, r_target):
    """
    Smooth energy: penalize pairs closer than 2*r_target and walls closer than r_target.
    Uses soft-min via log-sum-exp for smoothness.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid issues on diagonal
    
    # Only consider pairs that are too close
    # Use a smooth penalty: sum of max(0, 2*r - dist)^2
    # For gradient, we need d/dx of max(0, 2*r - dist)^2
    
    # Wall distances
    dists_wall = np.minimum(np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]), 
                            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1]))
    
    # Smooth penalty for walls
    wall_pen = np.sum(np.maximum(0, r_target - dists_wall) ** 2)
    
    # Smooth penalty for pairs
    # For each pair (i,j) with i<j, if dist < 2*r_target, penalize
    # Use vectorized computation
    mask = dists < 2 * r_target
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    mask_pairs = pair_dists < 2 * r_target
    
    if np.any(mask_pairs):
        close_dists = pair_dists[mask_pairs]
        # Penalty: sum (2*r - dist)^2 for close pairs
        pen_val = np.sum((2 * r_target - close_dists) ** 2)
        
        # Gradient: d/dx_i of (2*r - |x_i - x_j|)^2 = -2*(2*r - |x_i - x_j|) * (x_i - x_j)/|x_i - x_j|
        # For each close pair, compute gradient contribution
        close_indices = np.where(mask_pairs)[0]
        i_idx, j_idx = iu[0][close_indices], iu[1][close_indices]
        
        grad = np.zeros_like(x)
        for k in range(len(close_indices)):
            i, j = i_idx[k], j_idx[k]
            d = 2 * r_target - close_dists[k]
            if d > 0:
                dx = pts[i] - pts[j]
                dist_val = close_dists[k]
                if dist_val > 1e-10:
                    grad[2*i] += -2 * d * dx[0] / dist_val
                    grad[2*i+1] += -2 * d * dx[1] / dist_val
                    grad[2*j] += 2 * d * dx[0] / dist_val
                    grad[2*j+1] += 2 * d * dx[1] / dist_val
    else:
        pen_val = 0.0
        grad = np.zeros_like(x)
    
    # Wall gradient
    wall_grad = np.zeros_like(x)
    for i in range(N):
        dx_wall = np.minimum(0.5 - pts[i, 0], 0.5 + pts[i, 0])
        dy_wall = np.minimum(0.5 - pts[i, 1], 0.5 + pts[i, 1])
        dwall = min(dx_wall, dy_wall)
        if dwall < r_target:
            d = r_target - dwall
            if dx_wall <= dy_wall:
                # x-wall is limiting
                if 0.5 - pts[i, 0] <= 0.5 + pts[i, 0]:
                    wall_grad[2*i] += 2 * d  # d/dx (0.5-x) = -1, so d/dx max(0, r-(0.5-x))^2 = 2*(r-(0.5-x))*1
                else:
                    wall_grad[2*i] -= 2 * d  # d/dx (0.5+x) = 1, so d/dx max(0, r-(0.5+x))^2 = -2*(r-(0.5+x))
            else:
                if 0.5 - pts[i, 1] <= 0.5 + pts[i, 1]:
                    wall_grad[2*i+1] += 2 * d
                else:
                    wall_grad[2*i+1] -= 2 * d
    
    total_energy = pen_val + wall_pen
    total_grad = grad + wall_grad
    
    return total_energy, total_grad

def _hex_grid_init():
    """Create a hexagonal lattice initialization."""
    pts = []
    spacing = 0.07  # Approximate spacing for 200 points in unit square
    row = 0
    while len(pts) < N:
        x_offset = (row % 2) * spacing / 2
        for col in range(15):
            x = -0.5 + x_offset + col * spacing
            y = -0.5 + row * spacing * 0.866
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
        row += 1
    
    pts = np.array(pts[:N])
    # Add small random jitter
    np.random.seed(42)
    pts += np.random.uniform(-0.01, 0.01, pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _grid_init():
    """Create a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init():
    """Create a random initialization."""
    np.random.seed(123)
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _push_apart(pts, n_iter=100, learning_rate=0.01):
    """
    Simple push-apart algorithm to separate overlapping circles.
    """
    pts = pts.copy()
    for it in range(n_iter):
        # Compute pairwise forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        # Force: repulsion for close pairs
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 0.08:  # Only push apart if too close
                    force = (0.08 - d) / d if d > 1e-10 else 0.01
                    forces[i] += force * (pts[i] - pts[j])
        
        # Wall forces
        for i in range(N):
            for coord in range(2):
                if pts[i, coord] < -0.45:
                    forces[i, coord] += 0.01
                elif pts[i, coord] > 0.45:
                    forces[i, coord] -= 0.01
        
        pts += learning_rate * forces
        pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def _optimize_for_radius(pts, r_target, max_iter=500, tol=1e-10):
    """
    Optimize positions to achieve minimum radius r_target.
    Uses gradient descent with line search.
    """
    x = pts.flatten().copy()
    lr = 0.01
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(x, r_target)
        
        if energy < tol:
            break
        
        # Gradient step
        x -= lr * grad
        
        # Clip to bounds
        x = np.clip(x, -0.49, 0.49)
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr *= 0.9
    
    return x.reshape(N, 2)

def _polish_centres(pts, r_target, max_iter=1000, tol=1e-12):
    """
    High-precision polish pass using Newton-like method.
    """
    x = pts.flatten().copy()
    lr = 0.001
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(x, r_target)
        
        if energy < tol:
            break
        
        # Gradient step with smaller learning rate
        x -= lr * grad
        x = np.clip(x, -0.49, 0.49)
        
        if it % 100 == 0:
            lr *= 0.95
    
    return x.reshape(N, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_x = None
    best_r = 0.0
    
    # Try different initializations
    initializations = [
        ('hex', _hex_grid_init()),
        ('grid', _grid_init()),
        ('random', _random_init()),
    ]
    
    # Start with a reasonable radius estimate
    # For 200 circles in unit square, approximate radius ~ 1/sqrt(200) / 2
    r_start = 0.035
    
    for init_name, init_pts in initializations:
        if time.time() > deadline - 20.0:
            break
        
        pts = init_pts.copy()
        
        # Push apart to get a better starting point
        pts = _push_apart(pts, n_iter=50, learning_rate=0.005)
        
        # Iteratively increase radius
        r = r_start
        for r_iter in range(30):
            if time.time() > deadline - 10.0:
                break
            
            # Try to achieve this radius
            optimized = _optimize_for_radius(pts, r, max_iter=200, tol=1e-10)
            actual_r = _compute_radius(optimized)
            
            if actual_r >= r - 1e-6:
                # Success, try larger radius
                pts = optimized
                r *= 1.02
            else:
                # Failed, try smaller increment
                r *= 1.005
                if r < r_start * 0.5:
                    break
        
        # Polish
        if time.time() < deadline - 5.0:
            final_r = _compute_radius(pts)
            pts = _polish_centres(pts, final_r * 0.999, max_iter=500, tol=1e-12)
            final_r = _compute_radius(pts)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_x = pts.copy()
    
    if best_x is None:
        best_x = _grid_init()
    
    return best_x
# EVOLVE-BLOCK-END
