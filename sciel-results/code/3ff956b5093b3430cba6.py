"""
Pack 200 equal circles in a unit square using gradient-based optimization
with soft-min (log-sum-exp) surrogate, force-directed repulsion, and
high-precision local polish.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 145  # Leave margin for 150s kill

def _compute_radius_fast(pts):
    """
    Fast radius computation using cdist.
    """
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    D = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=-1))
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)


def _soft_min(values, beta=20.0):
    """
    Smooth approximation of min using log-sum-exp.
    As beta -> inf, this approaches the true min.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    m = np.min(values)
    if m < -1e10:
        return m
    # Use scaled log-sum-exp for numerical stability
    v = values - m
    # For large beta, exp(-beta * v) underflows for v > ~50/beta
    # Use a threshold to avoid overflow
    safe_v = np.clip(v, 0, 50.0 / beta)
    s = np.sum(np.exp(-beta * safe_v))
    if s <= 0:
        return m
    return m - np.log(s) / beta


def _soft_min_objective_and_grad(x, beta=20.0):
    """
    Compute the soft-min objective (negative radius) and its gradient.
    x is a flat array of shape (2*N,).
    Returns (f, grad) where f = -r_soft and grad = df/dx.
    """
    pts = x.reshape(N, 2)
    
    # Compute all pairwise distances and wall distances
    # Pairwise: for i < j, d_ij = |p_i - p_j|
    # Wall: for each point k, w_k = min(p_k+0.5, 0.5-p_k, p_ky+0.5, 0.5-p_ky)
    
    # We want r = min(min_{i<j} d_ij/2, min_k w_k)
    # Soft-min: r_soft = soft_min(all values / 2 for pairs, all wall values)
    
    # Collect all "radius candidates":
    # For each pair (i,j) with i<j: d_ij / 2
    # For each point k and each wall: wall distance
    
    # Build the gradient using chain rule on soft-min
    # soft_min(v) = min(v) - (1/beta) * log(sum(exp(-beta*(v_k - min(v)))))
    # d(soft_min)/dv_k = -exp(-beta*(v_k - min(v))) / sum(exp(-beta*(v_k' - min(v))))
    
    # We'll compute this efficiently.
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(D, np.inf)
    
    # Upper triangle indices
    iu, ju = np.triu_indices(N, k=1)
    pair_dists = D[iu, ju]  # shape (M,)
    
    # Wall distances for each point
    wall_x_min = pts[:, 0] + 0.5
    wall_x_max = 0.5 - pts[:, 0]
    wall_y_min = pts[:, 1] + 0.5
    wall_y_max = 0.5 - pts[:, 1]
    
    # All radius candidates
    all_vals = np.concatenate([pair_dists / 2.0, wall_x_min, wall_x_max, wall_y_min, wall_y_max])
    
    m = np.min(all_vals)
    if m < -1e10:
        return 0.0, np.zeros_like(x)
    
    v = all_vals - m
    safe_v = np.clip(v, 0, 50.0 / beta)
    exp_terms = np.exp(-beta * safe_v)
    s = np.sum(exp_terms)
    if s <= 0:
        return -m, np.zeros_like(x)
    
    weights = exp_terms / s  # sum to 1
    
    r_soft = m - np.log(s) / beta
    
    # Gradient of r_soft w.r.t. x
    grad = np.zeros_like(x)
    
    # For pair (i,j): d_ij/2 contributes with weight w_p
    # d(d_ij/2)/dx = 0.5 * (p_i - p_j) / d_ij
    for idx, (i, j) in enumerate(zip(iu, ju)):
        w = weights[idx]
        if w < 1e-15:
            continue
        di = pair_dists[idx]
        if di < 1e-12:
            continue
        # d(d_ij)/dx_i = (p_i - p_j) / d_ij
        # d(d_ij/2)/dx_i = 0.5 * (p_i - p_j) / d_ij
        diff_ij = pts[i] - pts[j]
        grad[2*i] += w * 0.5 * diff_ij[0] / di
        grad[2*i+1] += w * 0.5 * diff_ij[1] / di
        grad[2*j] -= w * 0.5 * diff_ij[0] / di
        grad[2*j+1] -= w * 0.5 * diff_ij[1] / di
    
    # Wall contributions
    base = len(pair_dists)
    # wall_x_min[k] = pts[k,0] + 0.5, grad = 1 for x
    for k in range(N):
        w = weights[base + k]
        if w > 1e-15:
            grad[2*k] += w
    
    base += N
    # wall_x_max[k] = 0.5 - pts[k,0], grad = -1 for x
    for k in range(N):
        w = weights[base + k]
        if w > 1e-15:
            grad[2*k] -= w
    
    base += N
    # wall_y_min[k] = pts[k,1] + 0.5, grad = 1 for y
    for k in range(N):
        w = weights[base + k]
        if w > 1e-15:
            grad[2*k+1] += w
    
    base += N
    # wall_y_max[k] = 0.5 - pts[k,1], grad = -1 for y
    for k in range(N):
        w = weights[base + k]
        if w > 1e-15:
            grad[2*k+1] -= w
    
    f = -r_soft
    return f, grad

def _force_directed_step(pts, n_steps=50, deadline=None):
    """
    Force-directed repulsion: push overlapping circles apart.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    pts = pts.copy()
    r = _compute_radius_fast(pts)
    if r <= 0:
        # Estimate a reasonable radius
        r = 0.03
    
    for step in range(n_steps):
        if time.time() > deadline - 2:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion (only for pairs closer than 2*r * 1.5)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(D, np.inf)
        
        # Only consider pairs that are too close
        threshold = 2 * r * 1.5
        close_mask = (D < threshold) & (D > 1e-12)
        np.fill_diagonal(close_mask, False)
        
        for i in range(N):
            for j in range(i+1, N):
                if close_mask[i, j]:
                    di = D[i, j]
                    # Repulsive force proportional to overlap
                    overlap = 2 * r - di
                    if overlap > 0:
                        f = 0.1 * overlap / di
                        forces[i] += f * (pts[i] - pts[j])
                        forces[j] -= f * (pts[i] - pts[j])
        
        # Wall repulsion
        for k in range(N):
            # x walls
            if pts[k, 0] + 0.5 < r:
                forces[k, 0] += 0.1 * (r - (pts[k, 0] + 0.5))
            if 0.5 - pts[k, 0] < r:
                forces[k, 0] -= 0.1 * (r - (0.5 - pts[k, 0]))
            # y walls
            if pts[k, 1] + 0.5 < r:
                forces[k, 1] += 0.1 * (r - (pts[k, 1] + 0.5))
            if 0.5 - pts[k, 1] < r:
                forces[k, 1] -= 0.1 * (r - (0.5 - pts[k, 1]))
        
        # Apply forces with damping
        pts += 0.5 * forces
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts

def _bottleneck_sa(pts, start_time, deadline, max_iter=5000, T_init=0.01, T_min=1e-6, alpha=0.95):
    """
    Simulated Annealing on the exact radius objective, focusing on bottlenecks.
    """
    pts = pts.copy()
    r = _compute_radius_fast(pts)
    best_r = r
    best_pts = pts.copy()
    
    T = T_init
    rng = np.random.default_rng()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Find current bottlenecks
        pair_indices, wall_indices = _find_bottlenecks(pts, r)
        
        if not pair_indices and not wall_indices:
            # No bottlenecks? Should not happen if r is computed correctly, 
            # unless r is limited by something else or numerical issues.
            # If no bottlenecks, we can't improve? 
            # Actually, if no constraints are active, we can move freely.
            # But r is min(d/2, w). If no d/2 <= r and no w <= r, then r < min(d/2, w).
            # This implies we can increase r.
            # In that case, we should just do a random move or a gradient step.
            # For simplicity, if no bottlenecks, pick a random pair and push apart.
            i, j = rng.integers(0, N, 2)
            if i == j: continue
            pair_indices = [(i, j)]
            
        # Choose a move
        # Bias towards pairs if they are the limiting factor, or walls.
        # For simplicity, pick randomly from all available bottlenecks.
        # If there are many, pick a random one.
        
        move_type = None
        if pair_indices and (not wall_indices or rng.random() < 0.7):
            move_type = 'pair'
            i, j = pair_indices[rng.integers(0, len(pair_indices))]
        else:
            move_type = 'wall'
            k, axis, side = wall_indices[rng.integers(0, len(wall_indices))]
            
        new_pts = pts.copy()
        
        # Generate move
        # Step size proportional to current radius
        step = r * 0.1
        
        if move_type == 'pair':
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1e-10:
                # Random direction
                angle = rng.uniform(0, 2*np.pi)
                dx, dy = np.cos(angle), np.sin(angle)
                dist = 1.0
            else:
                dx, dy = dx/dist, dy/dist
                
            # Move i away from j, j away from i
            # Add some jitter to escape local minima
            jitter_mag = step * 0.5 * rng.random()
            jitter_angle = rng.uniform(0, 2*np.pi)
            jx, jy = jitter_mag * np.cos(jitter_angle), jitter_mag * np.sin(jitter_angle)
            
            new_pts[i, 0] += dx * step + jx
            new_pts[i, 1] += dy * step + jy
            new_pts[j, 0] -= dx * step - jx
            new_pts[j, 1] -= dy * step - jy
            
            # Clip to square
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
        else: # wall
            k, axis, side = move_type_data # Wait, I need to unpack this properly
            # Re-unpack from the chosen index
            k, axis, side = wall_indices[rng.integers(0, len(wall_indices))]
            
            # Move away from wall
            # axis 0: x, axis 1: y
            # side 0: min (-0.5), side 1: max (0.5)
            
            if axis == 0:
                if side == 0: # Left wall, move right (+x)
                    new_pts[k, 0] += step
                else: # Right wall, move left (-x)
                    new_pts[k, 0] -= step
            else:
                if side == 0: # Bottom wall, move up (+y)
                    new_pts[k, 1] += step
                else: # Top wall, move down (-y)
                    new_pts[k, 1] -= step
            
            # Add jitter perpendicular to wall
            jitter_mag = step * 0.5 * rng.random()
            if axis == 0:
                new_pts[k, 1] += jitter_mag * (1 if rng.random() < 0.5 else -1)
            else:
                new_pts[k, 0] += jitter_mag * (1 if rng.random() < 0.5 else -1)
            
            new_pts = np.clip(new_pts, -0.5, 0.5)
            
        # Evaluate
        new_r = _compute_radius_fast(new_pts)
        
        # Acceptance criterion
        if new_r > r:
            pts = new_pts
            r = new_r
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        else:
            # Simulated annealing acceptance
            delta = r - new_r # Positive if worse
            if delta < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
            if accept:
                pts = new_pts
                r = new_r
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_pts, best_r

def _hexagonal_initialization(n, jitter=0.0):
    """
    Generate a hexagonal layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < n:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:n]
    else:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]
    
    if jitter > 0:
        points = points + jitter * np.random.randn(n, 2)
        points = np.clip(points, -0.5, 0.5)
    
    return points


def _generate_hex_initial_with_spacing(n, r_est=None):
    """
    Generate hex layout with a target radius estimate for better spacing.
    """
    if r_est is None:
        # Estimate from area: 200 circles of radius r in unit square
        # Area per circle ~ pi*r^2, but packing efficiency ~0.9
        # 200 * pi * r^2 * 0.9 <= 1 => r ~ 1/sqrt(200*pi*0.9)
        r_est = 1.0 / np.sqrt(n * np.pi * 0.9)
    
    spacing_x = 2 * r_est
    spacing_y = r_est * np.sqrt(3)
    
    points = []
    row = 0
    while len(points) < n:
        y = -0.5 + r_est + row * spacing_y
        if y > 0.5 - r_est:
            break
        offset = 0.0 if row % 2 == 0 else spacing_x / 2
        col = 0
        while True:
            x = -0.5 + r_est + offset + col * spacing_x
            if x > 0.5 - r_est:
                break
            points.append([x, y])
            col += 1
        row += 1
    
    points = np.array(points)
    
    if len(points) < n:
        # Fill remaining with grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = np.vstack([points, grid[len(points):n]])
    else:
        # Select closest to center
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]
    
    return points

def _grid_initialization(n, jitter=0.0):
    """
    Generate a grid layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) < n:
        extra = n - len(grid)
        extra_pts = np.random.uniform(-0.45, 0.45, (extra, 2))
        grid = np.vstack([grid, extra_pts])
    
    grid = grid[:n]
    
    if jitter > 0:
        grid = grid + jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)
    
    return grid


def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))

def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))

def _run_optimization(pts, deadline):
    """
    Full optimization pipeline: force-directed -> L-BFGS -> local polish.
    """
    # Phase 1: Force-directed to resolve overlaps
    remaining = deadline - time.time()
    if remaining > 5:
        pts_fd = _force_directed_step(pts, n_steps=30, deadline=deadline)
        r_fd = _compute_radius_fast(pts_fd)
        r_init = _compute_radius_fast(pts)
        if r_fd > r_init:
            pts = pts_fd
    
    # Phase 2: L-BFGS with soft-min
    remaining = deadline - time.time()
    if remaining > 10:
        pts_lbfgs, r_lbfgs = _lbfgs_optimize(pts, deadline)
        if r_lbfgs > _compute_radius_fast(pts):
            pts = pts_lbfgs
    
    # Phase 3: Exact local polish
    remaining = deadline - time.time()
    if remaining > 5:
        pts_polished = _exact_local_polish(pts, max_iters=100, deadline=deadline)
        r_polished = _compute_radius_fast(pts_polished)
        if r_polished > _compute_radius_fast(pts):
            pts = pts_polished
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    initializers = [
        ("hex_spaced", lambda: _generate_hex_initial_with_spacing(N)),
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.005)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("grid_jitter", lambda: _grid_initialization(N, jitter=0.005)),
        ("random", lambda: _random_initialization(N)),
    ]
    
    best_coords = None
    best_radius = 0.0
    
    for name, init_func in initializers:
        if time.time() > deadline - 15:
            break
        
        try:
            coords = init_func()
            r = _compute_radius_fast(coords)
            
            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()
            
            # Run full optimization pipeline
            remaining = deadline - time.time() - 5
            if remaining > 15:
                opt_coords = _run_optimization(coords, deadline)
                opt_r = _compute_radius_fast(opt_coords)
                
                if opt_r > best_radius:
                    best_radius = opt_r
                    best_coords = opt_coords.copy()
                    
        except Exception:
            continue
    
    if best_coords is None or not np.all(np.isfinite(best_coords)):
        return _grid_initialization(N)
        
    return best_coords

# EVOLVE-BLOCK-START
def compute_radius(coords):
    return _compute_radius_fast(coords)

def objective(coords):
    r = compute_radius(coords)
    return -r

def initial_hex_layout(n):
    return _hexagonal_initialization(n)

# EVOLVE-BLOCK-END
