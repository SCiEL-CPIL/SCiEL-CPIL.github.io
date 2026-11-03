"""Pack 200 equal circles in a unit square using vectorized force-directed + L-BFGS-B optimization."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance using KDTree
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        pair_dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        min_pair_dist = np.min(pair_dists)

    return min(wall_dist, min_pair_dist / 2.0)


def _generate_hex_lattice(spacing, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal lattice of points inside the square [-0.5, 0.5]^2."""
    pts = []
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing

    y = -0.5 + offset_y
    row_idx = 0
    while y <= 0.5 + 1e-9:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + offset_x + shift
        while x <= 0.5 + 1e-9:
            if -0.5 - 1e-9 <= x <= 0.5 + 1e-9 and -0.5 - 1e-9 <= y <= 0.5 + 1e-9:
                pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1

    if not pts:
        return np.array([])
    return np.array(pts)


def _generate_hex_initial(spacing=0.07, n_rows=None, n_cols=None, jitter=0.0):
    """Generate a hexagonal initial configuration with exactly N points."""
    if n_rows is None or n_cols is None:
        # Estimate rows/cols for N points
        n_cols = int(np.sqrt(N * 2 / np.sqrt(3)))
        n_rows = int(np.ceil(N / n_cols))
    
    pts = []
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing
    
    y_start = -0.5 + spacing / 2
    x_start = -0.5 + spacing / 2
    
    for i in range(n_rows):
        shift = (col_w / 2.0) if i % 2 == 1 else 0.0
        for j in range(n_cols):
            x = x_start + j * col_w + shift
            y = y_start + i * row_h
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    # If we have too many, randomly select N
    if len(pts) > N:
        indices = np.random.choice(len(pts), N, replace=False)
        pts = pts[indices]
    elif len(pts) < N:
        # Add random points to fill
        while len(pts) < N:
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-0.5, 0.5)
            pts = np.vstack([pts, [[x, y]]])
    
    if jitter > 0:
        pts += np.random.randn(len(pts), 2) * jitter
    
    # Clip to bounds
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    return pts[:N]


def _soft_min_objective(pts, beta=50.0):
    """
    Compute soft-min objective (negative of soft-min of all constraints) and gradient.
    Returns (value, gradient) where gradient is w.r.t. the 2N-dimensional parameter vector.
    """
    n = len(pts)
    if n == 0:
        return 0.0, np.zeros(2 * N)
    
    # Wall distances: 4n values
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,      # distance to left wall
        0.5 - pts[:, 0],      # distance to right wall
        pts[:, 1] + 0.5,      # distance to bottom wall
        0.5 - pts[:, 1]       # distance to top wall
    ])
    
    # Pairwise distances / 2
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    
    if len(pairs) == 0:
        pair_vals = np.array([10.0])
        pair_idx = []
    else:
        pair_dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        pair_vals = pair_dists / 2.0
        pair_idx = pairs
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    soft_min_value = -(max_neg + np.log(sum_exp)) / beta
    
    # Gradient w.r.t. pts
    grad = np.zeros_like(pts)
    
    # Wall gradients
    weights_wall = exp_shifted[:4 * n] / sum_exp
    for i in range(n):
        grad[i, 0] += weights_wall[4 * i] * 1.0
        grad[i, 0] += weights_wall[4 * i + 1] * (-1.0)
        grad[i, 1] += weights_wall[4 * i + 2] * 1.0
        grad[i, 1] += weights_wall[4 * i + 3] * (-1.0)
    
    # Pair gradients
    if len(pair_idx) > 0:
        weights_pair = exp_shifted[4 * n:] / sum_exp
        for k, (i, j) in enumerate(pair_idx):
            if k >= len(pair_idx):
                break
            w = weights_pair[k]
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            grad[i] += w * diff / (2.0 * dist)
            grad[j] -= w * diff / (2.0 * dist)
    
    return soft_min_value, grad


def _objective_and_grad_flat(x, beta=50.0):
    """
    Flat parameter version for scipy.optimize.minimize.
    x is a 2N vector of [x0, y0, x1, y1, ...].
    Returns (objective_value, gradient).
    """
    n = N
    pts = x.reshape(n, 2)
    
    # Wall distances
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pairwise distances / 2
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    
    if len(pairs) == 0:
        pair_vals = np.array([10.0])
        pair_idx = []
    else:
        pair_dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        pair_vals = pair_dists / 2.0
        pair_idx = pairs
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft min
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    obj_val = -(max_neg + np.log(sum_exp)) / beta
    
    # Gradient
    grad = np.zeros(2 * n)
    
    # Wall gradients
    weights_wall = exp_shifted[:4 * n] / sum_exp
    for i in range(n):
        grad[2 * i] += weights_wall[4 * i] * 1.0
        grad[2 * i] += weights_wall[4 * i + 1] * (-1.0)
        grad[2 * i + 1] += weights_wall[4 * i + 2] * 1.0
        grad[2 * i + 1] += weights_wall[4 * i + 3] * (-1.0)
    
    # Pair gradients
    if len(pair_idx) > 0:
        weights_pair = exp_shifted[4 * n:] / sum_exp
        for k, (i, j) in enumerate(pair_idx):
            if k >= len(pair_idx):
                break
            w = weights_pair[k]
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            grad[2 * i] += w * diff[0] / (2.0 * dist)
            grad[2 * i + 1] += w * diff[1] / (2.0 * dist)
            grad[2 * j] -= w * diff[0] / (2.0 * dist)
            grad[2 * j + 1] -= w * diff[1] / (2.0 * dist)
    
    return obj_val, grad


def _lbfgs_polish(pts, beta_schedule=None, max_iters=500, time_limit=None):
    """
    Polish using L-BFGS-B on the soft-min objective with increasing beta.
    """
    pts = pts.copy()
    n = len(pts)
    
    if beta_schedule is None:
        beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in beta_schedule:
        if time_limit is not None and time.time() > time_limit:
            break
        
        x0 = pts.reshape(-1)
        
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * n)
        
        # Use L-BFGS-B
        result = minimize(
            _objective_and_grad_flat,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        pts = result.x.reshape(n, 2)
    
    return pts


def _vectorized_force_step(pts, r_target, dt=0.001, repulsion_strength=5.0, wall_strength=10.0):
    """
    Vectorized force-directed step using KDTree for neighbor finding.
    Returns updated positions.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Wall forces (vectorized)
    forces[:, 0] += wall_strength * np.maximum(0, (-0.5 + r_target) - pts[:, 0])
    forces[:, 0] -= wall_strength * np.maximum(0, pts[:, 0] - (0.5 - r_target))
    forces[:, 1] += wall_strength * np.maximum(0, (-0.5 + r_target) - pts[:, 1])
    forces[:, 1] -= wall_strength * np.maximum(0, pts[:, 1] - (0.5 - r_target))
    
    # Repulsive forces using KDTree
    tree = KDTree(pts)
    cutoff = max(2 * r_target * 1.5, 0.15)
    
    for i in range(n):
        neighbors = tree.query_ball_point(pts[i], cutoff)
        for j in neighbors:
            if i == j:
                continue
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                forces[i] += np.random.randn(2) * 0.005
                continue
            if dist < 2 * r_target:
                overlap = 2 * r_target - dist
                force_mag = repulsion_strength * overlap / (dist * dist + 1e-10)
                forces[i] += force_mag * diff
    
    pts = pts + dt * forces
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    
    return pts


def _force_directed_polish(pts, r_target, max_iters=200, dt_start=0.002, dt_end=0.0001, time_limit=None):
    """
    Force-directed polishing with cooling.
    """
    pts = pts.copy()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Anneal dt
        progress = it / max_iters
        dt = dt_start * (dt_end / dt_start) ** progress
        
        pts = _vectorized_force_step(pts, r_target, dt=dt)
    
    return pts


def construct():
    t0 = time.time()
    
    best_r = 0.0
    best_pts = None
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Generate multiple hexagonal initial configurations
    # 2. Polish each with force-directed + L-BFGS-B
    # 3. Keep the best result
    
    # Generate several initial configurations
    spacings = [0.065, 0.07, 0.075, 0.08, 0.085]
    initial_configs = []
    
    for spacing in spacings:
        if time.time() - t0 > _TIME_LIMIT - 20:
            break
        
        # Generate hex initial with slight jitter
        for jitter in [0.0, 0.005, 0.01]:
            if time.time() - t0 > _TIME_LIMIT - 20:
                break
            
            pts = _generate_hex_initial(spacing=spacing, jitter=jitter)
            if len(pts) == N:
                initial_configs.append(pts.copy())
    
    # Also add a random initialization
    random_pts = np.random.uniform(-0.5, 0.5, (N, 2))
    initial_configs.append(random_pts)
    
    # Process each initial configuration
    for init_pts in initial_configs:
        if time.time() - t0 > _TIME_LIMIT - 15:
            break
        
        r_init = _get_radius(init_pts)
        
        # Force-directed polish
        fd_pts = _force_directed_polish(
            init_pts, 
            r_target=r_init * 1.01,
            max_iters=100,
            dt_start=0.002,
            dt_end=0.0001,
            time_limit=_TIME_LIMIT - 10
        )
        r_fd = _get_radius(fd_pts)
        
        # L-BFGS-B polish with increasing beta
        lbfgs_pts = _lbfgs_polish(
            fd_pts,
            beta_schedule=[10.0, 25.0, 50.0, 100.0, 200.0],
            max_iters=200,
            time_limit=_TIME_LIMIT - 5
        )
        r_lbfgs = _get_radius(lbfgs_pts)
        
        # Track best
        candidates = [
            (init_pts, r_init),
            (fd_pts, r_fd),
            (lbfgs_pts, r_lbfgs)
        ]
        
        for pts_candidate, r_candidate in candidates:
            if r_candidate > best_r:
                best_r = r_candidate
                best_pts = pts_candidate.copy()
    
    # Final intensive L-BFGS-B polish on the best result
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 10:
        final_pts = _lbfgs_polish(
            best_pts,
            beta_schedule=[50.0, 100.0, 200.0, 400.0],
            max_iters=300,
            time_limit=_TIME_LIMIT - 3
        )
        r_final = _get_radius(final_pts)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_pts.copy()
    
    # Fallback if something went wrong
    if best_pts is None:
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
