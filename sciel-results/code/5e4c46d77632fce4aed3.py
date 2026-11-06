"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START


def _radius(centers):
    """Compute the radius: min of half pairwise distances and wall distances."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall = 0.5 - np.abs(centers)
    min_wall = float(np.min(wall))
    if min_wall < 0:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = float(np.min(dist)) / 2.0
    return min(min_wall, min_pair)


def _soft_min_dist(centers, alpha=80.0):
    """Compute a differentiable soft-min of all constraint distances.
    
    Returns the soft-min value (higher is better, target is the radius)
    and the gradient with respect to centers.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    
    # Wall distances: for each center and each axis, distance to nearest wall
    # Wall distance for center i, axis d: 0.5 - |centers[i,d]|
    # Gradient of wall distance w.r.t. centers[i,d]: -sign(centers[i,d])
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Combine all distances into a single array
    # Pairwise: (N*(N-1)/2) values
    # Wall: (N*3) values
    n_pairs = n * (n - 1) // 2
    n_walls = n * 3
    total = n_pairs + n_walls
    
    dists = np.zeros(total)
    # Fill pairwise distances
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dists[idx] = dist[i, j]
            idx += 1
    # Fill wall distances
    for i in range(n):
        for d in range(3):
            dists[idx] = wall[i, d]
            idx += 1
    
    # Soft-min using log-sum-exp
    # soft_min = -(1/alpha) * log(sum(exp(-alpha * d_i)))
    # To avoid overflow, subtract max
    neg_alpha_dists = -alpha * dists
    max_val = np.max(neg_alpha_dists)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_alpha_dists - max_val)))
    soft_min_val = -log_sum_exp / alpha
    
    # Gradient: d(soft_min)/d(centers)
    # soft_min = -(1/alpha) * log(sum(exp(-alpha * d_k)))
    # d(soft_min)/d(x) = -(1/alpha) * (1/S) * sum(-alpha * exp(-alpha*d_k) * d(d_k)/dx)
    # where S = sum(exp(-alpha*d_k))
    # = sum(exp(-alpha*d_k) * d(d_k)/dx) / S
    
    weights = np.exp(neg_alpha_dists - max_val)
    S = np.sum(weights)
    # Normalize weights (these are proportional to exp(-alpha*d_k))
    weights_norm = weights / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise distance gradients
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights_norm[idx]
            d = dist[i, j]
            if d > 1e-15:
                dir_vec = (centers[i] - centers[j]) / d
                grad[i] += w * dir_vec
                grad[j] -= w * dir_vec
            idx += 1
    
    # Wall distance gradients
    for i in range(n):
        for d in range(3):
            w = weights_norm[idx]
            # wall[i,d] = 0.5 - |centers[i,d]|
            # d(wall)/d(centers[i,d]) = -sign(centers[i,d])
            if centers[i, d] >= 0:
                grad[i, d] += -w
            else:
                grad[i, d] += w
            idx += 1
    
    return soft_min_val, grad


def _objective_and_grad(x, alpha=80.0):
    """Objective: negative soft-min radius (minimize this)."""
    centers = x.reshape(N, 3)
    val, grad = _soft_min_dist(centers, alpha)
    return -val, -grad


def _make_bcc_init(seed=42):
    """Body-centered cubic lattice initialization."""
    rng = np.random.RandomState(seed)
    points = []
    # BCC: corners and body centers of a grid
    # Use a 4x4x4 grid of cells, each with corner and body center
    # But we only need 50 points
    # Let's use a simpler approach: take a subset of BCC lattice points
    
    # Generate BCC lattice in [0,1]^3
    n = 4
    step = 1.0 / n
    pts = []
    for ix in range(n + 1):
        for iy in range(n + 1):
            for iz in range(n + 1):
                pts.append([ix * step, iy * step, iz * step])
                # Body center
                if ix < n and iy < n and iz < n:
                    pts.append([(ix + 0.5) * step, (iy + 0.5) * step, (iz + 0.5) * step])
    
    pts = np.array(pts)
    # Translate to [-0.5, 0.5]
    pts = pts - 0.5
    
    # Shuffle and take N
    idx = rng.permutation(len(pts))[:N]
    centers = pts[idx].copy()
    # Small jitter
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_grid_init(seed=42):
    """4x4x4 grid initialization, take first 50."""
    rng = np.random.RandomState(seed)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += rng.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_fcc_init(seed=42):
    """FCC-inspired initialization."""
    rng = np.random.RandomState(seed)
    # FCC: face centers and corners
    n = 4
    step = 1.0 / n
    pts = []
    for ix in range(n + 1):
        for iy in range(n + 1):
            for iz in range(n + 1):
                pts.append([ix * step, iy * step, iz * step])
                # Face centers
                if ix < n and iy < n:
                    pts.append([(ix + 0.5) * step, (iy + 0.5) * step, iz * step])
                if ix < n and iz < n:
                    pts.append([(ix + 0.5) * step, iy * step, (iz + 0.5) * step])
                if iy < n and iz < n:
                    pts.append([ix * step, (iy + 0.5) * step, (iz + 0.5) * step])
    
    pts = np.array(pts)
    pts = pts - 0.5
    idx = rng.permutation(len(pts))[:N]
    centers = pts[idx].copy()
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_random_init(seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _optimize_from(centers, time_budget=15.0, alpha_start=50.0, alpha_end=200.0):
    """Optimize from given centers using L-BFGS-B with soft-min objective."""
    start_time = time.time()
    current = centers.copy()
    x0 = current.flatten()
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    # Multi-scale alpha: start with lower alpha (smoother), increase for precision
    alphas = [alpha_start, alpha_start * 2, alpha_start * 4, alpha_end]
    
    for alpha in alphas:
        if time.time() - start_time > time_budget - 2.0:
            break
        try:
            def obj(x):
                return _objective_and_grad(x, alpha)[0]
            
            def obj_grad(x):
                return _objective_and_grad(x, alpha)
            
            result = minimize(
                obj,
                x0,
                jac=obj_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500}
            )
            new_centers = result.x.reshape(N, 3)
            new_radius = _radius(new_centers)
            old_radius = _radius(current)
            if new_radius > old_radius - 1e-12:
                current = new_centers
                x0 = result.x
        except Exception:
            break
    
    return current, _radius(current)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple initializations
    inits = [
        ("grid42", _make_grid_init(42)),
        ("grid123", _make_grid_init(123)),
        ("bcc42", _make_bcc_init(42)),
        ("bcc777", _make_bcc_init(777)),
        ("fcc42", _make_fcc_init(42)),
        ("rand1", _make_random_init(1)),
        ("rand2", _make_random_init(42)),
    ]
    
    for name, init_centers in inits:
        if time.time() > deadline - 20.0:
            break
        
        current = init_centers.copy()
        current_radius = _radius(current)
        
        # Optimize
        current, current_radius = _optimize_from(current, time_budget=15.0)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current.copy()
    
    if best_centers is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        best_centers, best_radius = _optimize_from(best_centers, time_budget=10.0,
                                                     alpha_start=100.0, alpha_end=400.0)
    
    # Clip to ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers


# EVOLVE-BLOCK-END
