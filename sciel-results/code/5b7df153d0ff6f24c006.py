"""Packing 50 equal spheres in a unit cube - optimized via gradient-based search."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
SEED = 42

def _pairwise_distances(centers):
    """Compute all pairwise distances and wall distances efficiently."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances (distance to each face)
    wall_dist = np.min(np.abs(centers - 0.5) * 0 + (0.5 - np.abs(centers)), axis=1)
    
    return dist, wall_dist

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    dist, wall_dist = _pairwise_distances(centers)
    min_pair = np.min(dist)
    min_wall = np.min(wall_dist)
    return min(min_pair / 2.0, min_wall)

def _smooth_objective(centers, beta):
    """Smooth surrogate for the min-distance objective using log-sum-exp."""
    dist, wall_dist = _pairwise_distances(centers)
    
    # For pairs: we want to maximize min pair distance / 2
    # For walls: we want to maximize min wall distance
    # Combined: maximize min(min_pair/2, min_wall)
    
    # Use soft-min via log-sum-exp for smoothness
    # For pair distances (upper triangle only to avoid double counting)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Soft min of pair distances / 2 and wall distances
    all_vals = np.concatenate([pair_dists / 2.0, wall_dist])
    
    # Log-sum-exp for soft-min: -logsumexp(-x)
    m = np.min(all_vals)
    soft_min = -np.log(np.sum(np.exp(-all_vals - m))) - m
    
    return -soft_min  # We minimize the negative

def _smooth_gradient(centers, beta):
    """Gradient of the smooth objective."""
    dist, wall_dist = _pairwise_distances(centers)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    all_vals = np.concatenate([pair_dists / 2.0, wall_dist])
    m = np.min(all_vals)
    weights = np.exp(-all_vals - m)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(centers)
    
    # Gradient from pair distances
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx] / 2.0  # factor of 1/2 from pair_dists/2
        diff = centers[i] - centers[j]
        d = np.sqrt(np.sum(diff**2)) + 1e-15
        grad[i] -= w * diff / d
        grad[j] += w * diff / d
    
    # Gradient from wall distances
    for idx in range(N):
        w = weights[N + idx]
        c = centers[idx]
        # d/dx of (0.5 - |x|) = -sign(x)
        grad[idx, 0] -= w * np.sign(c[0])
        grad[idx, 1] -= w * np.sign(c[1])
        grad[idx, 2] -= w * np.sign(c[2])
    
    return grad

def _bcc_init(n=50):
    """Initialize with BCC lattice points, scaled to fit in cube."""
    # BCC: corners + body center
    # Generate a 3x3x3 BCC grid (27 points), then add more
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                points.append([i, j, k])
                # Body center
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points, dtype=float)
    # Scale to fit in [-0.5, 0.5]^3
    # Current range: [0, 3] x [0, 3] x [0, 3] -> shift and scale
    points = (points - 1.5) / 3.0 * 0.95  # slight inset
    
    # Select n points - use a greedy selection to maximize min distance
    selected = [0]
    for _ in range(n - 1):
        best_idx = -1
        best_min_dist = -1
        for idx in range(len(points)):
            if idx in selected:
                continue
            d = np.min(np.linalg.norm(points[idx] - points[selected], axis=1))
            if d > best_min_dist:
                best_min_dist = d
                best_idx = idx
        selected.append(best_idx)
    
    return points[selected]

def _random_init(n=50, seed=0):
    """Random initialization inside the cube."""
    rng = np.random.RandomState(seed)
    return (rng.rand(n, 3) - 0.5) * 0.95

def _optimize_single(centers, max_time=20.0):
    """Run L-BFGS-B optimization with annealing."""
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Annealing schedule
    betas = [10, 5, 3, 2, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05]
    
    current = centers.copy()
    for beta in betas:
        if time.time() - start_time > max_time - 2:
            break
        
        def objective(x):
            return _smooth_objective(x.reshape(N, 3), beta)
        
        def gradient(x):
            return _smooth_gradient(x.reshape(N, 3), beta).flatten()
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(
                objective, current.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            current = result.x.reshape(N, 3)
            r = _compute_radius(current)
            if r > best_radius:
                best_radius = r
                best_centers = current.copy()
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    """Construct centers for 50 spheres in unit cube maximizing radius."""
    global_start = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple initialization strategies
    inits = []
    
    # BCC-based initialization
    inits.append(_bcc_init(N))
    
    # Random initializations
    for seed in [0, 1, 2, 3, 4]:
        inits.append(_random_init(N, seed))
    
    # Cubic grid (fallback)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # Optimize each initialization
    for i, init in enumerate(inits):
        elapsed = time.time() - global_start
        if elapsed > deadline - 15:
            break
        
        # Time budget per init: decreasing as we go
        remaining = deadline - elapsed - 5
        per_init_time = min(20.0, remaining / (len(inits) - i))
        
        centers, radius = _optimize_single(init, max_time=per_init_time)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final polish on best result
    elapsed = time.time() - global_start
    if best_centers is not None and elapsed < deadline - 10:
        # High-precision polish with very low beta
        final_init = best_centers.copy()
        betas_final = [0.03, 0.02, 0.01, 0.005]
        current = final_init
        for beta in betas_final:
            if time.time() - global_start > deadline - 5:
                break
            
            def objective(x):
                return _smooth_objective(x.reshape(N, 3), beta)
            
            def gradient(x):
                return _smooth_gradient(x.reshape(N, 3), beta).flatten()
            
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                result = minimize(
                    objective, current.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
                )
                current = result.x.reshape(N, 3)
                r = _compute_radius(current)
                if r > best_radius:
                    best_radius = r
                    best_centers = current.copy()
            except Exception:
                pass
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
