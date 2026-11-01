"""Circle packing in unit square - maximize minimum distance to boundary and between circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _soft_min_neg(x, beta=50.0):
    """Smooth approximation of -min(x) using log-sum-exp."""
    # -min(x) = -(-max(-x)) = max(-x)
    # max(-x) ≈ (1/beta) * log(sum(exp(beta * (-x))))
    neg_x = -x
    max_val = np.max(neg_x)
    if max_val > 50:
        return -max_val  # avoid overflow
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * (neg_x - max_val)))) + max_val

def _compute_radius(centers):
    """Compute the actual radius (min distance to boundary and half min pairwise distance)."""
    centers = np.asarray(centers)
    # Distance to boundary
    boundary_dists = np.min(np.array([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ]), axis=1)
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        min_pair = np.inf
    else:
        # Compute all pairwise distances efficiently
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _objective(centers_flat, beta=50.0):
    """Objective to minimize: -soft_min of all constraints (pairwise half-dists and boundary dists)."""
    centers = centers_flat.reshape(-1, 2)
    
    # Boundary distances
    boundary_dists = np.min(np.array([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ]), axis=1)
    
    # Pairwise half-distances
    n = len(centers)
    if n < 2:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0
    
    # Collect all constraint values
    all_constraints = np.concatenate([boundary_dists, half_dists[half_dists < np.inf]])
    
    if len(all_constraints) == 0:
        return 0.0
    
    return _soft_min_neg(all_constraints, beta)

def _hex_init(offset_x=0.0, offset_y=0.0, scale=1.0):
    """Generate a hexagonal lattice initialization."""
    # Hexagonal lattice: spacing s, rows offset by s/2
    s = 0.18  # initial spacing, will be refined
    rows = []
    y = -0.5 + offset_y
    row_idx = 0
    while y <= 0.5:
        x_offset = (s / 2) if (row_idx % 2 == 1) else 0.0
        x = -0.5 + offset_x + x_offset
        while x <= 0.5:
            rows.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row_idx += 1
    
    points = np.array(rows)
    
    # If we have too many, randomly select N
    if len(points) >= N:
        # Prefer points near center for better packing
        center_dists = np.sum(points**2, axis=1)
        idx = np.argsort(center_dists)[:N]
        points = points[idx]
    elif len(points) > 0:
        # Pad with random points if needed
        extra = np.random.randn(N - len(points), 2) * 0.05
        points = np.vstack([points, extra])
    else:
        points = np.random.randn(N, 2) * 0.4
    
    return points[:N]

def _optimize_from_init(init_centers, max_iters=500, tol=1e-8):
    """Run local optimization from initial configuration."""
    x0 = init_centers.flatten()
    
    result = minimize(
        _objective,
        x0,
        method='BFGS',
        options={'maxiter': max_iters, 'gtol': tol, 'disp': False}
    )
    
    if result.success:
        best_centers = result.x.reshape(-1, 2)
    else:
        best_centers = x0.reshape(-1, 2)
    
    return best_centers

def _polish(centers):
    """Final high-precision polish with tighter tolerance."""
    x0 = centers.flatten()
    result = minimize(
        _objective,
        x0,
        method='BFGS',
        options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False}
    )
    
    if result.success:
        return result.x.reshape(-1, 2)
    return centers

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_centers = None
    best_radius = -np.inf
    
    # Multiple restarts with different hexagonal initializations
    np.random.seed(42)
    
    restart_configs = [
        # (offset_x, offset_y, description)
        (0.0, 0.0, "centered"),
        (0.05, 0.05, "shifted1"),
        (-0.05, 0.05, "shifted2"),
        (0.05, -0.05, "shifted3"),
        (-0.05, -0.05, "shifted4"),
        (0.03, 0.07, "shifted5"),
        (0.07, 0.03, "shifted6"),
        (0.0, 0.03, "shifted7"),
        (0.03, 0.0, "shifted8"),
        (0.02, 0.02, "shifted9"),
    ]
    
    for i, (ox, oy, desc) in enumerate(restart_configs):
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - 10:  # Leave 10s safety margin
            break
        
        init = _hex_init(ox, oy)
        centers = _optimize_from_init(init, max_iters=300)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        # Check time
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - 20:
            break
    
    # If we have time, do a polish on the best solution
    elapsed = time.time() - START_TIME
    if best_centers is not None and elapsed < TIME_LIMIT - 15:
        polished = _polish(best_centers)
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_centers = polished
            best_radius = polished_radius
    
    # Final safety check
    elapsed = time.time() - START_TIME
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Ensure valid output
    best_centers = np.asarray(best_centers, dtype=np.float64)
    if best_centers.shape != (N, 2):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Clip to ensure within bounds (with small margin to avoid numerical issues)
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.499, 0.499)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.499, 0.499)
    
    return best_centers
