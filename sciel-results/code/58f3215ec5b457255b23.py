"""Optimized 50-circle packing using grid-based neighbor lookup and L-BFGS-B relaxation."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _build_grid(points, cell_size):
    """Build a spatial grid for fast neighbor lookup."""
    grid = {}
    for i in range(len(points)):
        ix = int(np.floor(points[i, 0] / cell_size))
        iy = int(np.floor(points[i, 1] / cell_size))
        key = (ix, iy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid


def _get_neighbors(grid, x, y, cell_size):
    """Get indices of points in the 3x3 neighborhood of (x, y)."""
    ix = int(np.floor(x / cell_size))
    iy = int(np.floor(y / cell_size))
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (ix + dx, iy + dy)
            if key in grid:
                neighbors.extend(grid[key])
    return neighbors


def _objective_and_grad(flat, cell_size=0.28):
    """
    Objective: minimize -min_radius (i.e., maximize the packing radius).
    Uses a smooth approximation via log-sum-exp of violations.
    Returns (objective_value, gradient).
    """
    points = flat.reshape(N, 2)
    n = N
    grad = np.zeros_like(flat)
    
    # Compute all pairwise distances and boundary distances
    # For efficiency, use grid-based lookup
    grid = _build_grid(points, cell_size)
    
    # Collect all "constraints": pair distances and boundary margins
    # We want to maximize min(all_pair_dists/2, all_boundary_margins)
    # Equivalent to minimizing -min(...)
    # Use smooth max: -logsumexp(-log(d_i)) approximates -min(d_i)
    
    # Simpler: use a soft-min approximation
    # min(x) ≈ -1/beta * log(sum(exp(-beta * x_i))) for large beta
    beta = 50.0
    
    # Collect distances
    dists = []
    boundary_margins = []
    
    for i in range(n):
        x_i, y_i = points[i]
        r_i = np.hypot(x_i, y_i)
        boundary_margins.append(1.0 - r_i)
        
        neighbors = _get_neighbors(grid, x_i, y_i, cell_size)
        for j in neighbors:
            if j <= i:
                continue
            x_j, y_j = points[j]
            d = np.hypot(x_i - x_j, y_i - y_j)
            dists.append(d)
    
    # Convert to radius candidates
    radius_candidates = np.array(dists) / 2.0
    boundary_candidates = np.array(boundary_margins)
    
    # Combine all candidates
    all_candidates = np.concatenate([radius_candidates, boundary_candidates])
    
    # Handle zeros/negatives
    all_candidates = np.maximum(all_candidates, 1e-10)
    
    # Smooth minimum via log-sum-exp
    neg_log_sum = -np.log(np.sum(np.exp(-beta * all_candidates)))
    min_val = -neg_log_sum / beta
    
    # Gradient of the smooth minimum w.r.t. each candidate
    # d/dx_i of (-1/beta * log(sum(exp(-beta * x_j)))) = exp(-beta * x_i) / sum(exp(-beta * x_j))
    exp_neg_beta = np.exp(-beta * all_candidates)
    weights = exp_neg_beta / np.sum(exp_neg_beta)
    
    # Now propagate gradients back to points
    # For pair distances: d/dpoints of (d_ij / 2)
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # So d(min)/dx_i gets contribution from weight * (x_i - x_j) / (2 * d_ij)
    
    # Rebuild to propagate gradients
    grid2 = _build_grid(points, cell_size)
    
    idx = 0
    for i in range(n):
        x_i, y_i = points[i]
        r_i = np.hypot(x_i, y_i)
        
        # Boundary margin gradient
        w_b = weights[idx]
        idx += 1
        if r_i > 1e-10:
            # d(1 - r_i)/dx_i = -x_i/r_i
            grad[2*i] += -w_b * (-x_i / r_i)
            grad[2*i+1] += -w_b * (-y_i / r_i)
        
        # Pair distance gradients
        neighbors = _get_neighbors(grid2, x_i, y_i, cell_size)
        for j in neighbors:
            if j <= i:
                continue
            x_j, y_j = points[j]
            dx = x_i - x_j
            dy = y_i - y_j
            d = np.hypot(dx, dy)
            w_p = weights[idx]
            idx += 1
            if d > 1e-10:
                # d(d_ij/2)/dx_i = dx/(2*d)
                grad[2*i] += w_p * dx / (2.0 * d)
                grad[2*i+1] += w_p * dy / (2.0 * d)
                # d(d_ij/2)/dx_j = -dx/(2*d)
                grad[2*j] += w_p * (-dx) / (2.0 * d)
                grad[2*j+1] += w_p * (-dy) / (2.0 * d)
    
    # We want to maximize min, so minimize -min
    objective = -min_val
    
    return objective, -grad


def _compute_radius(points):
    """Compute the exact packing radius for a configuration."""
    n = len(points)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
            if d < min_dist:
                min_dist = d
    min_boundary = np.min(1.0 - np.hypot(points[:, 0], points[:, 1]))
    return min(min_dist / 2.0, min_boundary)


def _ring_initialization(ring_radii, ring_counts, seed=42):
    """Generate initial configuration with points on concentric rings."""
    rng = np.random.RandomState(seed)
    points = []
    for ring_r, count in zip(ring_radii, ring_counts):
        if count > 0:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False) + rng.uniform(0, 2*np.pi/count, 1)[0]
            for a in angles:
                points.append([ring_r * np.cos(a), ring_r * np.sin(a)])
    
    # Add center if needed
    while len(points) < N:
        points.append([0.0, 0.0])
    
    points = np.array(points[:N])
    return points


def _random_initialization(seed=42):
    """Generate a random initial configuration with some structure."""
    rng = np.random.RandomState(seed)
    points = []
    # Place points in a spiral-like pattern
    for i in range(N):
        t = i / N
        angle = t * 2 * np.pi * 3.0 + rng.uniform(-0.1, 0.1)
        r = 0.1 + 0.85 * t + rng.uniform(-0.05, 0.05)
        r = min(r, 0.95)
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)


def _hexagonal_initialization(spacing=0.255):
    """Generate hexagonal lattice initialization."""
    points = []
    r_max = 1.0 - spacing / 2.0
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= r_max:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    return points[:N]


def _optimize(points, max_iters=2000, ftol=1e-12, cell_size=0.28):
    """Run L-BFGS-B optimization on a starting configuration."""
    flat = points.flatten()
    
    # Bounds: keep points within unit circle (with small margin)
    margin = 0.01
    bounds = [(-1.0 + margin, 1.0 - margin)] * (2 * N)
    
    result = minimize(
        _objective_and_grad,
        flat,
        args=(cell_size,),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iters, 'ftol': ftol, 'gtol': 1e-8}
    )
    
    opt_points = result.x.reshape(N, 2)
    
    # Ensure points are strictly inside
    radii = np.hypot(opt_points[:, 0], opt_points[:, 1])
    max_r = np.max(radii)
    if max_r > 0.9999:
        opt_points *= 0.9999 / max_r
    
    return opt_points


def construct():
    """
    Construct the optimal packing of 50 equal circles in the unit disk.
    Returns array of shape (50, 2) with circle centers.
    """
    np.random.seed(42)
    
    best_points = None
    best_radius = -1.0
    
    # Multiple initializations
    initializations = []
    
    # 1. Hexagonal lattice
    initializations.append(_hexagonal_initialization(0.255))
    
    # 2. Ring-based: center + 2 rings
    initializations.append(_ring_initialization(
        [0.0, 0.28, 0.56, 0.82],
        [1, 7, 12, 18],
        seed=42
    ))
    
    # 3. Ring-based: different counts
    initializations.append(_ring_initialization(
        [0.0, 0.25, 0.50, 0.75, 0.92],
        [1, 6, 10, 14, 19],
        seed=123
    ))
    
    # 4. Random spiral
    initializations.append(_random_initialization(seed=42))
    
    # 5. Random spiral different seed
    initializations.append(_random_initialization(seed=999))
    
    # 6. Hexagonal with different spacing
    initializations.append(_hexagonal_initialization(0.26))
    
    for init in initializations:
        try:
            opt = _optimize(init, max_iters=1500, cell_size=0.28)
            r = _compute_radius(opt)
            if r > best_radius:
                best_radius = r
                best_points = opt.copy()
        except Exception:
            continue
    
    # Final polish on the best configuration with tighter tolerance
    if best_points is not None:
        try:
            polished = _optimize(best_points, max_iters=2000, ftol=1e-14, cell_size=0.25)
            r = _compute_radius(polished)
            if r > best_radius:
                best_radius = r
                best_points = polished
        except Exception:
            pass
    
    # Ensure validity: all points strictly inside, no coincident points
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    max_r = np.max(radii)
    if max_r > 0.99999:
        best_points *= 0.99999 / max_r
    
    # Check for coincident points and separate them
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_points[i, 0] - best_points[j, 0],
                         best_points[i, 1] - best_points[j, 1])
            if d < 1e-8:
                # Push apart
                angle = np.random.uniform(0, 2 * np.pi)
                push = 0.01
                best_points[i, 0] += push * np.cos(angle)
                best_points[i, 1] += push * np.sin(angle)
                best_points[j, 0] -= push * np.cos(angle)
                best_points[j, 1] -= push * np.sin(angle)
    
    return best_points
# EVOLVE-BLOCK-END
