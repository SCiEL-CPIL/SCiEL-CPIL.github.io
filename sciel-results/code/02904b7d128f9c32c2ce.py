# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for NaN or inf
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
                if min_area == 0:
                    return 0.0
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _neg_score(points: np.ndarray) -> float:
    """Negative of the score for minimization."""
    score = _compute_score(points)
    return -score


def _compute_score_fast(points: np.ndarray) -> float:
    """Faster score computation using vectorized triangle area calculation."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area calculation
    # For all triples i < j < k, compute area
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    
    # Precompute all pairs of differences
    # Use broadcasting to compute areas for all triples
    # Generate index arrays
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Compute vectors
    v1 = points[j_idx] - points[i_idx]  # (286, 2)
    v2 = points[k_idx] - points[i_idx]  # (286, 2)
    
    # Cross product (2D)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) / 2.0
    
    min_area = np.min(areas)
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _neg_score_fast(points: np.ndarray) -> float:
    """Negative of the fast score for minimization."""
    score = _compute_score_fast(points)
    return -score


def _lbfgsb_optimize(points: np.ndarray, deadline: float) -> np.ndarray:
    """Optimize points using L-BFGS-B with bounds [0, 1]."""
    n = points.shape[0]
    bounds = [(0.0, 1.0)] * (n * 2)
    
    # Flatten points for optimization
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        return _neg_score_fast(pts)
    
    # Use L-BFGS-B
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            return points
        
        return result.x.reshape(n, 2)
    except Exception:
        return points


def _simulated_annealing(points: np.ndarray, deadline: float) -> np.ndarray:
    """Perform simulated annealing on the point configuration."""
    n = points.shape[0]
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    current_points = points.copy()
    current_score = best_score
    
    T = 0.1  # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-6
    step_size = 0.05
    
    while time.time() < deadline and T > min_T:
        # Generate a random perturbation
        perturbed = current_points.copy()
        idx = np.random.randint(0, n)
        axis = np.random.randint(0, 2)
        delta = np.random.randn() * step_size
        perturbed[idx, axis] += delta
        
        # Clip to bounds
        np.clip(perturbed, 0, 1, out=perturbed)
        
        new_score = _compute_score_fast(perturbed)
        delta_score = new_score - current_score
        
        # Accept if better or with probability exp(delta_score / T)
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            current_points = perturbed
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cool down
        T *= cooling_rate
    
    return best_points


def _random_perturb(points: np.ndarray, magnitude: float = 0.1) -> np.ndarray:
    """Randomly perturb a point configuration."""
    perturbed = points.copy()
    noise = np.random.randn(*perturbed.shape) * magnitude
    perturbed += noise
    # Clip to [0, 1]
    np.clip(perturbed, 0, 1, out=perturbed)
    return perturbed


def _hexagonal_layout(n: int) -> np.ndarray:
    """Create a hexagonal lattice layout for n points."""
    points = []
    # Hexagonal lattice with spacing
    spacing = 0.25
    row_height = spacing * np.sqrt(3) / 2
    
    row = 0
    while len(points) < n:
        if row % 2 == 0:
            x_offset = 0.1
        else:
            x_offset = 0.1 + spacing / 2
        
        col = 0
        while len(points) < n:
            x = x_offset + col * spacing
            y = 0.1 + row * row_height
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
            col += 1
        row += 1
    
    return np.array(points[:n])


def _grid_layout(n: int) -> np.ndarray:
    """Create a grid layout for n points."""
    points = []
    # Try different grid sizes
    for cols in range(2, 6):
        rows = int(np.ceil(n / cols))
        for i in range(cols):
            for j in range(rows):
                if len(points) >= n:
                    break
                x = 0.1 + 0.8 * i / (cols - 1) if cols > 1 else 0.5
                y = 0.1 + 0.8 * j / (rows - 1) if rows > 1 else 0.5
                points.append([x, y])
        if len(points) >= n:
            break
    
    return np.array(points[:n])


def _concentric_layout(n: int) -> np.ndarray:
    """Create a concentric circle layout for n points."""
    points = [[0.5, 0.5]]  # Center point
    
    # Add points on concentric circles
    radii = [0.15, 0.3, 0.45]
    counts = [4, 6, 8]
    
    for r, count in zip(radii, counts):
        for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    return np.array(points[:n])


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Uses multiple initialization strategies followed by L-BFGS-B optimization and simulated annealing.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 570  # Safety margin: 30s before 600s limit
    
    best_points = None
    best_score = 0.0
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal layout
    initial_configs.append(_hexagonal_layout(n))
    
    # Config 2: Grid layout
    initial_configs.append(_grid_layout(n))
    
    # Config 3: Concentric layout
    initial_configs.append(_concentric_layout(n))
    
    # Config 4-13: Random initializations with different seeds
    for seed in range(10):
        rng = np.random.default_rng(seed=seed)
        random_pts = rng.random((n, 2)) * 0.9 + 0.05
        initial_configs.append(random_pts)
    
    # Evaluate all initial configs and keep the best
    for config in initial_configs:
        if time.time() > deadline - 120:
            break
        
        score = _compute_score_fast(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
    
    # Optimize the best configuration using L-BFGS-B
    if best_points is not None and time.time() < deadline - 60:
        optimized = _lbfgsb_optimize(best_points, deadline - 60)
        score = _compute_score_fast(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    # Try simulated annealing from the best point
    if best_points is not None and time.time() < deadline - 30:
        sa_result = _simulated_annealing(best_points, deadline - 30)
        score = _compute_score_fast(sa_result)
        if score > best_score:
            best_score = score
            best_points = sa_result.copy()
    
    # Multi-restart: perturb the best solution and optimize again
    if best_points is not None:
        for restart in range(20):
            if time.time() > deadline - 15:
                break
            
            perturbed = _random_perturb(best_points, magnitude=0.03 + 0.01 * restart)
            optimized = _lbfgsb_optimize(perturbed, deadline - 15)
            score = _compute_score_fast(optimized)
            
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
    
    # Final safety: ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback: simple grid
        best_points = _grid_layout(n)
    
    return best_points


# EVOLVE-BLOCK-END
