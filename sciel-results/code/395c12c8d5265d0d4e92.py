# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _surrogate_objective(points, p=10):
    """
    Smooth surrogate: use power-mean to approximate min/max ratio.
    Higher p makes it closer to true min/max.
    Returns negative of surrogate score for minimization.
    """
    d = pdist(points)
    if d_max == 0:
        return 0.0
    # Power mean approximation
    d_min_approx = (np.mean(d**p)) ** (1.0/p) * (1 - 0.1*(1 - p/100))  # slight bias toward min
    d_max_approx = (np.mean(d**p)) ** (1.0/p) * (1 + 0.1*(1 - p/100))  # slight bias toward max
    
    # Better: use actual min/max but smooth with log-sum-exp style
    # Actually, let's just use the real min/max - SA can handle it with proper temperature
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return -(d_min / d_max) ** 2


def _local_refine(points, n_iter=2000, seed=0):
    """
    Local refinement using coordinate descent with adaptive step sizes.
    Focus on improving the minimum distance while controlling maximum.
    """
    n = len(points)
    rng = np.random.default_rng(seed)
    points = points.copy()
    current_score = _compute_score(points)
    
    # Track best
    best_points = points.copy()
    best_score = current_score
    
    for it in range(n_iter):
        # Pick a random point
        idx = rng.integers(n)
        orig = points[idx].copy()
        
        # Compute distances from this point to all others
        dists = np.sqrt(np.sum((points - points[idx])**2, axis=1))
        dists[idx] = np.inf
        
        # Find nearest and farthest
        nearest = np.argmin(dists)
        farthest = np.argmax(dists)
        
        # Direction: away from nearest, toward center of farthest cluster
        dir_away = points[idx] - points[nearest]
        norm_away = np.linalg.norm(dir_away)
        if norm_away > 1e-10:
            dir_away = dir_away / norm_away
        else:
            dir_away = np.array([1.0, 0.0])
        
        # Try multiple step sizes in the away direction
        best_move = None
        best_score_candidate = current_score
        
        # Adaptive step sizes based on current min distance
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        base_step = d_min * 0.1 if d_min > 0 else 0.01
        
        for mult in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            step = base_step * mult
            new_pt = points[idx] + step * dir_away
            points[idx] = new_pt
            score = _compute_score(points)
            if score > best_score_candidate:
                best_score_candidate = score
                best_move = new_pt.copy()
        
        # Also try small random perturbations
        for _ in range(3):
            pert = rng.standard_normal(2) * base_step * 0.5
            new_pt = orig + pert
            points[idx] = new_pt
            score = _compute_score(points)
            if score > best_score_candidate:
                best_score_candidate = score
                best_move = new_pt.copy()
        
        if best_move is not None:
            points[idx] = best_move
            current_score = best_score_candidate
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idx] = orig
    
    return best_points, best_score


def _simulated_annealing(points, n_steps=50000, t_start=1.0, t_end=0.01, seed=0):
    """
    Simulated annealing for global exploration.
    """
    n = len(points)
    rng = np.random.default_rng(seed)
    points = points.copy()
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    n_steps_actual = min(n_steps, 50000)
    
    for step in range(n_steps_actual):
        # Temperature schedule: exponential cooling
        progress = step / n_steps_actual
        T = t_start * (t_end / t_start) ** progress
        
        # Pick a random point to move
        idx = rng.integers(n)
        orig = points[idx].copy()
        
        # Perturbation size proportional to temperature
        pert_size = T * 0.3
        perturbation = rng.standard_normal(2) * pert_size
        points[idx] = orig + perturbation
        
        new_score = _compute_score(points)
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / (T + 1e-10)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idx] = orig
    
    return best_points, best_score


def _generate_initial_configs(n=16, d=2, n_configs=10):
    """Generate diverse initial configurations."""
    configs = []
    
    # 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - grid.mean(axis=0)) / grid.std()
    configs.append(grid.copy())
    
    # Hexagonal lattice
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / hex_pts.std()
    configs.append(hex_pts.copy())
    
    # Concentric rings
    ring_configs = [
        [1, 3, 4, 8],
        [2, 3, 5, 6],
        [1, 4, 5, 6],
        [0, 4, 6, 6],
        [1, 5, 5, 5],
    ]
    for rings in ring_configs:
        pts = []
        for r, count in enumerate(rings):
            if count == 0:
                continue
            if r == 0:
                pts.append([0, 0])
            else:
                angles = np.linspace(0, 2*np.pi, count, endpoint=False)
                for a in angles:
                    pts.append([r * np.cos(a), r * np.sin(a)])
        if len(pts) == n:
            pts = np.array(pts)
            pts = (pts - pts.mean(axis=0)) / pts.std()
            configs.append(pts.copy())
    
    # Random
    rng = np.random.default_rng(42)
    while len(configs) < n_configs:
        pts = rng.standard_normal((n, d))
        pts = (pts - pts.mean(axis=0)) / pts.std()
        configs.append(pts)
    
    return configs


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Generate initial configurations
    configs = _generate_initial_configs(n, d, n_configs=12)
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Local refine each config
    for i, config in enumerate(configs):
        if time.time() > deadline - 120:
            break
        points, score = _local_refine(config, n_iter=1500, seed=i)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 2: Simulated annealing from best config
    if best_points is not None and time.time() < deadline - 60:
        points, score = _simulated_annealing(best_points, n_steps=30000, t_start=0.5, t_end=0.001, seed=123)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 3: Final local refinement
    if best_points is not None and time.time() < deadline - 10:
        points, score = _local_refine(best_points, n_iter=3000, seed=999)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
