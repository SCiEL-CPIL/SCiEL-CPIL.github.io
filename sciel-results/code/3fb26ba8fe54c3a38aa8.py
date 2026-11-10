# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _compute_score_fast(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using fully vectorized triangle areas."""
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
    
    # Fully vectorized: compute all C(n,3) triangle areas at once
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n-2), np.arange(1, n-1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = np.abs(v1x * v2y - v1y * v2x)
    min_area = float(np.min(cross)) * 0.5
    
    if min_area <= 0:
        return 0.0
    
    return min_area / A_hull


def _find_worst_triangle(points: np.ndarray) -> tuple:
    """Find the triple (i,j,k) with minimum triangle area and its area."""
    n = points.shape[0]
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n-2), np.arange(1, n-1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = np.abs(v1x * v2y - v1y * v2x)
    min_idx = np.argmin(cross)
    min_area = cross[min_idx] * 0.5
    
    return (i_idx[min_idx], j_idx[min_idx], k_idx[min_idx]), min_area


def _local_search(points: np.ndarray, deadline: float, max_iters: int = 30000) -> np.ndarray:
    """Local search that directly optimizes the worst triangle area."""
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    n = len(best_points)
    
    step_size = 0.03
    min_step = 1e-10
    decay = 0.95
    stall_count = 0
    max_stall = 50
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        (wi, wj, wk), min_area = _find_worst_triangle(best_points)
        
        improved = False
        for idx in (wi, wj, wk):
            for _ in range(6):
                dx = np.random.randn() * step_size
                dy = np.random.randn() * step_size
                
                candidate = best_points.copy()
                candidate[idx, 0] = np.clip(best_points[idx, 0] + dx, 0, 1)
                candidate[idx, 1] = np.clip(best_points[idx, 1] + dy, 0, 1)
                
                score = _compute_score_fast(candidate)
                if score > best_score:
                    best_points = candidate
                    best_score = score
                    improved = True
                    stall_count = 0
                    break
            if improved:
                break
        
        if not improved:
            stall_count += 1
            if stall_count >= max_stall:
                step_size *= decay
                if step_size < min_step:
                    break
                stall_count = 0
                max_stall = max(20, max_stall // 2)
    
    return best_points


def _simulated_annealing(points: np.ndarray, deadline: float,
                          T_init: float = 0.05, T_min: float = 1e-8,
                          cooling_rate: float = 0.998, max_iters: int = 50000) -> np.ndarray:
    """Simulated annealing with adaptive step size."""
    best_points = points.copy()
    current_points = points.copy()
    current_score = _compute_score_fast(current_points)
    best_score = current_score
    
    n = len(current_points)
    step_size = 0.04
    min_step = 1e-9
    reject_count = 0
    accept_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        T = max(T_init * (cooling_rate ** it), T_min)
        
        idx = np.random.randint(n)
        dx = np.random.randn() * step_size
        dy = np.random.randn() * step_size
        
        candidate = current_points.copy()
        candidate[idx, 0] = np.clip(current_points[idx, 0] + dx, 0, 1)
        candidate[idx, 1] = np.clip(current_points[idx, 1] + dy, 0, 1)
        
        candidate_score = _compute_score_fast(candidate)
        
        if candidate_score >= current_score:
            current_points = candidate
            current_score = candidate_score
            accept_count += 1
            if current_score > best_score:
                best_score = current_score
                best_points = candidate.copy()
        else:
            delta = current_score - candidate_score
            if np.random.random() < np.exp(-delta / max(T, 1e-12)):
                current_points = candidate
                current_score = candidate_score
                accept_count += 1
            else:
                reject_count += 1
        
        if (accept_count + reject_count) % 200 == 0 and (accept_count + reject_count) > 0:
            acc_rate = accept_count / (accept_count + reject_count)
            if acc_rate < 0.2:
                step_size *= 0.8
            elif acc_rate > 0.8:
                step_size *= 1.2
            step_size = max(min_step, min(step_size, 0.1))
            accept_count = 0
            reject_count = 0
    
    return best_points


def _regular_polygon_init(n: int) -> np.ndarray:
    """Regular n-gon centered at (0.5, 0.5) with radius 0.45."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 0.45
    xs = 0.5 + r * np.cos(angles)
    ys = 0.5 + r * np.sin(angles)
    return np.column_stack([xs, ys])


def _hexagonal_init(n: int) -> np.ndarray:
    """Hexagonal lattice initialization."""
    points = []
    spacing = 0.25
    row_height = spacing * np.sqrt(3) / 2
    
    row = 0
    while len(points) < n:
        y = 0.1 + row * row_height
        if row % 2 == 0:
            cols = 4
            x_start = 0.1
        else:
            cols = 3
            x_start = 0.1 + spacing / 2
        
        for col in range(cols):
            if len(points) >= n:
                break
            x = x_start + col * spacing
            points.append([x, y])
        row += 1
    
    return np.array(points[:n])


def _grid_init(n: int) -> np.ndarray:
    """Grid initialization."""
    points = []
    spacing = 0.2
    for i in range(5):
        for j in range(5):
            if len(points) >= n:
                break
            x = 0.1 + i * spacing
            y = 0.1 + j * spacing
            points.append([x, y])
    return np.array(points[:n])


def _concentric_init(n: int) -> np.ndarray:
    """Concentric circle initialization."""
    points = [[0.5, 0.5]]
    for r in [0.15, 0.3, 0.4]:
        num_pts = int(2 * np.pi * r / 0.2)
        for angle in np.linspace(0, 2 * np.pi, num_pts, endpoint=False):
            if len(points) >= n:
                break
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            points.append([x, y])
    return np.array(points[:n])


def _random_perturb(points: np.ndarray, magnitude: float = 0.1) -> np.ndarray:
    """Randomly perturb a point configuration."""
    perturbed = points.copy()
    noise = np.random.randn(*perturbed.shape) * magnitude
    perturbed += noise
    np.clip(perturbed, 0, 1, out=perturbed)
    return perturbed


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 570  # Safety margin: 30s before 600s limit
    
    best_points = None
    best_score = 0.0
    
    # Initial configurations
    initial_configs = [
        _regular_polygon_init(n),
        _hexagonal_init(n),
        _grid_init(n),
        _concentric_init(n),
    ]
    
    # Add random initializations
    for seed in range(15):
        rng = np.random.default_rng(seed=seed)
        random_pts = rng.random((n, 2)) * 0.9 + 0.05
        initial_configs.append(random_pts)
    
    # Evaluate all initial configs
    for config in initial_configs:
        if time.time() > deadline:
            break
        score = _compute_score_fast(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
    
    # Phase 1: Local search from best initial
    if best_points is not None:
        refined = _local_search(best_points, deadline, max_iters=10000)
        score = _compute_score_fast(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
    
    # Phase 2: Simulated annealing from best
    if best_points is not None and time.time() < deadline - 60:
        sa_result = _simulated_annealing(best_points, deadline, max_iters=30000)
        score = _compute_score_fast(sa_result)
        if score > best_score:
            best_score = score
            best_points = sa_result.copy()
    
    # Phase 3: Multi-restart with perturbations
    if best_points is not None:
        for restart in range(20):
            if time.time() > deadline - 30:
                break
            perturbed = _random_perturb(best_points, magnitude=0.03 + 0.01 * restart)
            refined = _local_search(perturbed, deadline, max_iters=3000)
            score = _compute_score_fast(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
    
    # Phase 4: Final polish with tighter local search
    if best_points is not None and time.time() < deadline - 10:
        polished = _local_search(best_points, deadline, max_iters=5000)
        score = _compute_score_fast(polished)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    # Safety fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon_init(n)
    
    return best_points


# EVOLVE-BLOCK-END
