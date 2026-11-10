# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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


def _local_search(points: np.ndarray, deadline: float, max_iters: int = 20000) -> np.ndarray:
    """Local search that directly optimizes the worst triangle area."""
    best_points = points.copy()
    best_score = _compute_score_fast(best_points)
    n = len(best_points)
    
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n-2), np.arange(1, n-1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
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
            for _ in range(4):
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
                          cooling_rate: float = 0.998, max_iters: int = 30000) -> np.ndarray:
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


def _smooth_polish(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    Smooth gradient-based polish using log-sum-exp soft-min of triangle areas.
    Uses multi-scale beta approach: start with low beta (smooth) then increase.
    """
    n = points.shape[0]
    
    # Precompute all triangle index triples
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n-2), np.arange(1, n-1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    def _soft_min_objective(flat, beta=50.0):
        """
        Objective: minimize -soft_min(triangle_areas) / A_hull
        Using soft-min via log-sum-exp for smoothness.
        """
        pts = flat.reshape(n, 2)
        pts = np.clip(pts, 1e-6, 1 - 1e-6)
        
        pi = pts[i_idx]
        pj = pts[j_idx]
        pk = pts[k_idx]
        
        v1x = pj[:, 0] - pi[:, 0]
        v1y = pj[:, 1] - pi[:, 1]
        v2x = pk[:, 0] - pi[:, 0]
        v2y = pk[:, 1] - pi[:, 1]
        
        cross = np.abs(v1x * v2y - v1y * v2x) * 0.5  # triangle areas
        
        # Soft min via log-sum-exp (negative for min)
        max_area = np.max(cross)
        log_sum_exp = max_area + np.log(np.sum(np.exp(beta * (cross - max_area))))
        soft_min = -log_sum_exp / beta
        
        # Compute hull area
        try:
            hull = ConvexHull(pts)
            A_hull = hull.volume
        except Exception:
            return 1e10  # penalty for degenerate hull
        
        if A_hull <= 1e-10:
            return 1e10
        
        # We want to maximize A_min / A_hull, so minimize negative
        return -soft_min / A_hull
    
    # Initial point
    x0 = points.flatten()
    
    best_result = None
    best_fun = _soft_min_objective(x0, beta=10.0)
    
    # Multi-scale beta approach: coarse to fine
    # Start with low beta (smooth landscape), then increase
    beta_schedule = [5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30:
            break
        
        # For each beta, try multiple L-BFGS-B runs with different bounds
        bounds_list = [
            [(1e-6, 1-1e-6)] * 2 * n,
            [(0.01, 0.99)] * 2 * n,
            [(0.05, 0.95)] * 2 * n,
        ]
        
        for bounds in bounds_list:
            if time.time() > deadline - 20:
                break
            
            try:
                result = minimize(
                    lambda x: _soft_min_objective(x, beta=beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result.x.copy()
                    x0 = result.x.copy()  # Use best result as starting point for next
            except Exception:
                continue
    
    if best_result is not None:
        polished = best_result.reshape(n, 2)
        polished = np.clip(polished, 0, 1)
        # Verify it's valid
        if np.all(np.isfinite(polished)):
            try:
                ConvexHull(polished)
                return polished
            except Exception:
                pass
    
    return points


def _hexagonal_initialization(n: int) -> np.ndarray:
    """Create a hexagonal lattice initialization."""
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


def _grid_initialization(n: int) -> np.ndarray:
    """Create a grid initialization."""
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


def _concentric_initialization(n: int) -> np.ndarray:
    """Create a concentric circle initialization."""
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


def _random_initialization(n: int, seed: int) -> np.ndarray:
    """Create a random initialization."""
    rng = np.random.default_rng(seed=seed)
    return rng.random((n, 2)) * 0.8 + 0.1


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Uses multiple initialization strategies, simulated annealing, local search,
    and smooth gradient-based polish.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600
    
    best_points = None
    best_score = 0.0
    
    # Generate multiple initial configurations
    initial_configs = []
    
    initial_configs.append(_hexagonal_initialization(n))
    initial_configs.append(_grid_initialization(n))
    initial_configs.append(_concentric_initialization(n))
    
    for seed in range(15):
        initial_configs.append(_random_initialization(n, seed))
    
    # Phase 1: Quick evaluation and selection
    scored_configs = []
    for config in initial_configs:
        if time.time() > deadline - 450:
            break
        score = _compute_score_fast(config)
        scored_configs.append((score, config))
    
    scored_configs.sort(key=lambda x: -x[0])
    top_configs = [config for _, config in scored_configs[:8]]
    
    # Phase 2: Simulated annealing on top candidates
    sa_deadline = deadline - 200
    for config in top_configs:
        if time.time() > sa_deadline:
            break
        
        sa_result = _simulated_annealing(config, sa_deadline,
                                           T_init=0.04, T_min=1e-8,
                                           cooling_rate=0.9985, max_iters=25000)
        sa_score = _compute_score_fast(sa_result)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_result.copy()
    
    # Phase 3: Local search refinement
    if best_points is not None:
        ls_deadline = deadline - 100
        ls_result = _local_search(best_points, ls_deadline, max_iters=30000)
        ls_score = _compute_score_fast(ls_result)
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_result.copy()
    
    # Phase 4: Smooth gradient-based polish (key innovation - multi-scale beta)
    if best_points is not None:
        polish_deadline = deadline - 40
        polished = _smooth_polish(best_points, polish_deadline)
        polish_score = _compute_score_fast(polished)
        
        if polish_score > best_score:
            best_score = polish_score
            best_points = polished.copy()
    
    # Phase 5: Final local search after polish
    if best_points is not None and time.time() < deadline - 25:
        final_ls = _local_search(best_points, deadline - 10, max_iters=15000)
        final_score = _compute_score_fast(final_ls)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_ls.copy()
    
    # Phase 6: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 50:
        for restart in range(5):
            if time.time() > deadline - 25:
                break
            
            perturbed = best_points.copy()
            noise = np.random.randn(*perturbed.shape) * 0.05
            perturbed += noise
            np.clip(perturbed, 0, 1, out=perturbed)
            
            quick_ls = _local_search(perturbed, deadline - 25, max_iters=5000)
            quick_score = _compute_score_fast(quick_ls)
            
            if quick_score > best_score:
                best_score = quick_score
                best_points = quick_ls.copy()
    
    # Final safety check
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _grid_initialization(n)
    
    return best_points


# EVOLVE-BLOCK-END
