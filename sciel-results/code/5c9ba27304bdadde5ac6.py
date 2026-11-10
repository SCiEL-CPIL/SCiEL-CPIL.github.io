# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _compute_metric(points: np.ndarray) -> float:
    """Compute (d_min / d_max)^2 for the given points."""
    if points.shape[0] < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    if d_min <= 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _fast_metric(points: np.ndarray) -> float:
    """Faster metric computation using vectorized distances."""
    n = points.shape[0]
    sq_dists = np.sum(points**2, axis=1)[:, np.newaxis] + np.sum(points**2, axis=1)[np.newaxis, :] - 2 * points @ points.T
    np.fill_diagonal(sq_dists, 0.0)
    sq_dists = np.maximum(sq_dists, 0.0)
    mask = sq_dists > 1e-12
    if not np.any(mask):
        return 0.0
    d_min_sq = np.min(sq_dists[mask])
    d_max_sq = np.max(sq_dists)
    if d_max_sq <= 0:
        return 0.0
    d_min = np.sqrt(d_min_sq)
    d_max = np.sqrt(d_max_sq)
    if d_min <= 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _repulsive_objective(x, n=14, d=3):
    """
    Objective: maximize sum of log(pairwise distances) with a penalty for max distance.
    We minimize the negative of this.
    """
    pts = x.reshape(n, d)
    # Center
    pts = pts - np.mean(pts, axis=0)
    
    # Compute all pairwise squared distances
    sq_dists = np.sum(pts**2, axis=1)[:, np.newaxis] + np.sum(pts**2, axis=1)[np.newaxis, :] - 2 * pts @ pts.T
    np.fill_diagonal(sq_dists, 0.0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Upper triangular indices
    iu = np.triu_indices(n, k=1)
    pair_sq = sq_dists[iu]
    
    # Log-sum of distances (promotes uniform spacing)
    log_sum = np.sum(np.log(pair_sq))
    
    # Penalty for max distance (keep diameter bounded)
    max_sq = np.max(pair_sq)
    # Soft barrier: penalize if max distance exceeds 1
    barrier = 100.0 * np.maximum(0.0, max_sq - 1.0)
    
    # Also penalize if min distance is too small (coincident points)
    min_sq = np.min(pair_sq)
    min_penalty = 10.0 * np.log(min_sq + 1e-10)
    
    return -(log_sum + min_penalty) + barrier


def _repulsive_gradient(x, n=14, d=3):
    """Gradient of the repulsive objective."""
    pts = x.reshape(n, d)
    pts = pts - np.mean(pts, axis=0)
    
    sq_dists = np.sum(pts**2, axis=1)[:, np.newaxis] + np.sum(pts**2, axis=1)[np.newaxis, :] - 2 * pts @ pts.T
    np.fill_diagonal(sq_dists, 0.0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    iu = np.triu_indices(n, k=1)
    pair_sq = sq_dists[iu]
    
    # Gradient of log_sum w.r.t. each point
    grad = np.zeros((n, d))
    # d/d(pts_i) of log(|pts_i - pts_j|^2) = 2*(pts_i - pts_j)/|pts_i - pts_j|^2
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        diff = pts[i] - pts[j]
        dist_sq = pair_sq[idx]
        grad[i] += 2.0 * diff / dist_sq
        grad[j] -= 2.0 * diff / dist_sq
    
    # Gradient of min_penalty: 10 * log(min_sq + eps)
    # The min pair
    min_idx = np.argmin(pair_sq)
    i_min, j_min = iu[0][min_idx], iu[1][min_idx]
    diff_min = pts[i_min] - pts[j_min]
    min_sq = pair_sq[min_idx]
    grad[i_min] += 10.0 * 2.0 * diff_min / (min_sq + 1e-10)
    grad[j_min] += 10.0 * (-2.0 * diff_min) / (min_sq + 1e-10)
    
    # Barrier gradient for max distance
    max_idx = np.argmax(pair_sq)
    i_max, j_max = iu[0][max_idx], iu[1][max_idx]
    if pair_sq[max_idx] > 1.0:
        diff_max = pts[i_max] - pts[j_max]
        grad[i_max] -= 200.0 * diff_max
        grad[j_max] += 200.0 * diff_max
    
    return grad.flatten()


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    
    best_points = None
    best_score = -1.0
    
    def _try_candidate(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            return
        if not np.all(np.isfinite(pts)):
            return
        score = _fast_metric(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # --- Initializations: diverse starting points ---
    
    # 1. Fibonacci sphere
    golden_ratio = (1 + np.sqrt(5)) / 2
    pts_fib = np.zeros((n, d))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        pts_fib[i, 0] = r * np.cos(theta)
        pts_fib[i, 1] = r * np.sin(theta)
        pts_fib[i, 2] = z
    _try_candidate(pts_fib)
    
    # 2. Icosahedron + 2 poles
    phi = (1 + 5**0.5) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=np.float64)
    ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norms
    pts_ico = np.vstack([ico, np.array([[0, 0, 1]], dtype=np.float64), np.array([[0, 0, -1]], dtype=np.float64)])
    _try_candidate(pts_ico)
    
    # 3. Random on sphere (multiple seeds)
    for seed in range(10):
        np.random.seed(seed * 1000 + 42)
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        _try_candidate(pts)
    
    # --- Main optimization: L-BFGS-B with repulsive potential ---
    
    if best_points is not None:
        # Use best initial point as starting point for optimization
        x0 = (best_points - np.mean(best_points, axis=0)).flatten()
        
        # Run L-BFGS-B
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            try:
                res = minimize(
                    _repulsive_objective,
                    x0,
                    jac=_repulsive_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                candidate = res.x.reshape(n, d)
                candidate = candidate - np.mean(candidate, axis=0)
                _try_candidate(candidate)
            except Exception:
                pass
        
        # Multiple restarts from different initializations
        np.random.seed(777)
        for restart in range(15):
            if time.time() > deadline - 30:
                break
            
            # Generate diverse starting points
            if restart < 5:
                # Perturbed icosahedron
                pts_start = ico.copy()
                extra = np.random.randn(2, d) * 0.5
                pts_start = np.vstack([pts_start, extra])
            elif restart < 10:
                # Random sphere
                pts_start = np.random.randn(n, d)
                norms = np.linalg.norm(pts_start, axis=1, keepdims=True)
                pts_start = pts_start / norms
            else:
                # Perturbed Fibonacci
                pts_start = pts_fib.copy()
                pts_start += np.random.randn(n, d) * 0.2
            
            x0_r = (pts_start - np.mean(pts_start, axis=0)).flatten()
            
            try:
                res = minimize(
                    _repulsive_objective,
                    x0_r,
                    jac=_repulsive_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
                )
                candidate = res.x.reshape(n, d)
                candidate = candidate - np.mean(candidate, axis=0)
                _try_candidate(candidate)
            except Exception:
                pass
    
    # --- Local polishing: coordinate descent on the metric ---
    
    if best_points is not None:
        current_pts = best_points.copy()
        current_score = best_score
        step_size = 0.05
        no_improve_count = 0
        
        for iter in range(30000):
            if time.time() > deadline - 10:
                break
            if no_improve_count > 2000:
                break
            
            # Random coordinate perturbation
            dim_idx = np.random.randint(n * d)
            point_idx = dim_idx // d
            coord_idx = dim_idx % d
            old_val = current_pts[point_idx, coord_idx]
            current_pts[point_idx, coord_idx] = old_val + np.random.randn() * step_size
            new_score = _fast_metric(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                step_size = max(step_size * 0.95, 0.001)
            else:
                current_pts[point_idx, coord_idx] = old_val
                no_improve_count += 1
                step_size = min(step_size * 1.02, 0.3)
        
        _try_candidate(current_pts)
    
    # --- Second polishing pass with smaller steps ---
    if best_points is not None:
        current_pts = best_points.copy()
        current_score = best_score
        step_size = 0.01
        
        for iter in range(20000):
            if time.time() > deadline - 5:
                break
            
            dim_idx = np.random.randint(n * d)
            point_idx = dim_idx // d
            coord_idx = dim_idx % d
            old_val = current_pts[point_idx, coord_idx]
            current_pts[point_idx, coord_idx] = old_val + np.random.randn() * step_size
            new_score = _fast_metric(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                step_size = max(step_size * 0.98, 0.0001)
            else:
                current_pts[point_idx, coord_idx] = old_val
                step_size = min(step_size * 1.01, 0.05)
        
        _try_candidate(current_pts)
    
    # Final verification
    if best_points is None:
        phi = (1 + 5**0.5) / 2
        ico = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=np.float64)
        ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
        ico = ico / ico_norms
        best_points = np.vstack([ico, np.array([[0, 0, 1]], dtype=np.float64), np.array([[0, 0, -1]], dtype=np.float64)])
    
    best_points = best_points.reshape(n, d)
    if not np.all(np.isfinite(best_points)):
        phi = (1 + 5**0.5) / 2
        ico = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=np.float64)
        ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
        ico = ico / ico_norms
        best_points = np.vstack([ico, np.array([[0, 0, 1]], dtype=np.float64), np.array([[0, 0, -1]], dtype=np.float64)])
    
    return best_points

# EVOLVE-BLOCK-END
