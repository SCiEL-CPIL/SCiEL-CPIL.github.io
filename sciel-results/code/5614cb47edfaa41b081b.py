# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _get_triangle_indices(n: int):
    """Precompute triangle index arrays for n points."""
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n),
        indexing='ij'
    )
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]


def _compute_score(points: np.ndarray, tri_indices=None) -> float:
    """Compute A_min / A_hull (true, nonsmooth)."""
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

    if tri_indices is None:
        tri_indices = _get_triangle_indices(n)
        
    i_idx, j_idx, k_idx = tri_indices
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    min_area = float(np.min(0.5 * cross))
    if min_area <= 0:
        return 0.0
    return min_area / A_hull


def _softmin_grad(points: np.ndarray, tri_indices, alpha: float):
    """
    Gradient of the soft-min objective w.r.t. each point coordinate.
    Returns (grad, value) where grad has shape (n, 2).
    The objective is the soft-min of the triangle areas.
    """
    n = points.shape[0]
    i_idx, j_idx, k_idx = tri_indices
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    m = np.min(areas)
    if m <= 0:
        return np.zeros((n, 2)), -1e9
    
    # Soft min: m - log(sum(exp(-alpha * (areas - m)))) / alpha
    # To avoid overflow, we shift by m
    exp_terms = np.exp(-alpha * (areas - m))
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        return np.zeros((n, 2)), -1e9
        
    weights = exp_terms / sum_exp
    val = m - np.log(sum_exp) / alpha

    # Per-triangle gradient of area = 0.5 * |v1 x v2|
    # raw = v1x * v2y - v1y * v2x
    raw = v1x * v2y - v1y * v2x
    sign = np.sign(raw)
    sign[sign == 0] = 1.0

    # d(area)/d(pi) = -0.5 * sign * (v2y, -v2x) + 0.5 * sign * (v1y, -v1x) ??? 
    # Let's re-derive carefully.
    # Area A = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    # Let D = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # A = 0.5 * sign(D) * D
    # dA/dx_i = 0.5 * sign(D) * (dD/dx_j * dx_j/dx_i + ...) -- No, D is function of all.
    # dD/dx_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
    # dD/dy_i = -(x_k - x_i) + (x_j - x_i) = x_j - x_k
    # dD/dx_j = (y_k - y_i)
    # dD/dy_j = -(x_k - x_i)
    # dD/dx_k = -(y_j - y_i)
    # dD/dy_k = (x_j - x_i)
    
    # So:
    # dA/dx_i = 0.5 * sign * (y_j - y_k)
    # dA/dy_i = 0.5 * sign * (x_j - x_k)
    # dA/dx_j = 0.5 * sign * (y_k - y_i)
    # dA/dy_j = 0.5 * sign * (-(x_k - x_i)) = 0.5 * sign * (x_i - x_k)
    # dA/dx_k = 0.5 * sign * (-(y_j - y_i)) = 0.5 * sign * (y_i - y_j)
    # dA/dy_k = 0.5 * sign * (x_j - x_i)

    darea_dpi = np.column_stack([
        0.5 * sign * (pj[:, 1] - pk[:, 1]),
        0.5 * sign * (pj[:, 0] - pk[:, 0])
    ])
    darea_dpj = np.column_stack([
        0.5 * sign * (pk[:, 1] - pi[:, 1]),
        0.5 * sign * (pi[:, 0] - pk[:, 0])
    ])
    darea_dpk = np.column_stack([
        0.5 * sign * (pi[:, 1] - pj[:, 1]),
        0.5 * sign * (pj[:, 0] - pi[:, 0])
    ])

    grad = np.zeros((n, 2))
    np.add.at(grad, i_idx, weights[:, None] * darea_dpi)
    np.add.at(grad, j_idx, weights[:, None] * darea_dpj)
    np.add.at(grad, k_idx, weights[:, None] * darea_dpk)
    
    return grad, val


def _gradient_ascent(points: np.ndarray, deadline: float, alpha: float = 50.0, tri_indices=None) -> np.ndarray:
    """
    Gradient ascent on the soft-min objective, with periodic true-score tracking.
    """
    if tri_indices is None:
        tri_indices = _get_triangle_indices(len(points))
        
    best = points.copy()
    best_score = _compute_score(best, tri_indices)
    cur = points.copy()
    n = len(cur)
    
    lr = 0.005
    min_lr = 1e-7
    stall = 0
    max_stall = 50
    
    while time.time() < deadline:
        grad, soft_val = _softmin_grad(cur, tri_indices, alpha)
        if not np.all(np.isfinite(grad)):
            break
            
        # Gradient ascent step
        step = cur + lr * grad
        # Keep points within a reasonable box to prevent divergence
        # The problem is scale invariant, but we fix the hull area implicitly 
        # by not normalizing. We just optimize relative positions.
        # However, if points fly apart, hull area grows, min area might grow too.
        # The ratio is what matters.
        # Let's clip to [0, 1] for stability, though strictly not required if we normalize.
        # Actually, the score is scale invariant. 
        # If we clip, we might restrict the search. 
        # Better: normalize after step? No, gradient depends on scale.
        # Let's just clip to a large box to avoid inf/nan.
        np.clip(step, -10, 10, out=step)

        cur_score = _compute_score(cur, tri_indices)
        step_score = _compute_score(step, tri_indices)
        
        if step_score >= cur_score:
            cur = step
            if step_score > best_score:
                best_score = step_score
                best = cur.copy()
                stall = 0
            else:
                stall += 1
        else:
            # Accept small worsening sometimes to escape local minima of the surrogate
            if np.random.random() < 0.1:
                cur = step
            stall += 1

        if stall >= max_stall:
            lr *= 0.8
            stall = 0
            if lr < min_lr:
                break
        elif stall == 0:
            # Slowly increase lr when making progress
            lr = min(lr * 1.02, 0.02)

    return best


def _build_hex_configuration(r1: float, r2: float, center: tuple) -> np.ndarray:
    """1 center + 6 on inner hexagon + 6 on outer hexagon = 13 points."""
    cx, cy = center
    points = [[cx, cy]]
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    return np.array(points)


def _build_random_config(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Random points in a square
    return rng.random((n, 2))


def _simulated_annealing(points: np.ndarray, deadline: float,
                         T_init: float = 0.01, T_min: float = 1e-9,
                         cooling_rate: float = 0.9995, max_iters: int = 20000, tri_indices=None) -> np.ndarray:
    best = points.copy()
    cur = points.copy()
    cur_score = _compute_score(cur, tri_indices)
    best_score = cur_score
    n = len(cur)

    step_size = 0.05
    min_step = 1e-8
    accept = 0
    reject = 0

    for it in range(max_iters):
        if time.time() > deadline:
            break
        T = max(T_init * (cooling_rate ** it), T_min)
        idx = np.random.randint(n)
        dx = np.random.randn() * step_size
        dy = np.random.randn() * step_size
        cand = cur.copy()
        cand[idx, 0] += dx
        cand[idx, 1] += dy
        cand_score = _compute_score(cand, tri_indices)
        
        if cand_score >= cur_score:
            cur = cand
            cur_score = cand_score
            accept += 1
            if cur_score > best_score:
                best_score = cur_score
                best = cand.copy()
        else:
            delta = cur_score - cand_score
            if np.random.random() < np.exp(-delta / max(T, 1e-12)):
                cur = cand
                cur_score = cand_score
                accept += 1
            else:
                reject += 1
                
        if (accept + reject) % 500 == 0 and (accept + reject) > 0:
            rate = accept / (accept + reject)
            if rate < 0.2:
                step_size *= 0.9
            elif rate > 0.8:
                step_size *= 1.1
            step_size = max(min_step, min(step_size, 0.1))
            accept = 0
            reject = 0
            
    return best


def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    # Evaluator kills at 630s. We aim to finish well before.
    deadline = start_time + 600 
    
    tri_indices = _get_triangle_indices(n)
    
    best_points = None
    best_score = 0.0

    # Phase 1: Generate diverse initial configurations
    configs = []
    # Hexagonal lattices are good starts
    for r1 in [0.2, 0.25, 0.3]:
        for r2 in [0.5, 0.6, 0.7]:
            configs.append(_build_hex_configuration(r1, r2, (0.5, 0.5)))
            
    # Random starts
    for seed in range(10):
        configs.append(_build_random_config(n, seed))
        
    # Sort by initial score to prioritize good basins
    scored_configs = []
    for cfg in configs:
        s = _compute_score(cfg, tri_indices)
        scored_configs.append((s, cfg))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Phase 2: Simulated Annealing on top configs
    # Allocate ~200s for SA
    sa_deadline = start_time + 200
    top_configs = [cfg for _, cfg in scored_configs[:5]]
    
    for cfg in top_configs:
        if time.time() > sa_deadline:
            break
        sa_result = _simulated_annealing(cfg, sa_deadline, tri_indices=tri_indices)
        s = _compute_score(sa_result, tri_indices)
        if s > best_score:
            best_score = s
            best_points = sa_result.copy()

    # Phase 3: Gradient Ascent on the best point
    # Allocate ~250s for Gradient Ascent
    if best_points is not None:
        ga_deadline = start_time + 450
        # Try different alpha values for soft-min
        for alpha in [30.0, 50.0, 80.0]:
            if time.time() > ga_deadline:
                break
            ga_result = _gradient_ascent(best_points, ga_deadline, alpha=alpha, tri_indices=tri_indices)
            s = _compute_score(ga_result, tri_indices)
            if s > best_score:
                best_score = s
                best_points = ga_result.copy()

    # Phase 4: Final Polish with Nelder-Mead on True Score
    # Allocate ~100s for Polish
    if best_points is not None and time.time() < deadline - 50:
        nm_deadline = deadline - 30
        
        def objective(flat_points):
            pts = flat_points.reshape(n, 2)
            s = _compute_score(pts, tri_indices)
            if s == 0.0:
                return 1e9
            return -s

        x0 = best_points.reshape(-1)
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.success:
                new_pts = res.x.reshape(n, 2)
                if np.all(np.isfinite(new_pts)):
                    new_score = _compute_score(new_pts, tri_indices)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
        except Exception:
            pass

    # Safety check
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _build_hex_configuration(0.25, 0.6, (0.5, 0.5))

    # Normalize to [0,1] box for consistency (though score is invariant)
    # This ensures no negative coordinates which might confuse some evaluators
    min_coords = np.min(best_points, axis=0)
    best_points = best_points - min_coords
    max_coords = np.max(best_points, axis=0)
    if np.any(max_coords > 0):
        best_points = best_points / max_coords
        
    return best_points


# EVOLVE-BLOCK-END
