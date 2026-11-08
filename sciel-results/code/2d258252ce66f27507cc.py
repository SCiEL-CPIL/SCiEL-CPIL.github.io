"""
Max-min triangle area optimization for 14 points in unit-area convex region.

Strategy: maximize a smooth log-sum-exp lower bound on min triangle area / hull area
using L-BFGS-B with analytical gradients from multiple diverse initializations,
then polish with coordinate-descent on the exact objective.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 143.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _hull_area_grad(pts):
    """
    Numerical gradient of convex hull area w.r.t. point coordinates.
    Returns array of shape (n, 2).
    """
    n = len(pts)
    grad = np.zeros((n, 2))
    eps = 1e-7
    for i in range(n):
        for d in range(2):
            pts_p = pts.copy()
            pts_p[i, d] += eps
            pts_m = pts.copy()
            pts_m[i, d] -= eps
            grad[i, d] = (_hull_area(pts_p) - _hull_area(pts_m)) / (2 * eps)
    return grad


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _triangle_areas_matrix(points):
    """
    Compute all triangle areas (2*area = |cross product|) as a flat array.
    Returns array of shape (364,) for N=14.
    Also returns the (i,j,k) index triplets.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    triples = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                cross = (x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i])
                areas.append(abs(cross))
                triples.append((i, j, k))
    return np.array(areas), triples


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3 = 364)."""
    areas, _ = _triangle_areas_matrix(points)
    return 0.5 * areas


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def _lse_obj_and_grad(pts_flat, beta):
    """
    Log-sum-exp based objective: minimize hull_area / soft_min(triangle_areas)
    where soft_min uses log-sum-exp approximation.
    
    Returns (objective_value, gradient) for L-BFGS-B.
    """
    points = pts_flat.reshape(N, 2)
    if np.any(~np.isfinite(points)):
        return 1e10, np.zeros_like(pts_flat)
    
    areas2, triples = _triangle_areas_matrix(points)
    # areas2 = 2*|cross|, so actual area = 0.5 * areas2
    # We work with areas2 for convenience (factor of 2 cancels in ratio)
    
    if np.min(areas2) < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    
    # soft_min via log-sum-exp: soft_min(a) = -(1/beta) * log(sum(exp(-beta * a)))
    z = -beta * areas2
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return 1e10, np.zeros_like(pts_flat)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10, np.zeros_like(pts_flat)
    
    obj_val = hull_area / soft_min_val
    
    # Gradient computation
    # d(obj)/d(pts) = (d(hull)/d(pts) * soft_min - hull * d(soft_min)/d(pts)) / soft_min^2
    
    # d(soft_min)/d(pts_k) where soft_min = -(1/beta)*log(sum(exp(-beta*a_m)))
    # = (1/beta) * sum_m [ (beta * exp(-beta*a_m)/sum(exp(-beta*a_m))) * da_m/d(pts_k) ]
    # = sum_m [ p_m * da_m/d(pts_k) ] where p_m = exp(-beta*a_m)/sum(exp(-beta*a_m))
    
    exp_z = np.exp(z - max_z)
    p = exp_z / np.sum(exp_z)  # softmax weights
    
    # da_m/d(pts_k) for each triangle m and point k
    # For triangle (i,j,k): a = |(x_j-x_i)(y_k-y_i) - (x_k-x_i)(y_j-y_i)|
    # da/dx_k = sign(cross) * (-(y_j - y_i))  [derivative w.r.t. x_k of cross]
    # da/dy_k = sign(cross) * (x_j - x_i)
    # da/dx_i = sign(cross) * (y_k - y_j)
    # da/dy_i = sign(cross) * (-(x_k - x_j))
    # da/dx_j = sign(cross) * (-(y_k - y_i))
    # da/dy_j = sign(cross) * (x_k - x_i)
    
    grad_soft_min = np.zeros((N, 2))
    x = points[:, 0]
    y = points[:, 1]
    
    for idx, (i, j, k) in enumerate(triples):
        if p[idx] < 1e-15:
            continue
        cross = (x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i])
        sign_c = np.sign(cross) if abs(cross) > 1e-15 else 1.0
        w = p[idx]
        
        # d(cross)/d(x_i) = -(y_k - y_j)
        grad_soft_min[i, 0] += w * sign_c * (-(y[k] - y[j]))
        # d(cross)/d(y_i) = -(x_k - x_j)
        grad_soft_min[i, 1] += w * sign_c * (-(x[k] - x[j]))
        # d(cross)/d(x_j) = (y_k - y_i)
        grad_soft_min[j, 0] += w * sign_c * (y[k] - y[i])
        # d(cross)/d(y_j) = -(x_k - x_i)
        grad_soft_min[j, 1] += w * sign_c * (-(x[k] - x[i]))
        # d(cross)/d(x_k) = -(y_j - y_i)
        grad_soft_min[k, 0] += w * sign_c * (-(y[j] - y[i]))
        # d(cross)/d(y_k) = (x_j - x_i)
        grad_soft_min[k, 1] += w * sign_c * (x[j] - x[i])
    
    # d(hull_area)/d(pts) - use numerical
    grad_hull = _hull_area_grad(points)
    
    # obj = hull / soft_min
    # d(obj)/d(pts) = (grad_hull * soft_min - hull * grad_soft_min) / soft_min^2
    grad_obj = (grad_hull * soft_min_val - hull_area * grad_soft_min) / (soft_min_val ** 2)
    
    return obj_val, grad_obj.flatten()


def _soft_obj(pts_flat, beta=100.0):
    """Soft objective: hull_area / soft_min(triangle areas)."""
    val, _ = _lse_obj_and_grad(pts_flat, beta)
    return val


def _optimize_lse_lbfgsb(init_pts, time_budget, beta_schedule=None):
    """
    Optimize using L-BFGS-B with log-sum-exp objective and analytical gradients.
    Uses a schedule of increasing beta values for progressive refinement.
    """
    if beta_schedule is None:
        beta_schedule = [20, 50, 100, 200, 400, 800]
    
    start_time = time.time()
    x = init_pts.flatten()
    best_x = x.copy()
    best_score = _score(init_pts)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_budget - 3:
            break
        
        def obj_func(x_flat):
            val, grad = _lse_obj_and_grad(x_flat, beta)
            return val, grad
        
        try:
            res = minimize(obj_func, x, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                                  'maxfun': 500, 'maxls': 50})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_x = res.x
                x = res.x
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)


def _optimize_full_points(time_budget, init_pts):
    """
    Optimize all 28 coordinates using L-BFGS-B with LSE objective,
    then polish with coordinate descent on exact score.
    """
    start_time = time.time()
    
    # Phase 1: L-BFGS-B with LSE
    pts = _optimize_lse_lbfgsb(init_pts, time_budget * 0.6)
    best_score = _score(pts)
    
    # Phase 2: Coordinate descent polish on exact score
    if time.time() - start_time < time_budget - 2:
        pts = _coordinate_descent_polish(pts, time_budget - (time.time() - start_time) - 1)
        s = _score(pts)
        if s > best_score:
            best_score = s
    
    return pts


def _coordinate_descent_polish(pts, time_budget):
    """
    Coordinate descent: perturb each point slightly in random directions
    to escape local minima of the exact min-triangle score.
    """
    start_time = time.time()
    best = pts.copy()
    best_score = _score(best)
    rng = np.random.RandomState(12345)
    
    step_size = 0.05
    for iteration in range(2000):
        if time.time() - start_time > time_budget:
            break
        if step_size < 1e-6:
            break
        
        # Pick a random point and random direction
        i = rng.randint(N)
        direction = rng.randn(2)
        direction /= np.linalg.norm(direction)
        
        for sign in [1, -1]:
            cand = best.copy()
            cand[i] += sign * step_size * direction
            s = _score(cand)
            if s > best_score:
                best_score = s
                best = cand
                break
        else:
            # No improvement, reduce step
            step_size *= 0.9
    
    return best


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_two_ring(n, inner_radius=0.5):
    """Two-ring initialization: 10 outer + 4 inner."""
    outer_n = 10
    inner_n = n - outer_n
    outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = inner_radius * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _initial_perturbed_polygon(n, seed=42, noise_scale=0.05):
    """Regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    angles = 2 * np.pi * np.arange(n) / n + rng.normal(0, noise_scale, n)
    radii = 1.0 + rng.normal(0, noise_scale, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_golden_spiral(n, seed=42):
    """Points on a golden-angle spiral, giving good coverage."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    angles = np.arange(n) * golden_angle
    radii = np.linspace(0.3, 1.0, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed_rings(n, seed=42):
    """Three-ring initialization: 7 outer, 4 middle, 3 inner."""
    outer_n, mid_n, inner_n = 7, 4, 3
    outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n
    mid_angles = 2 * np.pi * np.arange(mid_n) / mid_n + np.pi / mid_n
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    mid = 0.65 * np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])
    inner = 0.35 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, mid, inner])
    return _normalize_unit_area(pts)


def _initial_random_convex(n, seed=42):
    """Random points sorted by angle (convex position)."""
    rng = np.random.RandomState(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.7, 1.0, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return _normalize_unit_area(pts)


def _global_search_fast(time_budget):
    """
    Fast global search: evaluate many diverse initializations and keep the best.
    Uses structured + random configurations without expensive DE.
    """
    start_time = time.time()
    candidates = []
    
    # Structured initializations
    candidates.append(_initial_regular_polygon(N))
    candidates.append(_initial_two_ring(N, 0.4))
    candidates.append(_initial_two_ring(N, 0.5))
    candidates.append(_initial_two_ring(N, 0.6))
    candidates.append(_initial_mixed_rings(N))
    candidates.append(_initial_golden_spiral(N))
    
    # Perturbed polygons with various noise levels
    for seed in range(1, 15):
        for ns in [0.03, 0.08, 0.15]:
            candidates.append(_initial_perturbed_polygon(N, seed=seed, noise_scale=ns))
    
    # Random convex configurations
    for seed in range(1, 10):
        candidates.append(_initial_random_convex(N, seed=seed))
    
    best_pts = None
    best_score = 0.0
    for cand in candidates:
        if time.time() - start_time > time_budget - 2:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    
    # Phase 1: Fast global search over diverse initializations
    global_time = min(10.0, (DEADLINE - start_time) * 0.10)
    best_pts, best_score = _global_search_fast(global_time)
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
        best_score = _score(best_pts)
    
    # Phase 2: L-BFGS-B optimization from top initializations
    # Collect diverse starting points
    rng = np.random.RandomState(99)
    inits_list = [best_pts]
    
    # Add more diverse inits
    for seed in range(1, 8):
        inits_list.append(_initial_perturbed_polygon(N, seed=seed, noise_scale=0.1))
    inits_list.append(_initial_two_ring(N, 0.45))
    inits_list.append(_initial_mixed_rings(N))
    inits_list.append(_initial_random_convex(N, seed=77))
    
    # Evaluate and sort
    scored_inits = []
    for init in inits_list:
        s = _score(init)
        scored_inits.append((s, init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Deduplicate: keep only inits within 0.02 of best
    unique_inits = [scored_inits[0][1]]
    for s, init in scored_inits[1:]:
        if s > best_score - 0.02:
            unique_inits.append(init)
    
    # Optimize from top 5 initializations
    remaining = DEADLINE - 15 - (time.time() - start_time)
    if remaining > 10:
        n_opt = min(5, len(unique_inits))
        time_per_opt = remaining / n_opt * 0.85
        
        for idx in range(n_opt):
            if time.time() - start_time > DEADLINE - 15:
                break
            time_left = DEADLINE - 15 - (time.time() - start_time)
            if time_left < 8:
                break
            
            try:
                cand = _optimize_full_points(time_left * 0.9, unique_inits[idx])
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
            except Exception:
                pass
    
    # Phase 3: Final polish with high-beta LSE + coordinate descent
    if time.time() - start_time < DEADLINE - 3:
        polish_time = min(12.0, DEADLINE - 4 - (time.time() - start_time))
        if polish_time > 3:
            try:
                # High-beta LSE polish
                def final_lse(x):
                    val, grad = _lse_obj_and_grad(x, 2000.0)
                    return val, grad
                
                res = minimize(final_lse, best_pts.flatten(), method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10,
                                      'maxfun': 200})
                if res.x is not None:
                    cand = res.x.reshape(-1, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
                
                # Coordinate descent polish
                if time.time() - start_time < DEADLINE - 2:
                    cd_time = DEADLINE - 3 - (time.time() - start_time)
                    if cd_time > 1:
                        cand = _coordinate_descent_polish(best_pts, cd_time)
                        s = _score(cand)
                        if s > best_score:
                            best_score = s
                            best_pts = cand
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
